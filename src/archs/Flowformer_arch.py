import torch
import torch.nn as nn
import torch.nn.functional as F

from src.archs.rope import RotaryEmbedding
from src.utils.registry import ARCH_REGISTRY
import math


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1, use_pe=False):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.use_pe = use_pe
        if self.use_pe:
            self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x)
        if self.use_pe:
            x = x + self.position_embedding(x)
        return self.dropout(x)


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x

class ChannelAttention1D(nn.Module):
    """
    Args:
        channel (int): 输入通道数 C
        b (int, optional): 自适应卷积核大小公式中的超参数b，默认为1。
        gamma (int, optional): 自适应卷积核大小公式中的超参数gamma，默认为2。
    """
    def __init__(self, c_in, b=1, gamma=2):
        super(ChannelAttention1D, self).__init__()
        # 自适应计算一维卷积的核大小k：k = |(log2(C) + b) / gamma|，并确保为奇数
        kernel_size = int(abs((math.log(c_in, 2) + b) / gamma))
        # self.conv = nn.Conv1d(in_channels=c_in,
        #             out_channels=c_in,
        #             kernel_size=3,
        #             padding=2,
        #             padding_mode='circular')
        if kernel_size % 2 == 0:
            kernel_size += 1
        padding = kernel_size // 2

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # 使用一维卷积替代全连接层，输入和输出通道数均为1，在通道维度上进行卷积
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入张量 [B, L, C]
        Returns:
            torch.Tensor: 加权后的张量 [B, L, C]
        """
        y = self.avg_pool(x.transpose(1, 2)).transpose(1, 2)   # (B, 1, C)
        y = self.conv(y)
        y = self.sigmoid(y)
        return x * y

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, freqs_cis=None, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x, freqs_cis=freqs_cis,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, freqs_cis=None, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, freqs_cis=freqs_cis, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, freqs_cis=None, attn_mask=None, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)

        # apply RoPE
        if freqs_cis is not None:
            queries, keys = apply_rotary_emb(queries, keys, freqs_cis=freqs_cis)

        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class FullAttention(nn.Module):
    def __init__(self, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    """为RoPE预计算旋转矩阵 """
    # 计算词向量元素两两分组之后，每组元素对应的旋转角度
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 生成 token 序列索引 t = [0, 1,..., seq_len-1]
    t = torch.arange(seq_len, device=freqs.device)
    # freqs.shape = [seq_len, dim // 2]
    freqs = torch.outer(t, freqs).float()
    # torch.polar 的文档
    # https://pytorch.org/docs/stable/generated/torch.polar.html
    # 计算结果是个复数向量
    # 假设 freqs = [x, y]
    # 则 freqs_cis = [cos(x) + sin(x)i, cos(y) + sin(y)i]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(xq, xk, freqs_cis):  # xq [batch_size, seq_len, head, dim // head]
    """ 应用RoPE """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    # print(freqs_cis.shape)
    def reshape_for_broadcast(freqs_cis, x):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert freqs_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)

    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class SymmetricT5RelativeBias(nn.Module):
    """基于T5的相对位置编码的对称版本"""
    def __init__(self, num_heads, num_buckets=32, max_distance=128):
        super().__init__()
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance

        # 相对位置偏置参数
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

    def _relative_position_bucket(self, relative_position):
        """将相对位置映射到桶中（对称版本）"""
        relative_position = torch.abs(relative_position)  # 关键：使用绝对值确保对称

        # 近距离使用精细分桶，远距离使用粗糙分桶
        max_exact = self.num_buckets // 2
        is_small = relative_position < max_exact

        # 大距离使用对数分桶
        relative_position_if_large = max_exact + (
                torch.log(relative_position.float() / max_exact) /
                math.log(self.max_distance / max_exact) * (self.num_buckets - max_exact)
        ).long()

        relative_position_if_large = torch.minimum(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, self.num_buckets - 1)
        )

        return torch.where(is_small, relative_position, relative_position_if_large)

    def forward(self, seq_len):
        """生成对称相对位置偏置"""
        context_pos = torch.arange(seq_len, dtype=torch.long)[:, None]
        memory_pos = torch.arange(seq_len, dtype=torch.long)[None, :]
        relative_pos = memory_pos - context_pos

        # 使用绝对值确保对称性
        relative_pos = torch.abs(relative_pos)

        # 分桶
        rp_bucket = self._relative_position_bucket(relative_pos)

        # 获取偏置
        values = self.relative_attention_bias(rp_bucket)  # [seq_len, seq_len, num_heads]
        values = values.permute(2, 0, 1).unsqueeze(0)  # [1, num_heads, seq_len, seq_len]

        return values


class BidirectionalALiBi(nn.Module):
    """
    双向ALiBi位置编码
    特点：
    1. 基于距离的线性偏置（ALiBi核心思想）
    2. 使用绝对距离确保双向对称性
    3. 支持任意长度序列外推
    """

    def __init__(self, num_heads, max_slope=1.0, min_slope=0.01):
        """
        初始化双向ALiBi

        参数:
            num_heads: 注意力头的数量
            max_slope: 最大斜率值（最近距离）
            min_slope: 最小斜率值（最远距离）
        """
        super().__init__()
        self.num_heads = num_heads

        # 为每个头生成不同的斜率（几何序列）
        slopes = torch.tensor(
            [min_slope * (max_slope / min_slope) ** (i / (num_heads - 1))
             for i in range(num_heads)]
        )
        self.register_buffer('slopes', slopes)

        # 斜率参数（可学习）
        self.slope_params = nn.Parameter(torch.ones(num_heads))

    def forward(self, seq_len):
        """
        生成双向ALiBi位置偏置矩阵

        返回:
            bias: 位置偏置矩阵，形状为 [1, num_heads, seq_len, seq_len]
        """
        # 创建位置索引
        pos = torch.arange(seq_len, dtype=torch.float)

        # 计算绝对距离矩阵 |i - j|
        # 使用绝对距离确保双向对称性
        distance = torch.abs(pos[:, None] - pos[None, :])

        # 计算偏置矩阵
        # 使用广播机制为每个头应用不同的斜率
        slopes = self.slopes * self.slope_params
        bias = -slopes.view(1, -1, 1, 1) * distance

        # 调整维度顺序 [1, num_heads, seq_len, seq_len]
        return bias.permute(1, 0, 2).unsqueeze(0)
class FlowAttention(nn.Module):
    def __init__(self, attention_dropout=0.1):
        super(FlowAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)

    def kernel_method(self, x):
        return torch.sigmoid(x)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # kernel
        queries = self.kernel_method(queries)
        keys = self.kernel_method(keys)
        # incoming and outgoing
        normalizer_row = 1.0 / (torch.einsum("nhld,nhd->nhl", queries + 1e-6, keys.sum(dim=2) + 1e-6))
        normalizer_col = 1.0 / (torch.einsum("nhsd,nhd->nhs", keys + 1e-6, queries.sum(dim=2) + 1e-6))
        # reweighting
        normalizer_row_refine = (
            torch.einsum("nhld,nhd->nhl", queries + 1e-6, (keys * normalizer_col[:, :, :, None]).sum(dim=2) + 1e-6))
        normalizer_col_refine = (
            torch.einsum("nhsd,nhd->nhs", keys + 1e-6, (queries * normalizer_row[:, :, :, None]).sum(dim=2) + 1e-6))
        # competition and allocation
        normalizer_row_refine = torch.sigmoid(
            normalizer_row_refine * (float(queries.shape[2]) / float(keys.shape[2])))
        normalizer_col_refine = torch.softmax(normalizer_col_refine, dim=-1) * keys.shape[2]  # B h L vis
        # multiply
        kv = keys.transpose(-2, -1) @ (values * normalizer_col_refine[:, :, :, None])
        x = (((queries @ kv) * normalizer_row[:, :, :, None]) * normalizer_row_refine[:, :, :, None]).transpose(1,
                                                                                                                2).contiguous()
        return x, None



@ARCH_REGISTRY.register()
class Flowformer(nn.Module):
    """
    modified from https://github.com/thuml/Flowformer

    pe_mode (str) : optional 'Symmetric_Relative', 'Bidirectional_ALiBi', 'RoPE' or 'Absolute'
    """

    def __init__(self, seq_len, enc_in, c_out, d_model, dropout, n_heads, d_ff, activation, e_layers, pe_mode='ALiBi'):
        super(Flowformer, self).__init__()
        self.pred_len = seq_len

        self.enc_in = enc_in
        self.c_out = c_out
        self.pe_mode = pe_mode

        if self.pe_mode == 'Absolute':
            self.enc_embedding = DataEmbedding(self.enc_in, d_model, dropout, True)
        else:
            self.enc_embedding = DataEmbedding(self.enc_in, d_model, dropout, False)
            # if self.pe_mode == 'Symmetric_Relative':
            #     self.pos_encoder = SymmetricT5RelativeBias(num_heads)
            # elif self.pe_mode == 'Bidirectional_ALiBi':
            #     self.pos_encoder = BidirectionalALiBI(num_heads)
            if self.pe_mode == 'RoPE':
                # self.rope = RotaryEmbedding(dim=d_model // n_heads, freqs_for='lang', learned_freq=True, cache_if_possible=True, cache_max_seq_len=seq_len * 4)
                self.freqs_cis = precompute_freqs_cis(d_model // n_heads, seq_len * 2)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FlowAttention(attention_dropout=dropout), d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ChannelAttention1D(c_in=d_model) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        # decoder
        self.projection = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, c_out))

    def forecast(self, x_enc):
        seq_len = x_enc.shape[1]
        # Embedding
        enc_out = self.enc_embedding(x_enc)
        if self.pe_mode == 'RoPE':
            self.freqs_cis = self.freqs_cis.to(x_enc.device)
            freqs_cis = self.freqs_cis[: seq_len]
        else:
            freqs_cis = None
        enc_out, attns = self.encoder(enc_out, freqs_cis, attn_mask=None)

        dec_out = self.projection(enc_out)
        return dec_out

    def forward(self, x_enc):

        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]



@ARCH_REGISTRY.register()
class Flowformer_P(nn.Module):
    """
    modified from https://github.com/thuml/Flowformer

    pe_mode (str) : optional 'Symmetric_Relative', 'Bidirectional_ALiBi', 'RoPE' or 'Absolute'
    """

    def __init__(self, seq_len, enc_in, c_out, d_model, dropout, n_heads, d_ff, activation, e_layers, pe_mode='ALiBi', label_num=15, prototype_num=32):
        super(Flowformer_P, self).__init__()
        self.pred_len = seq_len

        self.enc_in = enc_in
        self.c_out = c_out
        self.pe_mode = pe_mode
        self.prototype = nn.Parameter(torch.rand(label_num * prototype_num, d_model, requires_grad=True))
        self.prototype_projection = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, c_out))
        self.label_num = label_num
        self.prototype_num = prototype_num

        if self.pe_mode == 'Absolute':
            self.enc_embedding = DataEmbedding(self.enc_in, d_model, dropout, True)
        else:
            self.enc_embedding = DataEmbedding(self.enc_in, d_model, dropout, False)
            # if self.pe_mode == 'Symmetric_Relative':
            #     self.pos_encoder = SymmetricT5RelativeBias(num_heads)
            # elif self.pe_mode == 'Bidirectional_ALiBi':
            #     self.pos_encoder = BidirectionalALiBI(num_heads)
            if self.pe_mode == 'RoPE':
                # self.rope = RotaryEmbedding(dim=d_model // n_heads, freqs_for='lang', learned_freq=True, cache_if_possible=True, cache_max_seq_len=seq_len * 4)
                self.freqs_cis = precompute_freqs_cis(d_model // n_heads, seq_len * 2)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FlowAttention(attention_dropout=dropout), d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ChannelAttention1D(c_in=d_model) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        # decoder
        self.projection = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, c_out))

    def forecast(self, x_enc):
        seq_len = x_enc.shape[1]
        # Embedding
        enc_out = self.enc_embedding(x_enc)
        if self.pe_mode == 'RoPE':
            self.freqs_cis = self.freqs_cis.to(x_enc.device)
            freqs_cis = self.freqs_cis[: seq_len]
        else:
            freqs_cis = None
        enc_out, attns = self.encoder(enc_out, freqs_cis, attn_mask=None)

        dec_out = self.projection(enc_out)
        return dec_out

    def forward(self, x_enc):
        dec_out = self.forecast(x_enc)
        prototype = self.prototype_projection(self.prototype).chunk(self.label_num, dim=0)
        # print(len(prototype), prototype[0].shape)
        prototype_dict = {k: v for k, v in zip([i for i in range(self.label_num)], [prototype[i] for i in range(self.label_num)])}
        return dec_out[:, -self.pred_len:, :], prototype_dict  # [B, L, D]

if __name__ == '__main__':
    model = Flowformer_P(seq_len=1000,
                       enc_in=4,
                       d_model=128,
                       c_out=64,
                       n_heads=8,
                       d_ff=512,
                       activation='relu',
                       dropout=0.1,
                       e_layers=8,
                       label_num=15,
                       prototype_num=64,
                       pe_mode='RoPE').cuda()
    # x = torch.randn(8, 100, 3)
    # y = model(x)
    # print(y.shape)
    from torchsummary import summary

    # for i in range(1000):
    #     x = torch.randn(1, 300, 5).cuda()
    #     y = model(x)
    summary(model, input_data=torch.rand(3, 1000, 4))

