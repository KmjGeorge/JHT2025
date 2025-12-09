import struct
import numpy as np
import h5py
import os

# 单个脉冲的PDW
class PDW:
    def __init__(self, freq, pa, pw, toa):
        self.freq = freq
        self.pa = pa
        self.pw = pw
        self.toa = toa


def read_pdw(path):
    pdws = []
    with h5py.File(path, 'r') as f:
        PAs = np.array(f['PA']).squeeze()
        Freqs = np.array(f['Freq']).squeeze() * 1e3
        PW = np.array(f['PW']).squeeze() * 1e3
        TOA = np.array(f['TOA']).squeeze()

        min_toa = np.min(TOA)

        if min_toa < 0:
            TOA += abs(min_toa)

        for pa, freq, pw, toa in zip(PAs, Freqs, PW, TOA):
            pdw = PDW(freq, pa, pw, toa)
            pdws.append(pdw)

    return pdws


class PDW_binary:
    def __init__(self, pa, pw, toa, freq):
        self.pdw_head = 0x5a5a
        self.pa = pa
        self.pw = pw
        if toa < 0:
            toa = 0
        self.toa = toa
        self.rf = freq
        self.rf_sec = 0x0000
        self.pa_1 = 0x0000
        self.pa_2 = 0x0000
        self.pa_3 = 0x0000
        self.pa_4 = 0x0000
        self.df_en = 0x0000

    def to_binary(self):
        pack_head = struct.pack('<H', self.pdw_head)
        pack_pa = struct.pack('<H', int((self.pa + 100) / 0.01))
        pack_pw = struct.pack('<I', int(self.pw))
        pack_toa = int(self.toa).to_bytes(8, byteorder='little', signed=False)
        pack_remain = struct.pack('<IHHHHHH', int(self.rf), self.rf_sec, self.pa_1, self.pa_2, self.pa_3, self.pa_4, self.df_en)
        pack = pack_head + pack_pa + pack_pw + pack_toa + pack_remain
        return pack


if __name__ == '__main__':
    root = r''
    save_root = r''
    for filename in os.listdir(root):
        pdws = read_pdw(os.path.join(root, filename))
        with open(os.path.join(save_root, filename.replace('.h5', '.bin')), 'wb') as f:
            for pdw in pdws:
                pdw_binary = PDW_binary(pdw.pa, pdw.pw, pdw.toa, pdw.freq)
                f.write(pdw_binary.to_binary())