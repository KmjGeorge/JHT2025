import numpy as np
import matplotlib.pyplot as plt
from librosa.display import matplotlib
from matplotlib.widgets import RectangleSelector
from matplotlib.backend_bases import MouseButton
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import os

from src.data.data_util import read_pdw

plt.rcParams['font.family'] = 'SimHei'
matplotlib.use('TkAgg')
class InteractiveScatterPlot:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 7))
        self.fig.canvas.manager.set_window_title('交互式散点图工具')
        self.x_data = np.array([])
        self.y_data = np.array([])
        self.labels = np.array([])
        self.scatter = None
        self.rect_selector = None
        self.annotation = None
        self.original_limits = None
        self.zoom_history = []

        # 创建按钮区域
        self.ax_load = plt.axes([0.15, 0.01, 0.15, 0.05])
        self.ax_reset = plt.axes([0.35, 0.01, 0.15, 0.05])
        self.ax_zoom_out = plt.axes([0.55, 0.01, 0.15, 0.05])
        self.ax_export = plt.axes([0.75, 0.01, 0.15, 0.05])

        self.btn_load = plt.Button(self.ax_load, '加载数据')
        self.btn_reset = plt.Button(self.ax_reset, '重置视图')
        self.btn_zoom_out = plt.Button(self.ax_zoom_out, '返回上一视图')
        self.btn_export = plt.Button(self.ax_export, '导出图像')

        self.btn_load.on_clicked(self.load_data)
        self.btn_reset.on_clicked(self.reset_view)
        self.btn_zoom_out.on_clicked(self.zoom_out)
        self.btn_export.on_clicked(self.export_image)

        # 添加标题和说明
        self.fig.text(0.5, 0.95, '交互式散点图工具',
                      ha='center', fontsize=16, fontweight='bold')
        self.fig.text(0.5, 0.90,
                      '功能: 加载数据 | 点击查看坐标 | 框选放大 | 重置视图 | 返回上一视图 | 导出图像',
                      ha='center', fontsize=10, color='gray')

        # 初始化矩形选择器
        self.rect_selector = RectangleSelector(
            self.ax, self.on_rectangle_select,
            useblit=True,
            button=[MouseButton.LEFT],
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=False,
            rectprops=dict(alpha=0.3, facecolor='lightblue', edgecolor='blue')
        )

        # 连接点击事件
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        # 设置坐标轴标签
        self.ax.set_xlabel('X轴', fontsize=12)
        self.ax.set_ylabel('Y轴', fontsize=12)
        self.ax.grid(True, linestyle='--', alpha=0.7)

        # 显示初始提示
        self.ax.text(0.5, 0.5, '请点击"加载数据"按钮导入数据文件',
                     ha='center', va='center',
                     fontsize=12, color='gray',
                     transform=self.ax.transAxes)

        plt.tight_layout(rect=[0, 0.07, 1, 0.95])

    def load_data(self, event=None):
        """打开文件对话框加载数据"""
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title='选择数据文件',
            filetypes=[('h5文件', '*.h5'), ('所有文件', '*.*')]
        )
        root.destroy()

        if not file_path:
            return

        try:
            pdw = read_pdw(file_path)
            self.x_data = pdw.TOAdots
            self.y_data = pdw.Freqs
            self.plot_data()
            self.original_limits = (self.ax.get_xlim(), self.ax.get_ylim())

        except Exception as e:
            self.show_error("加载错误", f"无法加载文件: {str(e)}")

    def plot_data(self):
        """绘制散点图"""
        self.ax.clear()
        self.scatter = self.ax.scatter(
            self.x_data, self.y_data,
            c='blue', alpha=0.7,
            edgecolors='white', s=50
        )

        # 添加标签（只在点较少时显示）
        if len(self.x_data) <= 50:
            for i, (x, y) in enumerate(zip(self.x_data, self.y_data)):
                self.ax.text(x, y, self.labels[i], fontsize=8, ha='right', va='bottom')

        self.ax.set_xlabel('X轴', fontsize=12)
        self.ax.set_ylabel('Y轴', fontsize=12)
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.set_title(f'数据点: {len(self.x_data)}个', fontsize=12)

        # 添加数据统计信息
        stats_text = f"X范围: {min(self.x_data):.2f}-{max(self.x_data):.2f}\nY范围: {min(self.y_data):.2f}-{max(self.y_data):.2f}"
        self.ax.text(0.98, 0.02, stats_text,
                     transform=self.ax.transAxes,
                     ha='right', va='bottom',
                     fontsize=9, bbox=dict(facecolor='white', alpha=0.7))

        self.fig.canvas.draw_idle()

    def on_click(self, event):
        """处理点击事件"""
        if event.inaxes != self.ax or self.scatter is None:
            return

        if event.button == MouseButton.LEFT:
            # 查找最近的数据点
            distances = np.sqrt(
                (self.x_data - event.xdata) ** 2 +
                (self.y_data - event.ydata) ** 2
            )
            min_idx = np.argmin(distances)
            min_distance = distances[min_idx]

            # 如果点击位置接近某个点
            if min_distance < 0.05 * max(self.ax.get_xlim()[1] - self.ax.get_xlim()[0],
                                         self.ax.get_ylim()[1] - self.ax.get_ylim()[0]):
                # 移除之前的标注
                if self.annotation:
                    self.annotation.remove()

                # 创建新的标注
                x, y = self.x_data[min_idx], self.y_data[min_idx]
                label = self.labels[min_idx]

                self.annotation = self.ax.annotate(
                    f"{label}\n({x:.4f}, {y:.4f})",
                    xy=(x, y),
                    xytext=(20, 20),
                    textcoords='offset points',
                    arrowprops=dict(
                        arrowstyle="->",
                        connectionstyle="arc3,rad=0.2",
                        color='red'
                    ),
                    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", alpha=0.8)
                )

                # 高亮点
                self.scatter.set_edgecolors(['white'] * len(self.x_data))
                colors = self.scatter.get_facecolors()
                if len(colors) > 0:
                    colors = colors.copy()
                    colors[min_idx] = [1, 0, 0, 1]  # 红色
                    self.scatter.set_facecolors(colors)

                self.fig.canvas.draw_idle()

    def on_rectangle_select(self, eclick, erelease):
        """处理矩形选择事件（放大）"""
        if not self.scatter:
            return

        # 保存当前视图状态
        self.zoom_history.append((self.ax.get_xlim(), self.ax.get_ylim()))

        # 获取矩形坐标
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        # 设置新的坐标轴范围
        self.ax.set_xlim(min(x1, x2), max(x1, x2))
        self.ax.set_ylim(min(y1, y2), max(y1, y2))

        # 更新标题
        self.ax.set_title(f'局部放大视图 (数据点: {len(self.x_data)}个)', fontsize=12)

        self.fig.canvas.draw_idle()

    def reset_view(self, event=None):
        """重置视图到原始范围"""
        if self.original_limits:
            self.ax.set_xlim(self.original_limits[0])
            self.ax.set_ylim(self.original_limits[1])
            self.ax.set_title(f'数据点: {len(self.x_data)}个', fontsize=12)
            self.zoom_history = []
            self.fig.canvas.draw_idle()

    def zoom_out(self, event=None):
        """返回到上一个视图"""
        if self.zoom_history:
            prev_lim = self.zoom_history.pop()
            self.ax.set_xlim(prev_lim[0])
            self.ax.set_ylim(prev_lim[1])

            if not self.zoom_history:
                self.ax.set_title(f'数据点: {len(self.x_data)}个', fontsize=12)
            else:
                self.ax.set_title(f'局部放大视图 (数据点: {len(self.x_data)}个)', fontsize=12)

            self.fig.canvas.draw_idle()

    def export_image(self, event=None):
        """导出当前图像为文件"""
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.asksaveasfilename(
            title='保存图像',
            filetypes=[
                ('PNG图像', '*.png'),
                ('JPEG图像', '*.jpg'),
                ('PDF文档', '*.pdf'),
                ('SVG矢量图', '*.svg')
            ],
            defaultextension='.png'
        )
        root.destroy()

        if file_path:
            try:
                self.fig.savefig(file_path, dpi=300, bbox_inches='tight')
                # 显示保存成功消息
                if self.annotation:
                    self.annotation.remove()
                    self.annotation = None

                self.ax.set_title(f'图像已保存: {os.path.basename(file_path)}', fontsize=12, color='green')
                self.fig.canvas.draw_idle()

                # 2秒后恢复原始标题
                plt.pause(2)
                self.ax.set_title(f'数据点: {len(self.x_data)}个', fontsize=12, color='black')
                self.fig.canvas.draw_idle()

            except Exception as e:
                self.show_error("保存错误", f"无法保存图像: {str(e)}")

    def show_error(self, title, message):
        """显示错误消息"""
        if self.annotation:
            self.annotation.remove()
            self.annotation = None

        self.ax.set_title(title, fontsize=12, color='red')

        # 在图表下方显示错误消息
        error_text = self.ax.text(
            0.5, -0.15, message,
            transform=self.ax.transAxes,
            ha='center', va='top',
            fontsize=10, color='red',
            wrap=True
        )

        self.fig.canvas.draw_idle()

        # 3秒后清除错误消息
        plt.pause(3)
        error_text.remove()
        if self.scatter:
            self.ax.set_title(f'数据点: {len(self.x_data)}个', fontsize=12, color='black')
        else:
            self.ax.set_title('', fontsize=12)
        self.fig.canvas.draw_idle()

    def show(self):
        """显示图表"""
        plt.show()


# 创建并显示交互式散点图工具
if __name__ == "__main__":
    plot_tool = InteractiveScatterPlot()
    plot_tool.show()