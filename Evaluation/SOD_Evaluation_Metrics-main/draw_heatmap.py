# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# from matplotlib.colors import LinearSegmentedColormap
# import matplotlib
# matplotlib.use("agg")
# # 自定义颜色映射（深蓝到深红）
# cmap_colors = [(0.0, '#000080'), (1.0, '#8b0000')]
# gradient_cmap = LinearSegmentedColormap.from_list('BlueRed', cmap_colors)
#
# def save_channel_heatmap(channel, filename):
#     """保存单个通道的热力图"""
#     fig = plt.figure(frameon=False, figsize=(10, 10))
#     ax = plt.Axes(fig, [0, 0, 1, 1])
#     ax.set_axis_off()
#     fig.add_axes(ax)
#     ax.imshow(channel, cmap='tab20c', aspect='auto')
#     plt.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=300)
#     plt.close()
#
# # 读取并转换图片
# image_path = "0.jpg"  # 替换为实际路径
# img = Image.open(image_path).convert('YCbCr')
# y, cb, cr = img.split()
#
# # 转换为数值矩阵并归一化
# y_array = np.array(y,dtype=float) / 255
# cb_array = np.array(cb, dtype=float) / 255
# cr_array = np.array(cr, dtype=float) / 255
#
# # 保存热力图
# save_channel_heatmap(y_array,"y_heatmap.png")
# save_channel_heatmap(cb_array, "cb_heatmap.png")
# save_channel_heatmap(cr_array, "cr_heatmap.png")
#
# print("Cb和Cr通道热力图已保存为 cb_heatmap.png 和 cr_heatmap.png")
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
# 创建网格数据
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
# Z = np.sin(X)*np.sin(Y)+0.5*np.sin(2*X)*np.cos(2*Y)
Z = -(np.exp(-((X-2)**2+(Y-2)**2)) + 2*np.exp(-((X+1)**2+(Y+0.1)**2)) + 1.2*np.exp(-((X-1)**2+(Y+1)**2))) + 1.5*np.exp(-((X)**2+(Y-0.5)**2))
# 创建三维表面图
fig = go.Figure(data=[go.Surface(
    z=Z, x=X, y=Y, showscale=False,
    colorscale=[
        [0, '#ACFAFB'],
        [0.5, '#80A4F9'],
        [1, '#8b0000']
    ],
    contours={
        "x": {"show": True, "color": "black", "width": 1},
        "y": {"show": True, "color": "black", "width": 1},
        "z": {"show": True, "color": "black", "width": 1},
    }
)])

# # 添加优化路径（示例）
# path_x = [2.030303, 1.242424, 0.3939394,-0.3939394,-0.8787879,-2.090909]
# path_y = [1.545455, 0.9393939, 1.060606,1.242424,0.5757576,1.606061]
# path_z = [1.392975, 0.715293, 0.17995677,-0.00261807,-0.5092354,-1.0097411]
#
path_x = [0.03030303, -0.2121212, -0.1515152,-0.6969697,-0.4545455,-1.181818]
path_y = [0.57575, 0.4545455, 0.09090909,0.3939394,-0.3333333,-0.1515152]
path_z = [1.142, 0.8220254, 0.42741947,-0.02757716,-0.51003434,-1.5912959]
fig.add_trace(go.Scatter3d(x=path_x, y=path_y, z=path_z,
                           mode='lines+markers',
                           line=dict(color='red', width=5),
                           marker=dict(size=3, color='red'),
                           name='优化路径'))
# path_x = [2.032727, 1.363636, 1.121212, 1.30303, 1.545455,1.121212]
# path_y = [1.244242, 0.8181818, 0.1515152, -0.3939394, -1, -1.545455]
# path_z = [1.35792975, 0.75111, 0.559508, -0.10901, -0.7517422, -1.271241]
path_x = [0.2727273,0.6969697,1.181818,1.30303,0.9393939,1]
path_y = [0.5151515,0.5151515,0.4545455,-0.1515152,-0.5151515,-1]
path_z = [1.164554,0.755287,0.2408195,-0.3191959,-0.7034035,-1]
fig.add_trace(go.Scatter3d(x=path_x, y=path_y, z=path_z,
                           mode='lines+markers',
                           line=dict(color='green', width=5),
                           marker=dict(size=3, color='green'),
                           name='优化路径'))
fig.update_layout(
    showlegend=False,  # 关闭图例
    scene=dict(
        xaxis=dict(visible=False),  # 隐藏X轴
        yaxis=dict(visible=False),  # 隐藏Y轴
        zaxis=dict(visible=False),  # 隐藏Z轴
    ),
    margin=dict(l=0, r=0, b=0, t=0)  # 可选：去除边缘空白
)

pio.renderers.default = "browser"     # 或 "vscode", "png", "svg" 等

# 显示图形
fig.show()

