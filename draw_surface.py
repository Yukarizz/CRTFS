import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

# 创建网格数据
x = np.linspace(-2, 3, 50)
y = np.linspace(-2.5, 3, 50)
X, Y = np.meshgrid(x, y)
Z = -(np.exp(-((X-2)**2+(Y-2)**2)) + 2*np.exp(-((X+1)**2+(Y+0.1)**2)) + 1.5*np.exp(-((X-1)**2+(Y+1)**2))) + 1.5*np.exp(-((X)**2+(Y-0.5)**2))


# 创建透明曲面
surface = go.Surface(
    z=Z, x=X, y=Y,
    colorscale=[
        [0, '#A69CDC'],
        [0.5, '#8AD9E4'],
        [1, '#F7F97C']
    ],
    opacity=0.6,
    showscale=False,
    hoverinfo='skip'
)

# 构造 x 向等距线（y 不变，x 变化）
x_lines = []
for j in range(0, Y.shape[0], 1):  # 每 4 行绘一条
    x_lines.append(go.Scatter3d(
        x=X[j, :],
        y=Y[j, :],
        z=Z[j, :],
        mode='lines',
        line=dict(color='gray', width=2),
        hoverinfo='skip',
        showlegend=False
    ))

# 构造 y 向等距线（x 不变，y 变化）
y_lines = []
for i in range(0, X.shape[1], 1):  # 每 4 列绘一条
    y_lines.append(go.Scatter3d(
        x=X[:, i],
        y=Y[:, i],
        z=Z[:, i],
        mode='lines',
        line=dict(color='gray', width=2),
        hoverinfo='skip',
        showlegend=False
    ))

# 添加优化路径（可选）
path_x = [2.030303, 1.242424, 0.3939394, -0.3939394, -0.8787879, -2.090909]
path_y = [1.545455, 0.9393939, 1.060606, 1.242424, 0.5757576, 1.606061]
path_z = [1.392975, 0.715293, 0.17995677, -0.00261807, -0.5092354, -1.0097411]
path1 = go.Scatter3d(
    x=path_x, y=path_y, z=path_z,
    mode='lines+markers',
    line=dict(color='red', width=5),
    marker=dict(size=4, color='red'),
    name='路径1'
)
path_x = [2.032727, 1.363636, 1.121212, 1.30303, 1.545455,1.121212]
path_y = [1.244242, 0.8181818, 0.1515152, -0.3939394, -1, -1.545455]
path_z = [1.35792975, 0.75111, 0.559508, -0.10901, -0.7517422, -1.271241]
path2 = go.Scatter3d(
    x=path_x, y=path_y, z=path_z,
    mode='lines+markers',
    line=dict(color='green', width=5),
    marker=dict(size=4, color='green'),
    name='路径1'
)

path_x = [0.03030303, -0.2121212, -0.1515152,-0.6969697,-0.4545455,-1.181818]
path_y = [0.57575, 0.4545455, 0.09090909,0.3939394,-0.3333333,-0.1515152]
path_z = [1.142, 0.8220254, 0.42741947,-0.02757716,-0.51003434,-1.5612959]
path3 = go.Scatter3d(
    x=path_x, y=path_y, z=path_z,
    mode='lines+markers',
    line=dict(color='green', width=5),
    marker=dict(size=4, color='green'),
    name='路径1'
)
path_x = [0.2727273,0.6969697,1.181818,1.30303,0.9393939,1]
path_y = [0.5151515,0.5151515,0.4545455,-0.1515152,-0.5151515,-1]
path_z = [1.064554,0.755287,0.2408195,-0.3191959,-0.7034035,-1.35]
path4 = go.Scatter3d(
    x=path_x, y=path_y, z=path_z,
    mode='lines+markers',
    line=dict(color='red', width=5),
    marker=dict(size=4, color='red'),
    name='路径1'
)
# 创建图形
fig = go.Figure(data=[surface] + x_lines + y_lines + [path3,path4])

# 配置布局
fig.update_layout(
    showlegend=False,
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False)
    ),
    margin=dict(l=0, r=0, b=0, t=0)
)

pio.renderers.default = "browser"
fig.show()
