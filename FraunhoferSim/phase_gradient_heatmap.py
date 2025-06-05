import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# --- 常數與參數設定 ---
W = 1.0          # 接面寬度 (歸一化)
N_points_x = 256 # 接面空間取樣點數
x_coords = np.linspace(-W/2, W/2, N_points_x) # 接面寬度座標

# phase_k0_norm 參數範圍
phase_k0_norm_values = np.linspace(0, 5, 501)  # 從0到5的範圍

def uniform_Jc(x_coords, W_val):
    """均勻電流密度"""
    return np.ones_like(x_coords)

def Jc_with_linear_phase_gradient(x_coords, W_val, J_amplitude_func, phase_k0_norm):
    """帶有內稟線性相位梯度的電流密度 (用於產生位移的對稱圖樣)，確保實部為正"""
    J_real = J_amplitude_func(x_coords, W_val)
    k_intrinsic = phase_k0_norm * np.pi 
    phase = k_intrinsic * (x_coords / (W_val/2))
    # 確保實部為正：(1 + cos(phase))/2
    cos_part = np.cos(phase)
    sin_part = np.sin(phase)
    real_part = J_real * (1 + cos_part) / 2
    imag_part = J_real * sin_part
    return real_part + 1j * imag_part

# 創建數據矩陣
real_parts_matrix = np.zeros((len(phase_k0_norm_values), len(x_coords)))
imag_parts_matrix = np.zeros((len(phase_k0_norm_values), len(x_coords)))
magnitude_matrix = np.zeros((len(phase_k0_norm_values), len(x_coords)))

# 計算每個phase_k0_norm值的電流密度分佈
for i, phase_k0_norm in enumerate(phase_k0_norm_values):
    J_eff = Jc_with_linear_phase_gradient(x_coords, W, uniform_Jc, phase_k0_norm)
    
    # 歸一化
    max_abs_val = np.max(np.abs(J_eff))
    if max_abs_val > 1e-9:
        J_eff = J_eff / max_abs_val
    
    real_parts_matrix[i, :] = np.real(J_eff)
    imag_parts_matrix[i, :] = np.imag(J_eff)
    magnitude_matrix[i, :] = np.abs(J_eff)

# 創建包含三個子圖的heatmap
fig = make_subplots(
    rows=3, cols=1,
    subplot_titles=(
        "實部 Re[J<sub>eff</sub>(x)] vs phase_k0_norm",
        "虛部 Im[J<sub>eff</sub>(x)] vs phase_k0_norm", 
        "幅度 |J<sub>eff</sub>(x)| vs phase_k0_norm"
    ),
    vertical_spacing=0.08
)

# 實部 Heatmap
fig.add_trace(
    go.Heatmap(
        z=real_parts_matrix,
        x=x_coords/W,  # 歸一化位置
        y=phase_k0_norm_values,
        colorscale='RdBu',
        name="實部",
        colorbar=dict(
            title="Re[J<sub>eff</sub>(x)]",
            titleside="right",
            x=1.02,
            y=0.85,
            len=0.25
        )
    ),
    row=1, col=1
)

# 虛部 Heatmap
fig.add_trace(
    go.Heatmap(
        z=imag_parts_matrix,
        x=x_coords/W,
        y=phase_k0_norm_values,
        colorscale='RdBu',
        name="虛部",
        colorbar=dict(
            title="Im[J<sub>eff</sub>(x)]",
            titleside="right",
            x=1.02,
            y=0.5,
            len=0.25
        )
    ),
    row=2, col=1
)

# 幅度 Heatmap
fig.add_trace(
    go.Heatmap(
        z=magnitude_matrix,
        x=x_coords/W,
        y=phase_k0_norm_values,
        colorscale='RdBu',
        name="幅度",
        colorbar=dict(
            title="|J<sub>eff</sub>(x)|",
            titleside="right",
            x=1.02,
            y=0.15,
            len=0.25
        )
    ),
    row=3, col=1
)

# 更新軸標籤
fig.update_xaxes(title_text="歸一化接面位置 x/W", row=3, col=1)
fig.update_yaxes(title_text="phase_k0_norm", row=1, col=1)
fig.update_yaxes(title_text="phase_k0_norm", row=2, col=1)
fig.update_yaxes(title_text="phase_k0_norm", row=3, col=1)

# 更新布局
fig.update_layout(
    height=1200,
    title_text="帶有內稟線性相位梯度的電流密度 vs phase_k0_norm 參數",
    title_x=0.5,
    font=dict(family="Arial", size=12),
    showlegend=False
)

fig.show()

# 創建單獨的詳細圖表，顯示特定phase_k0_norm值的例子
fig_examples = go.Figure()

# 選擇幾個具有代表性的phase_k0_norm值
example_values = [0, 0.5, 1.0, 1.5, 2.0]
colors = ['blue', 'green', 'red', 'orange', 'purple']

for i, phase_k0_norm in enumerate(example_values):
    J_eff = Jc_with_linear_phase_gradient(x_coords, W, uniform_Jc, phase_k0_norm)
    
    # 歸一化
    max_abs_val = np.max(np.abs(J_eff))
    if max_abs_val > 1e-9:
        J_eff = J_eff / max_abs_val
    
    # 繪製實部
    fig_examples.add_trace(go.Scatter(
        x=x_coords/W, 
        y=np.real(J_eff), 
        mode='lines',
        name=f'實部 (k₀={phase_k0_norm})',
        line=dict(color=colors[i], width=2)
    ))
    
    # 繪製虛部
    fig_examples.add_trace(go.Scatter(
        x=x_coords/W, 
        y=np.imag(J_eff), 
        mode='lines',
        name=f'虛部 (k₀={phase_k0_norm})',
        line=dict(color=colors[i], width=2, dash='dash')
    ))

fig_examples.update_layout(
    title="不同 phase_k0_norm 值的電流密度分佈示例",
    xaxis_title="歸一化接面位置 x/W",
    yaxis_title="歸一化有效電流密度 J<sub>eff</sub>(x)",
    height=600,
    legend=dict(x=1.02, y=1)
)

fig_examples.show()

# 創建相位分佈的heatmap
phase_matrix = np.zeros((len(phase_k0_norm_values), len(x_coords)))

for i, phase_k0_norm in enumerate(phase_k0_norm_values):
    J_eff = Jc_with_linear_phase_gradient(x_coords, W, uniform_Jc, phase_k0_norm)
    # 計算相位 (以弧度為單位)
    phase_matrix[i, :] = np.angle(J_eff)

fig_phase = go.Figure()

fig_phase.add_trace(
    go.Heatmap(
        z=phase_matrix,
        x=x_coords/W,
        y=phase_k0_norm_values,
        colorscale='RdBu',
        colorbar=dict(
            title="相位 (弧度)",
            titleside="right",
            x=1.02
        )
    )
)

fig_phase.update_layout(
    title="電流密度相位分佈 vs phase_k0_norm",
    xaxis_title="歸一化接面位置 x/W",
    yaxis_title="phase_k0_norm",
    height=600
)

fig_phase.show()

# 輸出統計信息
print("=" * 60)
print("帶有內稟線性相位梯度的電流密度分析")
print("=" * 60)
print(f"phase_k0_norm 範圍: {phase_k0_norm_values[0]:.2f} 到 {phase_k0_norm_values[-1]:.2f}")
print(f"空間點數: {len(x_coords)}")
print(f"參數點數: {len(phase_k0_norm_values)}")
print()
print("實部統計:")
print(f"  最小值: {np.min(real_parts_matrix):.6f}")
print(f"  最大值: {np.max(real_parts_matrix):.6f}")
print(f"  平均值: {np.mean(real_parts_matrix):.6f}")
print()
print("虛部統計:")
print(f"  最小值: {np.min(imag_parts_matrix):.6f}")
print(f"  最大值: {np.max(imag_parts_matrix):.6f}")
print(f"  平均值: {np.mean(imag_parts_matrix):.6f}")
print()
print("幅度統計:")
print(f"  最小值: {np.min(magnitude_matrix):.6f}")
print(f"  最大值: {np.max(magnitude_matrix):.6f}")
print(f"  平均值: {np.mean(magnitude_matrix):.6f}")

# 檢查實部是否始終為正
all_positive = np.all(real_parts_matrix >= 0)
print()
print(f"所有實部都為正值: {all_positive}")
if not all_positive:
    negative_count = np.sum(real_parts_matrix < 0)
    print(f"負值點數: {negative_count}")
else:
    print("✓ 成功！所有電流密度的實部都保持為正值")
