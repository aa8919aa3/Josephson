import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 常數與參數設定 ---
W = 1.0          # 接面寬度 (歸一化)
N_points_x = 128 # 接面空間取樣點數 (減少以加快計算)
x_coords = np.linspace(-W/2, W/2, N_points_x) # 接面寬度座標

# phase_k0_norm 參數範圍
phase_k0_norm_values = np.linspace(0, 2, 25)  # 減少點數以加快計算

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
phase_matrix = np.zeros((len(phase_k0_norm_values), len(x_coords)))

print("正在計算電流密度數據...")

# 計算每個phase_k0_norm值的電流密度分佈
for i, phase_k0_norm in enumerate(phase_k0_norm_values):
    if i % 5 == 0:
        print(f"進度: {i+1}/{len(phase_k0_norm_values)}")
    
    J_eff = Jc_with_linear_phase_gradient(x_coords, W, uniform_Jc, phase_k0_norm)
    
    # 歸一化
    max_abs_val = np.max(np.abs(J_eff))
    if max_abs_val > 1e-9:
        J_eff = J_eff / max_abs_val
    
    real_parts_matrix[i, :] = np.real(J_eff)
    imag_parts_matrix[i, :] = np.imag(J_eff)
    magnitude_matrix[i, :] = np.abs(J_eff)
    phase_matrix[i, :] = np.angle(J_eff)

print("數據計算完成！正在生成統一colorbar的圖表...")

# 創建2x2的子圖佈局，所有使用RdBu colorbar
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "實部 Re[J<sub>eff</sub>(x)] - RdBu Colorbar",
        "虛部 Im[J<sub>eff</sub>(x)] - RdBu Colorbar", 
        "幅度 |J<sub>eff</sub>(x)| - RdBu Colorbar",
        "相位 ∠J<sub>eff</sub>(x) - RdBu Colorbar"
    ),
    specs=[[{"type": "heatmap"}, {"type": "heatmap"}],
           [{"type": "heatmap"}, {"type": "heatmap"}]],
    vertical_spacing=0.12,
    horizontal_spacing=0.1
)

# 實部 Heatmap (左上)
fig.add_trace(
    go.Heatmap(
        z=real_parts_matrix,
        x=x_coords/W,
        y=phase_k0_norm_values,
        colorscale='RdBu',
        name="實部",
        zmin=0,
        zmax=1,
        colorbar=dict(
            title="Re[J<sub>eff</sub>]",
            titleside="right",
            x=0.44,
            y=0.85,
            len=0.35,
            thickness=15
        )
    ),
    row=1, col=1
)

# 虛部 Heatmap (右上)
fig.add_trace(
    go.Heatmap(
        z=imag_parts_matrix,
        x=x_coords/W,
        y=phase_k0_norm_values,
        colorscale='RdBu',
        name="虛部",
        colorbar=dict(
            title="Im[J<sub>eff</sub>]",
            titleside="right",
            x=1.02,
            y=0.85,
            len=0.35,
            thickness=15
        )
    ),
    row=1, col=2
)

# 幅度 Heatmap (左下)
fig.add_trace(
    go.Heatmap(
        z=magnitude_matrix,
        x=x_coords/W,
        y=phase_k0_norm_values,
        colorscale='RdBu',
        name="幅度",
        zmin=0,
        zmax=1,
        colorbar=dict(
            title="|J<sub>eff</sub>|",
            titleside="right",
            x=0.44,
            y=0.35,
            len=0.35,
            thickness=15
        )
    ),
    row=2, col=1
)

# 相位 Heatmap (右下)
fig.add_trace(
    go.Heatmap(
        z=phase_matrix,
        x=x_coords/W,
        y=phase_k0_norm_values,
        colorscale='RdBu',
        name="相位",
        colorbar=dict(
            title="∠J<sub>eff</sub> (rad)",
            titleside="right",
            x=1.02,
            y=0.35,
            len=0.35,
            thickness=15
        )
    ),
    row=2, col=2
)

# 更新軸標籤
for row in [1, 2]:
    for col in [1, 2]:
        fig.update_xaxes(title_text="歸一化位置 x/W", row=row, col=col)
        fig.update_yaxes(title_text="phase_k0_norm", row=row, col=col)

# 更新布局
fig.update_layout(
    height=1000,
    width=1200,
    title_text="統一RdBu Colorbar的線性相位梯度電流密度分析",
    title_x=0.5,
    font=dict(family="Arial", size=12),
    showlegend=False
)

fig.show()

# 創建單獨的垂直堆疊圖表，展示不同的colorbar位置配置
fig_vertical = make_subplots(
    rows=4, cols=1,
    subplot_titles=(
        "實部 Re[J<sub>eff</sub>(x)]",
        "虛部 Im[J<sub>eff</sub>(x)]", 
        "幅度 |J<sub>eff</sub>(x)|",
        "相位 ∠J<sub>eff</sub>(x)"
    ),
    vertical_spacing=0.08
)

# 垂直排列的實部 Heatmap
fig_vertical.add_trace(
    go.Heatmap(
        z=real_parts_matrix,
        x=x_coords/W,
        y=phase_k0_norm_values,
        colorscale='RdBu',
        zmin=0,
        zmax=1,
        colorbar=dict(
            title="Re[J<sub>eff</sub>]",
            titleside="right",
            x=1.02,
            y=0.875,
            len=0.2,
            thickness=12
        )
    ),
    row=1, col=1
)

# 垂直排列的虛部 Heatmap
fig_vertical.add_trace(
    go.Heatmap(
        z=imag_parts_matrix,
        x=x_coords/W,
        y=phase_k0_norm_values,
        colorscale='RdBu',
        colorbar=dict(
            title="Im[J<sub>eff</sub>]",
            titleside="right",
            x=1.02,
            y=0.625,
            len=0.2,
            thickness=12
        )
    ),
    row=2, col=1
)

# 垂直排列的幅度 Heatmap
fig_vertical.add_trace(
    go.Heatmap(
        z=magnitude_matrix,
        x=x_coords/W,
        y=phase_k0_norm_values,
        colorscale='RdBu',
        zmin=0,
        zmax=1,
        colorbar=dict(
            title="|J<sub>eff</sub>|",
            titleside="right",
            x=1.02,
            y=0.375,
            len=0.2,
            thickness=12
        )
    ),
    row=3, col=1
)

# 垂直排列的相位 Heatmap
fig_vertical.add_trace(
    go.Heatmap(
        z=phase_matrix,
        x=x_coords/W,
        y=phase_k0_norm_values,
        colorscale='RdBu',
        colorbar=dict(
            title="∠J<sub>eff</sub> (rad)",
            titleside="right",
            x=1.02,
            y=0.125,
            len=0.2,
            thickness=12
        )
    ),
    row=4, col=1
)

# 更新垂直圖表的軸標籤
fig_vertical.update_xaxes(title_text="歸一化位置 x/W", row=4, col=1)
for row in [1, 2, 3, 4]:
    fig_vertical.update_yaxes(title_text="phase_k0_norm", row=row, col=1)

# 更新垂直圖表布局
fig_vertical.update_layout(
    height=1400,
    width=1000,
    title_text="垂直排列的RdBu Colorbar線性相位梯度分析",
    title_x=0.5,
    font=dict(family="Arial", size=11),
    showlegend=False
)

fig_vertical.show()

# 輸出colorbar配置說明
print("\n" + "=" * 80)
print("RdBu Colorbar 配置總結")
print("=" * 80)
print("已將所有Heatmap的colorbar統一設定為RdBu色彩方案")
print()
print("Colorbar位置配置:")
print("1. 2x2佈局圖表:")
print("   - 左上 (實部): x=0.44, y=0.85")
print("   - 右上 (虛部): x=1.02, y=0.85") 
print("   - 左下 (幅度): x=0.44, y=0.35")
print("   - 右下 (相位): x=1.02, y=0.35")
print()
print("2. 垂直排列圖表:")
print("   - 實部: x=1.02, y=0.875")
print("   - 虛部: x=1.02, y=0.625")
print("   - 幅度: x=1.02, y=0.375")
print("   - 相位: x=1.02, y=0.125")
print()
print("統計信息:")
print(f"  實部範圍: [0, 1] (確保正值)")
print(f"  虛部範圍: [{np.min(imag_parts_matrix):.3f}, {np.max(imag_parts_matrix):.3f}]")
print(f"  幅度範圍: [0, 1]")
print(f"  相位範圍: [{np.min(phase_matrix):.3f}, {np.max(phase_matrix):.3f}] 弧度")
print()
print("✓ 所有colorbar已成功統一為RdBu色彩方案並優化位置配置！")
