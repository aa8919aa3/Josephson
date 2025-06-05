import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 常數與參數設定 ---
W = 1.0          # 接面寬度 (歸一化)
N_points_x = 256 # 接面空間取樣點數
x_coords = np.linspace(-W/2, W/2, N_points_x) # 接面寬度座標

# 掃描的外部磁通量範圍 (歸一化)
phi_ext_over_phi0_scan = np.linspace(-4, 4, 401) # 縮小範圍以便觀察

# phase_k0_norm 參數範圍
phase_k0_norm_values = np.linspace(0, 5, 501)  # 減少點數以加快計算

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
current_density_real_matrix = np.zeros((len(phase_k0_norm_values), len(x_coords)))
current_density_imag_matrix = np.zeros((len(phase_k0_norm_values), len(x_coords)))
Ic_patterns_matrix = np.zeros((len(phase_k0_norm_values), len(phi_ext_over_phi0_scan)))

print("正在計算電流密度和Ic圖樣...")

# 計算每個phase_k0_norm值的電流密度分佈和對應的Ic(Phi)圖樣
for i, phase_k0_norm in enumerate(phase_k0_norm_values):
    if i % 5 == 0:
        print(f"進度: {i+1}/{len(phase_k0_norm_values)}")
    
    # 計算電流密度
    J_eff = Jc_with_linear_phase_gradient(x_coords, W, uniform_Jc, phase_k0_norm)
    
    # 歸一化電流密度
    max_abs_val = np.max(np.abs(J_eff))
    if max_abs_val > 1e-9:
        J_eff = J_eff / max_abs_val
    
    current_density_real_matrix[i, :] = np.real(J_eff)
    current_density_imag_matrix[i, :] = np.imag(J_eff)
    
    # 計算Ic(Phi)圖樣
    Ic_vs_phi = []
    for phi_norm in phi_ext_over_phi0_scan:
        integrand = J_eff * np.exp(1j * 2 * np.pi * phi_norm * (x_coords / W))
        dx = W / (N_points_x - 1) if N_points_x > 1 else W 
        Ic_val = np.abs(np.sum(integrand) * dx) 
        Ic_vs_phi.append(Ic_val)
    
    Ic_patterns_matrix[i, :] = np.array(Ic_vs_phi)
    
    # 歸一化Ic圖樣
    max_ic_val = np.max(Ic_patterns_matrix[i, :])
    if max_ic_val > 1e-9:
        Ic_patterns_matrix[i, :] /= max_ic_val

print("計算完成！正在生成圖表...")

# 創建包含四個子圖的複合圖表
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "電流密度實部 Re[J<sub>eff</sub>(x)] vs phase_k0_norm",
        "電流密度虛部 Im[J<sub>eff</sub>(x)] vs phase_k0_norm", 
        "臨界電流圖樣 I<sub>c</sub>(Φ) vs phase_k0_norm",
        "選定參數的Ic(Φ)曲線比較"
    ),
    specs=[[{"type": "heatmap"}, {"type": "heatmap"}],
           [{"type": "heatmap"}, {"type": "scatter"}]],
    vertical_spacing=0.12,
    horizontal_spacing=0.08
)

# 實部 Heatmap
fig.add_trace(
    go.Heatmap(
        z=current_density_real_matrix,
        x=x_coords/W,
        y=phase_k0_norm_values,
        colorscale='RdBu',
        name="實部",
        zmin=0,
        zmax=1,
        colorbar=dict(
            title="Re[J<sub>eff</sub>]",
            titleside="right",
            x=0.47,
            y=0.75,
            len=0.4
        )
    ),
    row=1, col=1
)

# 虛部 Heatmap
fig.add_trace(
    go.Heatmap(
        z=current_density_imag_matrix,
        x=x_coords/W,
        y=phase_k0_norm_values,
        colorscale='RdBu',
        name="虛部",
        colorbar=dict(
            title="Im[J<sub>eff</sub>]",
            titleside="right",
            x=1.02,
            y=0.75,
            len=0.4
        )
    ),
    row=1, col=2
)

# Ic圖樣 Heatmap
fig.add_trace(
    go.Heatmap(
        z=Ic_patterns_matrix,
        x=phi_ext_over_phi0_scan,
        y=phase_k0_norm_values,
        colorscale='RdBu',
        name="Ic圖樣",
        colorbar=dict(
            title="I<sub>c</sub>(Φ)",
            titleside="right",
            x=0.47,
            y=0.25,
            len=0.4
        )
    ),
    row=2, col=1
)

# 選定參數的Ic(Φ)曲線比較
example_indices = [0, 26, 51, 101, 201, 301, 401, 500]  # 對應phase_k0_norm = 0, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0
colors = ['blue', 'green', 'red', 'orange', 'purple', 'cyan', 'magenta', 'black']

for i, idx in enumerate(example_indices):
    phase_k0_norm = phase_k0_norm_values[idx]
    fig.add_trace(
        go.Scatter(
            x=phi_ext_over_phi0_scan,
            y=Ic_patterns_matrix[idx, :],
            mode='lines',
            name=f'k₀={phase_k0_norm:.2f}',
            line=dict(color=colors[i], width=2),
            showlegend=True
        ),
        row=2, col=2
    )

# 更新軸標籤
fig.update_xaxes(title_text="歸一化位置 x/W", row=1, col=1)
fig.update_xaxes(title_text="歸一化位置 x/W", row=1, col=2)
fig.update_xaxes(title_text="歸一化磁通量 Φ/Φ₀", row=2, col=1)
fig.update_xaxes(title_text="歸一化磁通量 Φ/Φ₀", row=2, col=2)

fig.update_yaxes(title_text="phase_k0_norm", row=1, col=1)
fig.update_yaxes(title_text="phase_k0_norm", row=1, col=2)
fig.update_yaxes(title_text="phase_k0_norm", row=2, col=1)
fig.update_yaxes(title_text="歸一化臨界電流", row=2, col=2)

# 更新布局
fig.update_layout(
    height=1000,
    width=1400,
    title_text="線性相位梯度電流密度的完整分析: J<sub>eff</sub>(x) 和 I<sub>c</sub>(Φ) vs phase_k0_norm",
    title_x=0.5,
    font=dict(family="Arial", size=11),
    legend=dict(x=0.75, y=0.35)
)

fig.show()

# 創建3D表面圖
fig_3d = go.Figure()

# 創建3D表面圖顯示電流密度實部
fig_3d.add_trace(go.Surface(
    z=current_density_real_matrix,
    x=x_coords/W,
    y=phase_k0_norm_values,
    colorscale='RdBu',
    name="實部",
    colorbar=dict(
        title="Re[J<sub>eff</sub>]",
        titleside="right",
        x=1.02
    )
))

fig_3d.update_layout(
    title="電流密度實部的3D表面圖",
    scene=dict(
        xaxis_title="歸一化位置 x/W",
        yaxis_title="phase_k0_norm",
        zaxis_title="Re[J<sub>eff</sub>(x)]",
        camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.5, y=1.5, z=1.5)
        )
    ),
    height=700
)

fig_3d.show()

# 輸出詳細統計信息
print("\n" + "=" * 80)
print("線性相位梯度電流密度的完整分析結果")
print("=" * 80)
print(f"phase_k0_norm 範圍: {phase_k0_norm_values[0]:.2f} 到 {phase_k0_norm_values[-1]:.2f}")
print(f"空間採樣點數: {len(x_coords)}")
print(f"參數採樣點數: {len(phase_k0_norm_values)}")
print(f"磁通量採樣點數: {len(phi_ext_over_phi0_scan)}")
print()

print("電流密度統計:")
print(f"  實部範圍: [{np.min(current_density_real_matrix):.6f}, {np.max(current_density_real_matrix):.6f}]")
print(f"  虛部範圍: [{np.min(current_density_imag_matrix):.6f}, {np.max(current_density_imag_matrix):.6f}]")
print(f"  實部平均值: {np.mean(current_density_real_matrix):.6f}")
print(f"  虛部平均值: {np.mean(current_density_imag_matrix):.6f}")
print()

print("Ic圖樣統計:")
print(f"  Ic值範圍: [{np.min(Ic_patterns_matrix):.6f}, {np.max(Ic_patterns_matrix):.6f}]")
print(f"  Ic平均值: {np.mean(Ic_patterns_matrix):.6f}")
print()

# 分析不同phase_k0_norm值對應的物理特性
print("物理特性分析:")
for i, phase_k0_norm in enumerate([0, 0.5, 1.0, 1.5, 2.0]):
    if phase_k0_norm <= phase_k0_norm_values[-1]:
        idx = np.argmin(np.abs(phase_k0_norm_values - phase_k0_norm))
        real_std = np.std(current_density_real_matrix[idx, :])
        imag_std = np.std(current_density_imag_matrix[idx, :])
        ic_max = np.max(Ic_patterns_matrix[idx, :])
        print(f"  phase_k0_norm = {phase_k0_norm:.1f}: 實部變化 = {real_std:.4f}, 虛部變化 = {imag_std:.4f}, 最大Ic = {ic_max:.4f}")

print(f"\n✓ 驗證: 所有電流密度實部都為正值: {np.all(current_density_real_matrix >= 0)}")
print("分析完成！")
