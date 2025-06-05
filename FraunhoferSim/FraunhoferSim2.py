import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.fft import fft, fftshift
from scipy.special import jv, yv # Bessel functions J_n and Y_n
from scipy.interpolate import lagrange # Lagrange polynomials

# --- 常數與參數設定 ---
Phi0 = 1.0       # 磁通量量子 (歸一化)
W = 1.0          # 接面寬度 (歸一化)
N_points_x = 256 # 接面空間取樣點數
x_coords = np.linspace(-W/2, W/2, N_points_x) # 接面寬度座標

# 掃描的外部磁通量範圍 (歸一化)
phi_ext_over_phi0_scan = np.linspace(-8, 8, 800) # 擴大範圍以觀察更多旁瓣

# --- 定義不同的電流密度分佈函數 Jc(x) ---
def uniform_Jc(x_coords, W_val):
    """均勻電流密度"""
    return np.ones_like(x_coords)

def symmetric_edge_gaussian_Jc(x_coords, W_val, peak_width_factor=0.1, edge_offset_factor=0.05):
    """對稱邊緣集中的電流密度 (兩個高斯峰)"""
    sigma = W_val * peak_width_factor
    peak1_center = -W_val/2 + W_val * edge_offset_factor + sigma 
    peak2_center = W_val/2 - W_val * edge_offset_factor - sigma
    return np.exp(-(x_coords - peak1_center)**2 / (2 * sigma**2)) + \
           np.exp(-(x_coords - peak2_center)**2 / (2 * sigma**2))

def asymmetric_linear_Jc(x_coords, W_val):
    """不對稱線性梯度電流密度 (僅正值)"""
    return 0.7 + 0.5 * (x_coords / (W_val/2)) 

def one_side_step_Jc(x_coords, W_val, step_start_norm=-0.5, step_end_norm=-0.2):
    """只有一邊的不對稱邊緣集中 (階梯)"""
    J = np.zeros_like(x_coords)
    start_x = step_start_norm * W_val
    end_x = step_end_norm * W_val
    actual_start = np.maximum(start_x, -W_val/2)
    actual_end = np.minimum(end_x, W_val/2)
    if actual_start < actual_end:
         J[(x_coords >= actual_start) & (x_coords <= actual_end)] = 1.0
    return J

def asymmetric_multistep_Jc(x_coords, W_val, x_split_norm=-0.1, h1=1.0, h2=0.4):
    """不對稱多階梯 (兩段不同高度, 僅正值)"""
    J = np.zeros_like(x_coords)
    x_split = x_split_norm * W_val
    J[x_coords <= x_split] = h1
    J[x_coords > x_split] = h2
    J[x_coords < -W_val/2] = 0
    J[x_coords > W_val/2] = 0
    return J

def symmetric_edge_steps_Jc(x_coords, W_val, depth_norm=0.2, base_level=0.0):
    """雙邊均勻邊緣集中 (對稱階梯)"""
    J = np.full_like(x_coords, base_level)
    edge_width = depth_norm * W_val
    J[(x_coords >= -W_val/2) & (x_coords <= -W_val/2 + edge_width)] = 1.0
    J[(x_coords >= W_val/2 - edge_width) & (x_coords <= W_val/2)] = 1.0
    return J

def bilateral_opposite_shape_edge_Jc(x_coords, W_val, 
                                     left_step_end_norm=-0.25, left_h=1.0,
                                     right_peak_center_norm=0.3, right_peak_h=0.8, right_peak_sigma_norm=0.07):
    """雙邊異形邊緣集中 (左階梯, 右高斯)"""
    J_left = np.zeros_like(x_coords)
    left_step_end = left_step_end_norm * W_val
    J_left[(x_coords >= -W_val/2) & (x_coords <= left_step_end)] = left_h 

    J_right = np.zeros_like(x_coords)
    right_peak_center = right_peak_center_norm * W_val
    right_sigma = right_peak_sigma_norm * W_val
    J_right = right_peak_h * np.exp(-(x_coords - right_peak_center)**2 / (2 * right_sigma**2)) 

    J_combined = np.zeros_like(x_coords)
    mask_left = (x_coords >= -W_val/2) & (x_coords <= left_step_end)
    mask_right_exclusive = (x_coords > left_step_end) 

    J_combined[mask_left] = J_left[mask_left]
    J_combined[mask_right_exclusive] = J_right[mask_right_exclusive]
    return J_combined

def Jc_with_linear_phase_gradient(x_coords, W_val, J_amplitude_func, phase_k0_norm):
    """帶有內稟線性相位梯度的電流密度 (用於產生位移的對稱圖樣)"""
    J_real = J_amplitude_func(x_coords, W_val)
    k_intrinsic = phase_k0_norm * np.pi 
    return J_real * np.exp(1j * k_intrinsic * (x_coords / (W/2)))

def asymmetric_phi0_profile(x_coords, W_val, A=np.pi/2, x_d_norm=-0.2, sigma_norm=0.15):
    """一個不對稱的內稟相位移 phi_0^{int}(x) 範例 (例如，一側的相位突變)"""
    x_d = x_d_norm * W_val
    sigma = sigma_norm * W_val
    return A * np.exp(-(x_coords - x_d)**2 / (2 * sigma**2))

def Jc_with_asymmetric_phi0(x_coords, W_val, J_amplitude_func, phi0_int_func, phi0_params):
    """帶有不對稱內稟相位移的電流密度 (用於產生不對稱的Ic(Phi)圖樣)"""
    J_real = J_amplitude_func(x_coords, W_val)
    phi0_intrinsic = phi0_int_func(x_coords, W_val, **phi0_params)
    return J_real * np.exp(-1j * phi0_intrinsic) # J_eff = J_c * exp(-i * phi_0_int)

# 選定的電流密度分佈與數學表達式 (使用HTML格式，避免LaTeX渲染問題)
current_distributions = {
    "均勻 (Uniform)": {
        "profile": uniform_Jc(x_coords, W),
        "math": "J<sub>c</sub>(x) = J<sub>0</sub> (constant)"
    },
    "對稱邊緣高斯 (Symm. Edge Gaussian)": {
        "profile": symmetric_edge_gaussian_Jc(x_coords, W, peak_width_factor=0.08, edge_offset_factor=0.05),
        "math": "J<sub>c</sub>(x) = exp(-(x-x<sub>1</sub>)<sup>2</sup>/(2σ<sup>2</sup>)) + exp(-(x-x<sub>2</sub>)<sup>2</sup>/(2σ<sup>2</sup>))"
    },
    "不對稱線性 (僅正值) (Asymm. Linear)": {
        "profile": asymmetric_linear_Jc(x_coords, W),
        "math": "J<sub>c</sub>(x) = 0.7 + 0.5 × (x/(W/2))"
    },
    "單邊階梯 (One-side Step)": {
        "profile": one_side_step_Jc(x_coords, W, step_start_norm=-0.5, step_end_norm=-0.2),
        "math": "J<sub>c</sub>(x) = { J<sub>0</sub> if x<sub>1</sub> ≤ x ≤ x<sub>2</sub>; 0 otherwise }"
    },
    "不對稱多階梯 (僅正值) (Asymm. Multi-Steps)": {
        "profile": asymmetric_multistep_Jc(x_coords, W, x_split_norm=-0.1, h1=1.0, h2=0.4),
        "math": "J<sub>c</sub>(x) = { h<sub>1</sub> if x ≤ x<sub>s</sub>; h<sub>2</sub> if x > x<sub>s</sub> }"
    },
    "雙邊均勻邊緣 (Symm. Edge Steps)": {
        "profile": symmetric_edge_steps_Jc(x_coords, W, depth_norm=0.15, base_level=0.0),
        "math": "J<sub>c</sub>(x) = { J<sub>0</sub> if |x| > W/2 - d; 0 otherwise }"
    },
    "雙邊異形邊緣 (Bilateral Diff. Shapes)": {
        "profile": bilateral_opposite_shape_edge_Jc(x_coords, W, left_step_end_norm=-0.2, left_h=0.9, right_peak_center_norm=0.3, right_peak_h=1.0, right_peak_sigma_norm=0.1),
        "math": "J<sub>c</sub>(x) = { h<sub>L</sub> (left step); h<sub>R</sub> exp(-(x-x<sub>R</sub>)<sup>2</sup>/(2σ<sub>R</sub><sup>2</sup>)) (right Gaussian) }"
    },
    "均勻Jc+線性相位梯度 (僅正值) (Shifted Uniform)": {
        "profile": Jc_with_linear_phase_gradient(x_coords, W, lambda x, w: uniform_Jc(x,w), phase_k0_norm=0.5),
        "math": "J<sub>eff</sub>(x) = J<sub>0</sub> exp(i k<sub>0</sub> x)"
    },
    "均勻Jc+不對稱相位 (僅正值) (Asymmetric Ic(Phi))": {
        "profile": Jc_with_asymmetric_phi0(x_coords, W, 
                                              lambda x, w: uniform_Jc(x,w), # 基底 Jc(x)
                                              asymmetric_phi0_profile,     # 不對稱 phi_0^{int}(x) 函數
                                              {'A': np.pi*0.8, 'x_d_norm': -0.25, 'sigma_norm': 0.2}), # phi_0^{int}(x) 的參數
        "math": "J<sub>eff</sub>(x) = J<sub>0</sub> exp(-i φ<sub>0</sub><sup>int</sup>(x)),<br>where φ<sub>0</sub><sup>int</sup>(x) = A exp(-(x-x<sub>d</sub>)<sup>2</sup>/(2σ<sup>2</sup>))"
    }
}

# 歸一化電流密度分佈
for name, data in current_distributions.items():
    Jc_profile_complex = data["profile"]
    if np.iscomplexobj(Jc_profile_complex):
        max_abs_val = np.max(np.abs(Jc_profile_complex))
        if max_abs_val > 1e-9:
            current_distributions[name]["profile"] = Jc_profile_complex / max_abs_val
    else: 
        max_abs_val = np.max(np.abs(Jc_profile_complex))
        if max_abs_val > 1e-9: 
             current_distributions[name]["profile"] = Jc_profile_complex / max_abs_val 

# 計算 Ic(Phi) 圖樣
Ic_patterns = {}
for name, data in current_distributions.items():
    Jc_profile_complex = data["profile"]
    Ic_vs_phi = []
    for phi_norm in phi_ext_over_phi0_scan:
        integrand = Jc_profile_complex * np.exp(1j * 2 * np.pi * phi_norm * (x_coords / W))
        dx = W / (N_points_x -1) if N_points_x > 1 else W 
        Ic_val = np.abs(np.sum(integrand) * dx) 
        Ic_vs_phi.append(Ic_val)

    Ic_patterns[name] = np.array(Ic_vs_phi)
    max_ic_val_pattern = np.max(Ic_patterns[name])
    if max_ic_val_pattern > 1e-9:
         Ic_patterns[name] /= max_ic_val_pattern

# 創建圖表 - 方法1: 使用HTML格式的數學表達式
fig = make_subplots(rows=3, cols=1,
                    shared_xaxes=False,
                    row_heights=[0.35, 0.35, 0.3],
                    subplot_titles=("電流密度分佈 J<sub>eff</sub>(x) (實部與虛部，已歸一化至|·|<sub>max</sub>=1)", 
                                    "對應的夫琅禾費圖樣 I<sub>c</sub>(Φ<sub>ext</sub>) (已歸一化至I<sub>c,peak</sub>=1)",
                                    "數學表達式"))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# 繪製電流密度分佈
for i, (name, data) in enumerate(current_distributions.items()):
    Jc_profile_complex = data["profile"]
    color = colors[i % len(colors)]
    
    if np.iscomplexobj(Jc_profile_complex):
        fig.add_trace(go.Scatter(x=x_coords/W, y=np.real(Jc_profile_complex), mode='lines', 
                                 name=f'{name} (實部)', legendgroup=name, 
                                 line=dict(color=color)),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=x_coords/W, y=np.imag(Jc_profile_complex), mode='lines', 
                                 name=f'{name} (虛部)', legendgroup=name, 
                                 line=dict(color=color, dash='dot')),
                      row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=x_coords/W, y=Jc_profile_complex, mode='lines', 
                                 name=f'{name}', legendgroup=name, 
                                 line=dict(color=color)),
                      row=1, col=1)

    # 繪製 Ic(Phi) 圖樣
    fig.add_trace(go.Scatter(x=phi_ext_over_phi0_scan, y=Ic_patterns[name], mode='lines',
                             name=f'{name} - Ic(Φ)', legendgroup=name, 
                             showlegend=False, 
                             line=dict(color=color)),
                  row=2, col=1)

# 方法1: 使用HTML格式添加數學表達式文本
math_text_html = ""
for i, (name, data) in enumerate(current_distributions.items()):
    math_text_html += f"<b>{i+1}. {name}:</b><br>{data['math']}<br><br>"

fig.add_annotation(
    text=math_text_html,
    xref="paper", yref="paper",
    x=0.05, y=0.28,
    xanchor="left", yanchor="top",
    showarrow=False,
    font=dict(size=11),
    bgcolor="rgba(255,255,255,0.9)",
    bordercolor="rgba(0,0,0,0.3)",
    borderwidth=1
)

# 更新軸標籤和範圍
fig.update_xaxes(title_text="歸一化接面位置 x/W", row=1, col=1)
fig.update_yaxes(title_text="歸一化有效電流密度 J<sub>eff</sub>(x)", row=1, col=1, range=[-1.1, 1.1]) 

fig.update_xaxes(title_text="歸一化外部磁通量 (Φ<sub>ext</sub> / Φ<sub>0</sub>)", row=2, col=1, dtick=1)
fig.update_yaxes(title_text="歸一化臨界電流 (I<sub>c</sub> / I<sub>c,peak</sub>)", row=2, col=1, range=[-0.05, 1.1]) 

# 隱藏第三個子圖的軸
fig.update_xaxes(visible=False, row=3, col=1)
fig.update_yaxes(visible=False, row=3, col=1)

# 設定 MathJax 支援 (如果需要LaTeX渲染)
fig.update_layout(
    height=1800, 
    title_text="指定電流密度分佈及其對應的夫琅禾費干涉圖樣與數學表達式",
    title_x=0.5,
    legend_title_text="有效電流密度分佈類型",
    legend_tracegroupgap = 5,
    # 嘗試啟用MathJax支援
    font=dict(family="Arial", size=12)
)

# 如果您希望使用LaTeX格式，可以嘗試以下設定：
# fig.update_layout(
#     font=dict(family="Computer Modern"),  # LaTeX字體
#     title_font_size=16,
# )

fig.show()

# 或者，如果上述方法仍有問題，可以使用以下替代方案：
print("如果圖表中的數學表達式仍有顯示問題，請嘗試以下解決方案：")
print("1. 更新 Plotly: pip install --upgrade plotly")
print("2. 在 Jupyter 中啟用 MathJax: %config InlineBackend.figure_format = 'retina'")
print("3. 或使用純文字輸出數學表達式：")
print("\n數學表達式列表：")
for i, (name, data) in enumerate(current_distributions.items()):
    print(f"{i+1}. {name}: {data['math']}")