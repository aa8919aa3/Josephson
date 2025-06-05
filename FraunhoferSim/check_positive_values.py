import numpy as np

# 檢查J_eff函數的實數部分是否都為正值
J0_amplitude = 1.0
k0 = 1.0
x_min = 0
x_max = 4 * np.pi
num_points = 200

x_values = np.linspace(x_min, x_max, num_points)

def J_eff_formula(x, J0, k0):
    """
    計算有效電流密度 J_eff(x) = J0 * ((1 + cos(k0 * x))/2 + i * sin(k0 * x))
    返回複數值，確保實部始終為正
    """
    cos_part = np.cos(k0 * x)
    sin_part = np.sin(k0 * x)
    # 將實部從[-1,1]映射到[0,1]範圍，確保為正值
    real_part = J0 * (1 + cos_part) / 2
    imag_part = J0 * sin_part
    return real_part + 1j * imag_part

# 計算J_eff值
J_eff_values = J_eff_formula(x_values, J0_amplitude, k0)

# 檢查實部
real_parts = np.real(J_eff_values)
min_real = np.min(real_parts)
max_real = np.max(real_parts)

print(f"實部的最小值: {min_real:.6f}")
print(f"實部的最大值: {max_real:.6f}")
print(f"實部是否都為正值: {np.all(real_parts >= 0)}")

# 檢查虛部範圍
imag_parts = np.imag(J_eff_values)
min_imag = np.min(imag_parts)
max_imag = np.max(imag_parts)

print(f"虛部的最小值: {min_imag:.6f}")
print(f"虛部的最大值: {max_imag:.6f}")

# 檢查幅度
amplitudes = np.abs(J_eff_values)
min_amp = np.min(amplitudes)
max_amp = np.max(amplitudes)

print(f"幅度的最小值: {min_amp:.6f}")
print(f"幅度的最大值: {max_amp:.6f}")

# 檢查是否有負的實部
negative_real_count = np.sum(real_parts < 0)
print(f"負實部的數量: {negative_real_count}")

if negative_real_count == 0:
    print("✓ 成功！所有J_eff的實數部分都為正值")
else:
    print("✗ 仍有負的實數部分")
