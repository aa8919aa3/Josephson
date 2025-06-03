"""
參數估計工具

提供自動參數估計和驗證功能。
"""

import numpy as np
from scipy import signal
from .lmfit_tools import lmfit_curve_fit, curve_fit_compatible

def estimate_initial_parameters(phi_ext, current, model_type='simplified'):
    """
    估計初始參數
    
    Parameters:
    -----------
    phi_ext : array-like
        外部磁通陣列
    current : array-like
        電流陣列
    model_type : str
        模型類型 ('full' 或 'simplified')
    
    Returns:
    --------
    dict : 估計的初始參數
    """
    
    phi_ext = np.array(phi_ext)
    current = np.array(current)
    
    # 基本統計量
    I_mean = np.mean(current)
    I_std = np.std(current)
    phi_range = phi_ext.max() - phi_ext.min()
    
    # 估計頻率（使用 FFT）
    # 確保等間距
    if not np.allclose(np.diff(phi_ext), np.diff(phi_ext)[0], rtol=1e-3):
        phi_uniform = np.linspace(phi_ext.min(), phi_ext.max(), len(phi_ext))
        current_uniform = np.interp(phi_uniform, phi_ext, current)
    else:
        phi_uniform = phi_ext
        current_uniform = current
    
    # FFT 分析
    fft_vals = np.fft.fft(current_uniform - I_mean)
    freqs = np.fft.fftfreq(len(current_uniform), np.mean(np.diff(phi_uniform)))
    
    # 只取正頻率
    pos_mask = freqs > 0
    freqs = freqs[pos_mask]
    power = np.abs(fft_vals[pos_mask])**2
    
    # 找主要頻率
    if len(power) > 0:
        dominant_idx = np.argmax(power)
        estimated_freq = freqs[dominant_idx]
    else:
        estimated_freq = 1.0 / phi_range  # 備用估計
    
    # 估計振幅
    estimated_Ic = I_std * 2  # 粗略估計
    
    # 估計其他參數
    estimated_params = {
        'Ic': estimated_Ic,
        'phi_0': 0.0,  # 初始相位
        'f': estimated_freq,
        'k': 0.0,  # 二次項（初始設為 0）
        'r': 0.0,  # 線性項（初始設為 0）
        'C': I_mean,  # 常數項設為平均值
        'd': np.mean(phi_ext)  # 偏移設為磁通中心
    }
    
    if model_type == 'full':
        estimated_params['T'] = 0.5  # 非線性參數的保守估計
    
    print(f"📊 估計的初始參數:")
    for key, value in estimated_params.items():
        print(f"   {key}: {value:.6e}")
    
    return estimated_params

def validate_parameters(params, model_type='simplified'):
    """
    驗證參數合理性
    
    Parameters:
    -----------
    params : dict
        參數字典
    model_type : str
        模型類型
    
    Returns:
    --------
    dict : 驗證結果
    """
    
    validation_results = {
        'valid': True,
        'warnings': [],
        'errors': []
    }
    
    # 檢查必需參數
    required_params = ['Ic', 'phi_0', 'f', 'k', 'r', 'C', 'd']
    if model_type == 'full':
        required_params.append('T')
    
    for param in required_params:
        if param not in params:
            validation_results['errors'].append(f"缺少參數: {param}")
            validation_results['valid'] = False
    
    if not validation_results['valid']:
        return validation_results
    
    # 物理約束檢查
    if params['Ic'] <= 0:
        validation_results['errors'].append("臨界電流 Ic 必須為正值")
        validation_results['valid'] = False
    
    if params['f'] <= 0:
        validation_results['warnings'].append("頻率 f 為負值或零，可能不合理")
    
    if model_type == 'full':
        if not (0 <= params['T'] <= 1):
            validation_results['warnings'].append("非線性參數 T 通常在 [0, 1] 範圍內")
    
    # 數值範圍檢查
    if abs(params['Ic']) > 1e-3:  # 1 mA
        validation_results['warnings'].append("臨界電流異常大，請檢查單位")
    
    if abs(params['f']) > 1e8:  # 100 MHz
        validation_results['warnings'].append("頻率異常高，請檢查")
    
    return validation_results

def refine_parameters(phi_ext, current, initial_params, model_type='simplified'):
    """
    精細化參數估計
    
    Parameters:
    -----------
    phi_ext : array-like
        外部磁通陣列
    current : array-like
        電流陣列
    initial_params : dict
        初始參數
    model_type : str
        模型類型
    
    Returns:
    --------
    dict : 精細化的參數
    """
    
    from ..models.josephson_physics import full_josephson_model, simplified_josephson_model
    
    phi_ext = np.array(phi_ext)
    current = np.array(current)
    
    # 選擇模型函數
    if model_type == 'full':
        model_func = full_josephson_model
        param_names = ['Ic', 'phi_0', 'f', 'T', 'k', 'r', 'C', 'd']
    else:
        model_func = simplified_josephson_model
        param_names = ['Ic', 'phi_0', 'f', 'k', 'r', 'C', 'd']
    
    # 準備初始猜測
    initial_guess = [initial_params[name] for name in param_names]
    
    try:
        # 使用多次優化以提高穩健性
        best_params = None
        best_residual = np.inf
        
        for attempt in range(3):
            # 添加小幅隨機擾動
            perturbed_guess = [p * (1 + 0.1 * np.random.randn()) for p in initial_guess]
            
            try:
                popt, pcov = curve_fit_compatible(
                    model_func, phi_ext, current, 
                    p0=perturbed_guess, maxfev=5000
                )
                
                # 計算殘差
                fitted_current = model_func(phi_ext, *popt)
                residual = np.sum((current - fitted_current)**2)
                
                if residual < best_residual:
                    best_residual = residual
                    best_params = popt
                    
            except:
                continue
        
        if best_params is not None:
            refined_params = dict(zip(param_names, best_params))
            
            print(f"✅ 參數精細化成功")
            print(f"   殘差改善: {best_residual:.2e}")
            
            return refined_params
        else:
            print("⚠️ 參數精細化失敗，使用初始估計")
            return initial_params
            
    except Exception as e:
        print(f"⚠️ 參數精細化出錯: {e}")
        return initial_params