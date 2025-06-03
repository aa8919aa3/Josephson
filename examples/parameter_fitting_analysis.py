#!/usr/bin/env python3
"""
參數擬合分析示例
使用 lmfit 進行約瑟夫遜結的參數擬合
"""

import numpy as np
import sys
from pathlib import Path

# 添加主模組路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

# Josephson 參數定義
JOSEPHSON_PARAMS = {
    'Ic': 1e-6,      # 臨界電流
    'phi_0': 0.1,    # 相位偏移
    'f': 0.5,        # 頻率係數
    'T': 0.8,        # 溫度參數
    'k': 1e-7,       # 二次項係數
    'r': 1e-8,       # 線性項係數
    'C': 1e-9,       # 常數項
    'd': 0.05        # 偏移參數
}

class MockAnalyzer:
    """模擬分析器用於測試"""
    def __init__(self):
        # 生成測試數據
        Phi_ext = np.linspace(-1, 1, 100)
        
        # 全模型
        def full_model(Phi_ext, Ic, phi_0, f, T, k, r, C, d):
            phase_term = 2 * np.pi * f * (Phi_ext - d) - phi_0
            term1 = Ic * np.sin(phase_term)
            sin_half = np.sin(phase_term / 2)
            denominator_arg = np.maximum(1 - T * sin_half**2, 1e-12)
            term2 = np.sqrt(denominator_arg)
            term3 = k * (Phi_ext - d)**2
            term4 = r * (Phi_ext - d)
            return term1 / term2 + term3 + term4 + C
        
        # 簡化模型
        def sine_model(Phi_ext, Ic, phi_0, f, k, r, C, d):
            phase_term = 2 * np.pi * f * (Phi_ext - d) - phi_0
            term1 = Ic * np.sin(phase_term)
            term3 = k * (Phi_ext - d)**2
            term4 = r * (Phi_ext - d)
            return term1 + term3 + term4 + C
        
        # 生成數據
        I_full = full_model(Phi_ext, **JOSEPHSON_PARAMS)
        
        # 為簡化模型移除 T 參數
        sine_params = {k: v for k, v in JOSEPHSON_PARAMS.items() if k != 'T'}
        I_sine = sine_model(Phi_ext, **sine_params)
        
        # 添加噪聲
        noise_level = 0.05 * np.max(I_full)
        I_full_noisy = I_full + np.random.normal(0, noise_level, len(I_full))
        I_sine_noisy = I_sine + np.random.normal(0, noise_level, len(I_sine))
        
        self.data = {
            'Phi_ext': Phi_ext,
            'full_model': {
                'name': '完整約瑟夫遜模型',
                'I_clean': I_full,
                'I_noisy': I_full_noisy
            },
            'sine_model': {
                'name': '簡化正弦模型',
                'I_clean': I_sine,
                'I_noisy': I_sine_noisy
            }
        }

def advanced_parameter_fitting(analyzer):
    """
    高級參數擬合分析
    """
    print("\n🔧 高級參數擬合分析")
    print("="*50)
    
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from josephson_analysis.utils.lmfit_tools import curve_fit_compatible
    
    # 定義擬合函數
    def full_model_fit(Phi_ext, Ic, phi_0, f, T, k, r, C, d):
        phase_term = 2 * np.pi * f * (Phi_ext - d) - phi_0
        term1 = Ic * np.sin(phase_term)
        sin_half = np.sin(phase_term / 2)
        denominator_arg = np.maximum(1 - T * sin_half**2, 1e-12)
        term2 = np.sqrt(denominator_arg)
        term3 = k * (Phi_ext - d)**2
        term4 = r * (Phi_ext - d)
        return term1 / term2 + term3 + term4 + C
    
    def sine_model_fit(Phi_ext, Ic, phi_0, f, k, r, C, d):
        phase_term = 2 * np.pi * f * (Phi_ext - d) - phi_0
        term1 = Ic * np.sin(phase_term)
        term3 = k * (Phi_ext - d)**2
        term4 = r * (Phi_ext - d)
        return term1 + term3 + term4 + C
    
    Phi_ext = analyzer.data['Phi_ext']
    
    # 擬合兩個模型
    for model_type, fit_func in [('full_model', full_model_fit), ('sine_model', sine_model_fit)]:
        print(f"\n📊 {analyzer.data[model_type]['name']} 參數擬合:")
        
        I_noisy = analyzer.data[model_type]['I_noisy']
        
        # 設置初始猜測（基於已知參數）
        if model_type == 'full_model':
            p0 = [JOSEPHSON_PARAMS['Ic'], JOSEPHSON_PARAMS['phi_0'], 
                  JOSEPHSON_PARAMS['f'], JOSEPHSON_PARAMS['T'],
                  JOSEPHSON_PARAMS['k'], JOSEPHSON_PARAMS['r'], 
                  JOSEPHSON_PARAMS['C'], JOSEPHSON_PARAMS['d']]
            param_names = ['Ic', 'phi_0', 'f', 'T', 'k', 'r', 'C', 'd']
        else:
            p0 = [JOSEPHSON_PARAMS['Ic'], JOSEPHSON_PARAMS['phi_0'], 
                  JOSEPHSON_PARAMS['f'], JOSEPHSON_PARAMS['k'], 
                  JOSEPHSON_PARAMS['r'], JOSEPHSON_PARAMS['C'], 
                  JOSEPHSON_PARAMS['d']]
            param_names = ['Ic', 'phi_0', 'f', 'k', 'r', 'C', 'd']
        
        try:
            # 執行擬合使用 lmfit (L-BFGS-B)
            popt, pcov = curve_fit_compatible(fit_func, Phi_ext, I_noisy, p0=p0, maxfev=10000)
            
            # 計算擬合值
            I_fitted = fit_func(Phi_ext, *popt)
            
            # 計算統計
            residuals = I_noisy - I_fitted
            r_squared = 1 - np.sum(residuals**2) / np.sum((I_noisy - np.mean(I_noisy))**2)
            rmse = np.sqrt(np.mean(residuals**2))
            
            print(f"   擬合成功！R² = {r_squared:.6f}, RMSE = {rmse:.2e}")
            print(f"   參數估計結果:")
            
            for i, (name, fitted_val) in enumerate(zip(param_names, popt)):
                true_val = JOSEPHSON_PARAMS.get(name, 'N/A')
                std_err = np.sqrt(pcov[i, i]) if pcov is not None else 0
                
                if isinstance(true_val, (int, float)):
                    error_pct = abs(fitted_val - true_val) / abs(true_val) * 100
                    print(f"     {name}: {fitted_val:.6e} ± {std_err:.2e} (真實值: {true_val:.2e}, 誤差: {error_pct:.1f}%)")
                else:
                    print(f"     {name}: {fitted_val:.6e} ± {std_err:.2e}")
                    
        except Exception as e:
            print(f"   擬合失敗: {e}")

if __name__ == "__main__":
    print("🚀 開始參數擬合分析（使用 lmfit + L-BFGS-B）")
    print("="*60)
    
    # 創建模擬分析器
    analyzer = MockAnalyzer()
    
    # 執行參數擬合分析
    advanced_parameter_fitting(analyzer)
    
    print("\n✅ 參數擬合分析完成！")