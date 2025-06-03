#!/usr/bin/env python3
"""
進階物理模型系統
包含溫度效應、非線性項和更複雜的約瑟夫森結物理
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from josephson_analysis.utils.lmfit_tools import curve_fit_compatible
from scipy.optimize import minimize
from scipy.stats import pearsonr
from scipy.special import jv, ellipk, ellipe  # 貝塞爾函數和橢圓積分
import json
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib with English fonts only
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')

class AdvancedJosephsonPhysics:
    """
    進階約瑟夫森結物理模型
    包含溫度效應、非線性項和多種物理機制
    """
    
    def __init__(self):
        # 物理常數
        self.flux_quantum = 2.067833831e-15  # 磁通量子 (Wb)
        self.kB = 1.380649e-23              # 玻爾茲曼常數 (J/K)
        self.e = 1.602176634e-19            # 電子電荷 (C)
        self.hbar = 1.054571817e-34         # 約化普朗克常數 (J·s)
        
    def temperature_suppression_factor(self, T, Tc):
        """
        溫度抑制因子（基於BCS理論）
        """
        if T >= Tc:
            return 0.0
        
        t = T / Tc
        # BCS溫度依賴性的近似
        if t < 0.9:
            return np.sqrt(1 - t**4) * (1.74 * np.sqrt(1 - t))
        else:
            # 接近Tc時的線性近似
            return 3.06 * (1 - t)**(1.5)
    
    def fraunhofer_pattern_exact(self, phi_ext, Ic0, junction_width, junction_length, 
                                penetration_depth=None):
        """
        精確的Fraunhofer衍射圖樣
        考慮結幾何形狀和穿透深度
        """
        # 標準化磁通
        phi_norm = phi_ext / self.flux_quantum
        
        # 幾何因子
        if penetration_depth is not None:
            # 考慮穿透深度的修正
            effective_width = junction_width + 2 * penetration_depth
            beta = np.pi * phi_norm * effective_width / junction_length
        else:
            beta = np.pi * phi_norm * junction_width / junction_length
        
        # Fraunhofer圖樣
        with np.errstate(divide='ignore', invalid='ignore'):
            sinc_term = np.sin(beta) / beta
            sinc_term = np.where(np.abs(beta) < 1e-10, 1.0, sinc_term)
        
        return Ic0 * np.abs(sinc_term)
    
    def thermal_noise_current(self, T, R_normal, bandwidth=1e6):
        """
        熱噪聲電流（Johnson-Nyquist噪聲）
        """
        noise_current_rms = np.sqrt(4 * self.kB * T * bandwidth / R_normal)
        return noise_current_rms
    
    def flux_creep_effect(self, phi_ext, U0, T, attempt_frequency=1e12):
        """
        磁通蠕變效應
        """
        if T == 0:
            return 1.0
        
        kT = self.kB * T
        # 簡化的磁通蠕變模型
        creep_factor = np.exp(-U0 / kT) * attempt_frequency
        return 1.0 / (1.0 + creep_factor * np.abs(phi_ext))
    
    def josephson_inductance_effect(self, phi_ext, Ic, phi_0=0):
        """
        約瑟夫森電感效應對電流的影響
        """
        phase = 2 * np.pi * phi_ext / self.flux_quantum + phi_0
        
        # 約瑟夫森電感 L_J = hbar / (2e * Ic * cos(phase))
        with np.errstate(divide='ignore', invalid='ignore'):
            cos_phase = np.cos(phase)
            cos_phase = np.where(np.abs(cos_phase) < 1e-10, 1e-10, cos_phase)
            
        # 電感效應對電流的修正
        inductance_factor = 1.0 / np.sqrt(1.0 + 0.1 * np.abs(cos_phase))
        return inductance_factor
    
    def advanced_josephson_model(self, phi_ext, Ic0, T=4.2, Tc=9.0, 
                                junction_width=1e-6, junction_length=10e-6,
                                penetration_depth=200e-9, R_normal=10.0,
                                U0_creep=100, phi_0=0, asymmetry=0.0,
                                nonlinearity=0.0, background_slope=0.0):
        """
        進階約瑟夫森結模型
        整合多種物理效應
        """
        # 溫度抑制
        temp_factor = self.temperature_suppression_factor(T, Tc)
        Ic_temp = Ic0 * temp_factor
        
        # Fraunhofer圖樣
        Ic_fraunhofer = self.fraunhofer_pattern_exact(
            phi_ext, Ic_temp, junction_width, junction_length, penetration_depth
        )
        
        # 磁通蠕變效應
        creep_factor = self.flux_creep_effect(phi_ext, U0_creep, T)
        Ic_with_creep = Ic_fraunhofer * creep_factor
        
        # 約瑟夫森電感效應
        inductance_factor = self.josephson_inductance_effect(phi_ext, Ic_temp, phi_0)
        Ic_with_inductance = Ic_with_creep * inductance_factor
        
        # 不對稱性效應
        if asymmetry != 0:
            phase_modulation = 1 + asymmetry * np.cos(4 * np.pi * phi_ext / self.flux_quantum)
            Ic_with_asymmetry = Ic_with_inductance * phase_modulation
        else:
            Ic_with_asymmetry = Ic_with_inductance
        
        # 非線性效應
        if nonlinearity != 0:
            phi_norm = phi_ext / self.flux_quantum
            nonlinear_correction = 1 + nonlinearity * phi_norm**2
            Ic_nonlinear = Ic_with_asymmetry * nonlinear_correction
        else:
            Ic_nonlinear = Ic_with_asymmetry
        
        # 背景線性項
        background = background_slope * phi_ext
        
        return Ic_nonlinear + background
    
    def fit_advanced_model_to_data(self, phi_ext, Ic_measured, initial_params=None):
        """
        將進階模型擬合到實驗數據
        """
        if initial_params is None:
            # 估計初始參數
            Ic0_estimate = np.max(Ic_measured)
            
            initial_params = {
                'Ic0': Ic0_estimate,
                'T': 4.2,
                'Tc': 9.0,
                'junction_width': 1e-6,
                'junction_length': 10e-6,
                'penetration_depth': 200e-9,
                'R_normal': 10.0,
                'U0_creep': 100,
                'phi_0': 0.0,
                'asymmetry': 0.0,
                'nonlinearity': 0.0,
                'background_slope': 0.0
            }
        
        # 定義擬合函數
        def fit_function(phi, Ic0, T, asymmetry, nonlinearity, background_slope):
            return self.advanced_josephson_model(
                phi, Ic0, T=T, Tc=initial_params['Tc'],
                junction_width=initial_params['junction_width'],
                junction_length=initial_params['junction_length'],
                penetration_depth=initial_params['penetration_depth'],
                R_normal=initial_params['R_normal'],
                U0_creep=initial_params['U0_creep'],
                phi_0=initial_params['phi_0'],
                asymmetry=asymmetry,
                nonlinearity=nonlinearity,
                background_slope=background_slope
            )
        
        # 執行擬合
        try:
            bounds = (
                [Ic0_estimate * 0.1, 0.1, -0.5, -1.0, -1e-9],  # 下界
                [Ic0_estimate * 5.0, 20.0, 0.5, 1.0, 1e-9]     # 上界
            )
            
            popt, pcov = curve_fit_compatible(
                fit_function, phi_ext, Ic_measured,
                p0=[initial_params['Ic0'], initial_params['T'], 
                    initial_params['asymmetry'], initial_params['nonlinearity'],
                    initial_params['background_slope']],
                bounds=bounds,
                maxfev=5000
            )
            
            fitted_params = {
                'Ic0': popt[0],
                'T': popt[1],
                'asymmetry': popt[2],
                'nonlinearity': popt[3],
                'background_slope': popt[4]
            }
            
            # 計算擬合品質
            fitted_Ic = fit_function(phi_ext, *popt)
            r_squared = 1 - np.sum((Ic_measured - fitted_Ic)**2) / np.sum((Ic_measured - np.mean(Ic_measured))**2)
            rmse = np.sqrt(np.mean((Ic_measured - fitted_Ic)**2))
            
            return {
                'success': True,
                'fitted_params': fitted_params,
                'fitted_curve': fitted_Ic,
                'r_squared': r_squared,
                'rmse': rmse,
                'covariance': pcov
            }
            
        except Exception as e:
            print(f"擬合失敗: {e}")
            return {'success': False, 'error': str(e)}

class AdvancedSimulationGenerator:
    """
    進階模擬數據生成器
    使用更精確的物理模型
    """
    
    def __init__(self):
        self.physics = AdvancedJosephsonPhysics()
    
    def generate_enhanced_simulation(self, exp_file, output_dir):
        """
        生成增強的模擬數據
        """
        # 載入實驗數據
        exp_data = pd.read_csv(exp_file)
        phi_ext = exp_data['y_field'].values
        Ic_exp = exp_data['Ic'].values
        
        print(f"🔬 為 {exp_file.name} 生成進階模擬數據...")
        
        # 擬合進階模型到實驗數據
        fit_result = self.physics.fit_advanced_model_to_data(phi_ext, Ic_exp)
        
        if fit_result['success']:
            print(f"   擬合成功，R² = {fit_result['r_squared']:.4f}")
            
            # 使用擬合參數生成新的模擬數據
            fitted_params = fit_result['fitted_params']
            
            # 生成理想模擬數據
            Ic_sim_ideal = self.physics.advanced_josephson_model(
                phi_ext,
                fitted_params['Ic0'],
                T=fitted_params['T'],
                asymmetry=fitted_params['asymmetry'],
                nonlinearity=fitted_params['nonlinearity'],
                background_slope=fitted_params['background_slope']
            )
            
            # 添加適當的噪聲
            noise_level = np.std(Ic_exp - fit_result['fitted_curve'])
            noise = np.random.normal(0, noise_level, len(Ic_sim_ideal))
            Ic_sim_noisy = Ic_sim_ideal + noise
            
            # 創建模擬數據框
            sim_data = pd.DataFrame({
                'y_field': phi_ext,
                'Ic': Ic_sim_noisy
            })
            
            # 保存模擬數據
            output_file = output_dir / f"advanced_sim_{exp_file.name}"
            sim_data.to_csv(output_file, index=False)
            
            # 計算與原始實驗數據的相關性
            correlation, p_value = pearsonr(Ic_exp, Ic_sim_noisy)
            
            return {
                'success': True,
                'output_file': output_file,
                'fitted_params': fitted_params,
                'fit_r_squared': fit_result['r_squared'],
                'correlation': correlation,
                'p_value': p_value,
                'noise_level': noise_level
            }
        else:
            print(f"   擬合失敗: {fit_result.get('error', '未知錯誤')}")
            return {'success': False, 'error': fit_result.get('error', '未知錯誤')}

def compare_models_comprehensive(exp_data_dir, basic_sim_dir, advanced_sim_dir, results_dir):
    """
    綜合比較基礎模型和進階模型的性能
    """
    print("\n📊 綜合模型比較分析...")
    
    comparison_results = []
    
    for exp_file in exp_data_dir.glob("*.csv"):
        try:
            # 載入實驗數據
            exp_data = pd.read_csv(exp_file)
            exp_Ic = exp_data['Ic'].values
            
            # 載入基礎模擬數據
            basic_sim_file = basic_sim_dir / f"improved_sim_{exp_file.name}"
            advanced_sim_file = advanced_sim_dir / f"advanced_sim_{exp_file.name}"
            
            results = {'filename': exp_file.name}
            
            if basic_sim_file.exists():
                basic_data = pd.read_csv(basic_sim_file)
                basic_corr, _ = pearsonr(exp_Ic, basic_data['Ic'].values)
                results['basic_correlation'] = basic_corr
            else:
                results['basic_correlation'] = np.nan
                
            if advanced_sim_file.exists():
                advanced_data = pd.read_csv(advanced_sim_file)
                advanced_corr, _ = pearsonr(exp_Ic, advanced_data['Ic'].values)
                results['advanced_correlation'] = advanced_corr
            else:
                results['advanced_correlation'] = np.nan
            
            # 計算改進度
            if not np.isnan(results['basic_correlation']) and not np.isnan(results['advanced_correlation']):
                improvement = results['advanced_correlation'] - results['basic_correlation']
                results['improvement'] = improvement
                print(f"   {exp_file.name}: 基礎={results['basic_correlation']:.4f}, "
                      f"進階={results['advanced_correlation']:.4f}, "
                      f"改進={improvement:.4f}")
            
            comparison_results.append(results)
            
        except Exception as e:
            print(f"❌ 比較 {exp_file.name} 時發生錯誤: {e}")
    
    # 保存比較結果
    comparison_df = pd.DataFrame(comparison_results)
    comparison_file = results_dir / "model_comparison_results.csv"
    comparison_df.to_csv(comparison_file, index=False)
    
    # 生成摘要統計
    basic_mean = comparison_df['basic_correlation'].mean()
    advanced_mean = comparison_df['advanced_correlation'].mean()
    improvement_mean = comparison_df['improvement'].mean()
    
    print(f"\n📈 摘要統計:")
    print(f"   基礎模型平均相關係數: {basic_mean:.4f}")
    print(f"   進階模型平均相關係數: {advanced_mean:.4f}")
    print(f"   平均改進度: {improvement_mean:.4f}")
    
    return comparison_df

def main():
    """
    主執行函數
    """
    print("=== 進階物理模型系統 ===\n")
    
    # 設定路徑
    base_dir = Path("/Users/albert-mac/Code/GitHub/Josephson")
    exp_data_dir = base_dir / "data" / "experimental"
    basic_sim_dir = base_dir / "data" / "simulated"
    advanced_sim_dir = base_dir / "data" / "simulated" / "advanced"
    results_dir = base_dir / "results"
    
    # 創建進階模擬數據目錄
    advanced_sim_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    
    # 創建進階模擬生成器
    generator = AdvancedSimulationGenerator()
    
    # 生成進階模擬數據
    advanced_results = {}
    
    print("🔬 生成進階物理模型模擬數據...")
    for exp_file in exp_data_dir.glob("*.csv"):
        result = generator.generate_enhanced_simulation(exp_file, advanced_sim_dir)
        advanced_results[exp_file.name] = result
    
    # 保存進階模擬結果
    results_file = results_dir / "advanced_simulation_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        # 處理不可序列化的對象
        serializable_results = {}
        for filename, result in advanced_results.items():
            if result['success']:
                serializable_results[filename] = {
                    'fitted_params': result['fitted_params'],
                    'fit_r_squared': result['fit_r_squared'],
                    'correlation': result['correlation'],
                    'p_value': result['p_value'],
                    'noise_level': result['noise_level']
                }
            else:
                serializable_results[filename] = {'success': False, 'error': result.get('error', '未知錯誤')}
        
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print(f"💾 進階模擬結果已保存至: {results_file}")
    
    # 綜合比較分析
    comparison_df = compare_models_comprehensive(
        exp_data_dir, basic_sim_dir, advanced_sim_dir, results_dir
    )
    
    print("\n✅ 進階物理模型分析完成！")

if __name__ == "__main__":
    main()
