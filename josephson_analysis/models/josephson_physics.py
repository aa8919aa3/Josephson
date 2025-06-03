"""
Josephson 結物理模型和分析器

這個模組包含了 Josephson 結的物理模型以及相關的分析工具。
主要功能包括磁通響應模擬、週期性分析和參數估計。
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# 導入 lmfit 工具
from ..utils.lmfit_tools import lmfit_curve_fit, curve_fit_compatible

# 物理常數
FLUX_QUANTUM = 2.067833831e-15  # 磁通量子 Φ₀ (Wb)
PLANCK_CONSTANT = 6.62607015e-34  # 普朗克常數 (J⋅s)
ELEMENTARY_CHARGE = 1.602176634e-19  # 基本電荷 (C)

def full_josephson_model(phi_ext, Ic, phi_0, f, T, k, r, C, d):
    """
    完整的 Josephson 結模型，包含非線性項
    
    Parameters:
    -----------
    phi_ext : array-like
        外部磁通
    Ic : float
        臨界電流
    phi_0 : float
        相位偏移
    f : float
        週期頻率參數
    T : float
        非線性參數
    k : float
        二次項係數
    r : float
        線性項係數
    C : float
        常數項
    d : float
        偏移量
    
    Returns:
    --------
    array-like : 超導電流響應
    """
    phase_term = 2 * np.pi * f * (phi_ext - d) - phi_0
    
    term1 = Ic * np.sin(phase_term)
    
    # 計算非線性分母項，確保數值穩定性
    sin_half = np.sin(phase_term / 2)
    denominator_arg = 1 - T * sin_half**2
    denominator_arg = np.maximum(denominator_arg, 1e-12)  # 防止除零
    term2 = np.sqrt(denominator_arg)
    
    term3 = k * (phi_ext - d)**2
    term4 = r * (phi_ext - d)
    
    return term1 / term2 + term3 + term4 + C

def simplified_josephson_model(phi_ext, Ic, phi_0, f, k, r, C, d):
    """
    簡化的 Josephson 結模型（純正弦模型）
    
    Parameters:
    -----------
    phi_ext : array-like
        外部磁通
    Ic : float
        臨界電流
    phi_0 : float
        相位偏移
    f : float
        週期頻率參數
    k : float
        二次項係數
    r : float
        線性項係數
    C : float
        常數項
    d : float
        偏移量
    
    Returns:
    --------
    array-like : 超導電流響應
    """
    phase_term = 2 * np.pi * f * (phi_ext - d) - phi_0
    
    term1 = Ic * np.sin(phase_term)
    term3 = k * (phi_ext - d)**2
    term4 = r * (phi_ext - d)
    
    return term1 + term3 + term4 + C

class JosephsonPeriodicAnalyzer:
    """
    Josephson 結週期性信號分析器
    
    提供完整的磁通響應分析功能，包括數據生成、週期檢測、
    參數估計和統計評估。
    """
    
    def __init__(self, save_data=True):
        """
        初始化分析器
        
        Parameters:
        -----------
        save_data : bool
            是否自動保存生成的數據
        """
        self.save_data = save_data
        self.simulation_results = {}
        self.analysis_results = {}
        self.default_params = {
            'Ic': 1.0e-6,           # 臨界電流 (A)
            'phi_0': np.pi / 4,     # 相位偏移 (rad)
            'f': 5e4,               # 週期頻率參數
            'T': 0.8,               # 非線性參數
            'k': -0.01,             # 二次項係數
            'r': 5e-3,              # 線性項係數
            'C': 10.0e-6,           # 常數項 (A)
            'd': -10.0e-3,          # 偏移量
            'noise_level': 2e-7     # 雜訊強度 (A)
        }
    
    def generate_flux_sweep_data(self, phi_range=(-20e-5, 0e-5), n_points=500, 
                               model_type="both", **params):
        """
        生成磁通掃描數據
        
        Parameters:
        -----------
        phi_range : tuple
            外部磁通範圍
        n_points : int
            數據點數
        model_type : str
            模型類型 ("full", "simplified", "both")
        **params : dict
            模型參數
        
        Returns:
        --------
        dict : 包含磁通和電流數據的字典
        """
        print("🧲 生成 Josephson 結磁通掃描數據")
        print("="*50)
        
        # 更新參數
        model_params = self.default_params.copy()
        model_params.update(params)
        
        # 生成磁通陣列
        phi_ext = np.linspace(phi_range[0], phi_range[1], n_points)
        
        results = {'phi_ext': phi_ext, 'parameters': model_params}
        
        models_to_generate = ['full', 'simplified'] if model_type == "both" else [model_type]
        
        for model in models_to_generate:
            if model == 'full':
                # 過濾出 full_josephson_model 需要的參數，排除 noise_level
                physics_params = {k: v for k, v in model_params.items() 
                                if k in ['Ic', 'phi_0', 'f', 'T', 'k', 'r', 'C', 'd']}
                I_theory = full_josephson_model(phi_ext, **physics_params)
                model_name = '完整非線性模型'
            else:
                # 簡化模型不使用 T 和 noise_level 參數
                simple_params = {k: v for k, v in model_params.items() 
                               if k in ['Ic', 'phi_0', 'f', 'k', 'r', 'C', 'd']}
                I_theory = simplified_josephson_model(phi_ext, **simple_params)
                model_name = '簡化正弦模型'
            
            # 添加雜訊
            noise = model_params['noise_level'] * np.random.normal(size=phi_ext.shape)
            I_noisy = I_theory + noise
            
            # 保存結果
            results[f'{model}_model'] = {
                'name': model_name,
                'I_theory': I_theory,
                'I_noisy': I_noisy,
                'errors': np.full_like(I_noisy, model_params['noise_level'])
            }
            
            print(f"✅ 已生成 {model_name}")
            print(f"   電流範圍: {I_theory.min():.2e} 到 {I_theory.max():.2e} A")
            print(f"   SNR: {np.std(I_theory)/model_params['noise_level']:.1f}")
        
        self.simulation_results = results
        
        # 保存為 CSV
        if self.save_data:
            self._save_flux_sweep_data(results)
        
        print(f"\n🔬 實驗參數摘要:")
        print(f"   磁通範圍: {phi_range[0]:.2e} 到 {phi_range[1]:.2e}")
        print(f"   數據點數: {n_points}")
        print(f"   理論週期: {1/model_params['f']:.2e}")
        print(f"   雜訊水平: {model_params['noise_level']:.2e} A")
        
        return results
    
    def _save_flux_sweep_data(self, results):
        """保存磁通掃描數據到 CSV"""
        for model_key in results.keys():
            if model_key.endswith('_model'):
                df = pd.DataFrame({
                    'phi_ext': results['phi_ext'],
                    'I_theory': results[model_key]['I_theory'],
                    'I_noisy': results[model_key]['I_noisy'],
                    'errors': results[model_key]['errors']
                })
                
                filename = f'josephson_{model_key}_flux_sweep.csv'
                df.to_csv(filename, index=False)
                print(f"💾 數據已保存: {filename}")
    
    def analyze_periodicity(self, phi_ext=None, current=None, model_type="both"):
        """
        分析磁通響應的週期性
        
        Parameters:
        -----------
        phi_ext : array-like, optional
            外部磁通數據（如果未提供，使用模擬數據）
        current : array-like, optional
            電流數據（如果未提供，使用模擬數據）
        model_type : str
            要分析的模型類型
        
        Returns:
        --------
        dict : 週期性分析結果
        """
        from ..analysis.periodicity import enhanced_lomb_scargle_analysis, fft_period_analysis
        
        if phi_ext is None or current is None:
            if not self.simulation_results:
                raise ValueError("請先生成模擬數據或提供 phi_ext 和 current 參數")
            
            phi_ext = self.simulation_results['phi_ext']
            
        print("\n🔍 週期性分析")
        print("="*40)
        
        models_to_analyze = ['full_model', 'simplified_model'] if model_type == "both" else [f'{model_type}_model']
        
        for model in models_to_analyze:
            if model in self.simulation_results:
                if current is None:
                    current_data = self.simulation_results[model]['I_noisy']
                    errors = self.simulation_results[model]['errors']
                else:
                    current_data = current
                    errors = None
                
                print(f"\n📊 {self.simulation_results[model]['name']}")
                
                # Lomb-Scargle 分析
                ls_results = enhanced_lomb_scargle_analysis(
                    phi_ext, current_data, errors,
                    detrend_order=2
                )
                
                # FFT 分析
                fft_results = fft_period_analysis(phi_ext, current_data)
                
                # 合併結果
                analysis_result = {
                    'lomb_scargle': ls_results,
                    'fft': fft_results,
                    'model_name': self.simulation_results[model]['name']
                }
                
                self.analysis_results[model] = analysis_result
                
                # 打印主要結果
                self._print_periodicity_results(model)
        
        return self.analysis_results
    
    def _print_periodicity_results(self, model_key):
        """打印週期性分析結果"""
        if model_key not in self.analysis_results:
            return
        
        result = self.analysis_results[model_key]
        ls_result = result['lomb_scargle']
        fft_result = result['fft']
        true_freq = self.simulation_results['parameters']['f']
        
        print(f"   Lomb-Scargle 分析:")
        print(f"     檢測頻率: {ls_result['best_frequency']:.2e}")
        print(f"     頻率誤差: {abs(ls_result['best_frequency'] - true_freq)/true_freq*100:.2f}%")
        print(f"     最大功率: {ls_result['best_power']:.4f}")
        print(f"     R²: {ls_result['statistics']['r_squared']:.6f}")
        
        print(f"   FFT 分析:")
        print(f"     主要頻率: {fft_result['dominant_frequency']:.2e}")
        print(f"     週期: {1/fft_result['dominant_frequency']:.2e}")
        print(f"     功率比: {fft_result['power_ratio']:.2f}")
    
    def fit_model_parameters(self, phi_ext=None, current=None, model_type='full'):
        """
        擬合模型參數
        
        Parameters:
        -----------
        phi_ext : array-like, optional
            外部磁通數據
        current : array-like, optional
            電流數據
        model_type : str
            模型類型 ('full' 或 'simplified')
        
        Returns:
        --------
        dict : 擬合結果
        """
        if phi_ext is None or current is None:
            if not self.simulation_results:
                raise ValueError("請提供數據或先生成模擬數據")
            
            phi_ext = self.simulation_results['phi_ext']
            current = self.simulation_results[f'{model_type}_model']['I_noisy']
        
        print(f"\n🔧 {model_type.upper()} 模型參數擬合")
        print("="*40)
        
        # 選擇擬合函數和初始參數
        if model_type == 'full':
            fit_func = full_josephson_model
            param_names = ['Ic', 'phi_0', 'f', 'T', 'k', 'r', 'C', 'd']
            initial_guess = [self.default_params[name] for name in param_names]
        else:
            fit_func = simplified_josephson_model
            param_names = ['Ic', 'phi_0', 'f', 'k', 'r', 'C', 'd']
            initial_guess = [self.default_params[name] for name in param_names if name != 'T']
        
        try:
            # 準備 lmfit 參數
            initial_params = {name: val for name, val in zip(param_names, initial_guess)}
            
            # 執行非線性擬合（使用 L-BFGS-B 演算法）
            best_params, param_errors, fitted_current, fit_report = lmfit_curve_fit(
                fit_func, phi_ext, current, initial_params, method='lbfgsb'
            )
            
            # 為兼容性創建 popt 和 pcov
            popt = np.array([best_params[name] for name in param_names])
            param_errors_array = np.array([param_errors[name] for name in param_names])
            pcov = np.diag(param_errors_array**2)
            
            # 計算擬合值和統計
            # fitted_current 已由 lmfit_curve_fit 返回
            
            from ..analysis.statistics import ModelStatistics
            stats = ModelStatistics(
                y_true=current,
                y_pred=fitted_current,
                n_params=len(popt),
                model_name=f"擬合{model_type}模型"
            )
            
            fit_results = {
                'parameters': dict(zip(param_names, popt)),
                'covariance': pcov,
                'fitted_values': fitted_current,
                'statistics': stats,
                'initial_guess': dict(zip(param_names, initial_guess))
            }
            
            # 打印結果
            self._print_fit_results(fit_results, param_names)
            
            return fit_results
            
        except Exception as e:
            print(f"❌ 擬合失敗: {e}")
            return None
    
    def _print_fit_results(self, fit_results, param_names):
        """打印擬合結果"""
        print("   參數估計結果:")
        
        for name in param_names:
            fitted_val = fit_results['parameters'][name]
            initial_val = fit_results['initial_guess'][name]
            
            # 計算標準誤差
            param_idx = list(fit_results['parameters'].keys()).index(name)
            if fit_results['covariance'] is not None:
                std_err = np.sqrt(fit_results['covariance'][param_idx, param_idx])
            else:
                std_err = 0
            
            # 計算相對誤差
            if abs(initial_val) > 1e-12:
                rel_error = abs(fitted_val - initial_val) / abs(initial_val) * 100
                print(f"     {name}: {fitted_val:.6e} ± {std_err:.2e} (初始: {initial_val:.2e}, 誤差: {rel_error:.1f}%)")
            else:
                print(f"     {name}: {fitted_val:.6e} ± {std_err:.2e}")
        
        stats = fit_results['statistics']
        print(f"   擬合品質:")
        print(f"     R²: {stats.r_squared:.6f}")
        print(f"     RMSE: {stats.rmse:.2e}")
        print(f"     MAE: {stats.mae:.2e}")
    
    def generate_summary_report(self):
        """生成完整的分析摘要報告"""
        print("\n" + "="*80)
        print("📊 Josephson 結週期性信號分析摘要報告")
        print("="*80)
        
        if not self.simulation_results:
            print("❌ 沒有可用的分析數據")
            return
        
        params = self.simulation_results['parameters']
        
        print(f"\n🔬 物理參數:")
        print(f"   臨界電流 Ic:     {params['Ic']:.2e} A")
        print(f"   週期頻率 f:      {params['f']:.2e}")
        print(f"   相位偏移 φ₀:     {params['phi_0']:.3f} rad")
        print(f"   非線性參數 T:    {params['T']}")
        print(f"   雜訊水平:        {params['noise_level']:.2e} A")
        
        phi_ext = self.simulation_results['phi_ext']
        print(f"   磁通範圍:        {phi_ext.min():.2e} 到 {phi_ext.max():.2e}")
        print(f"   數據點數:        {len(phi_ext)}")
        
        if self.analysis_results:
            print(f"\n📈 週期性分析結果:")
            
            comparison_data = []
            for model_key in ['full_model', 'simplified_model']:
                if model_key in self.analysis_results:
                    result = self.analysis_results[model_key]
                    ls_result = result['lomb_scargle']
                    
                    freq_error = abs(ls_result['best_frequency'] - params['f']) / params['f'] * 100
                    
                    comparison_data.append({
                        '模型': result['model_name'],
                        '檢測頻率': f"{ls_result['best_frequency']:.2e}",
                        '頻率誤差(%)': f"{freq_error:.2f}",
                        'LS功率': f"{ls_result['best_power']:.4f}",
                        'R²': f"{ls_result['statistics']['r_squared']:.6f}",
                        'RMSE': f"{ls_result['statistics']['rmse']:.2e}"
                    })
            
            if comparison_data:
                df = pd.DataFrame(comparison_data)
                print("\n" + df.to_string(index=False))
        
        print("\n🎯 物理意義:")
        period = 1 / params['f']
        print(f"   理論週期:        {period:.2e}")
        print(f"   磁通量子關係:    週期 ≈ Φ₀/面積")
        print(f"   量子干涉:        電流調制反映磁通量子化效應")
        
        print("="*80)
