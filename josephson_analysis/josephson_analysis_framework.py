import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from astropy.timeseries import LombScargle
from .utils.lmfit_tools import lmfit_curve_fit, curve_fit_compatible
import warnings
warnings.filterwarnings('ignore')

# 設定 Plotly 為非互動模式，避免終端輸出問題
pio.renderers.default = "json"

class JosephsonAnalyzer:
    """
    Josephson 結分析器，整合數據生成、分析和統計評估
    """
    
    def __init__(self, save_data=True):
        self.save_data = save_data
        self.simulation_results = {}
        self.analysis_results = {}
        
    def generate_josephson_data(self, model_type="full", **params):
        """
        生成 Josephson 結模擬數據
        
        Parameters:
        -----------
        model_type : str
            模型類型 ("full" 或 "simplified")
        **params : dict
            模型參數
        """
        
        # 默認參數
        default_params = {
            'Ic': 1.0,           # 臨界電流
            'phi_0': np.pi/4,    # 相位偏移
            'f': 0.5,            # 頻率
            'T': 0.8,            # 非線性參數
            'k': 0.1,            # 二次項係數
            'r': 0.05,           # 線性項係數
            'C': 0.0,            # 常數項
            'd': 0.2,            # 偏移量
            'phi_range': (-2, 2), # 外部磁通範圍
            'n_points': 500,     # 數據點數
            'noise_level': 0.05  # 雜訊強度
        }
        
        # 更新參數
        for key, value in params.items():
            if key in default_params:
                default_params[key] = value
        
        # 生成外部磁通
        Phi_ext = np.linspace(default_params['phi_range'][0], 
                             default_params['phi_range'][1], 
                             default_params['n_points'])
        
        # 計算理論值
        if model_type == "full":
            I_theory = self._full_josephson_model(Phi_ext, **default_params)
            model_name = "完整 Josephson 模型"
        else:
            I_theory = self._simplified_josephson_model(Phi_ext, **default_params)
            model_name = "簡化 Josephson 模型"
        
        # 添加雜訊
        noise = default_params['noise_level'] * np.random.normal(size=Phi_ext.shape)
        I_noisy = I_theory + noise
        
        # 計算誤差（假設已知雜訊水平）
        errors = np.full_like(I_noisy, default_params['noise_level'])
        
        # 保存結果
        result = {
            'model_type': model_type,
            'model_name': model_name,
            'Phi_ext': Phi_ext,
            'I_theory': I_theory,
            'I_noisy': I_noisy,
            'errors': errors,
            'parameters': default_params.copy()
        }
        
        self.simulation_results[model_type] = result
        
        # 保存為 CSV（可選）
        if self.save_data:
            self._save_to_csv(result, model_type)
        
        print(f"✅ 已生成 {model_name} 數據:")
        print(f"   外部磁通範圍: {default_params['phi_range']}")
        print(f"   數據點數: {default_params['n_points']}")
        print(f"   雜訊水平: {default_params['noise_level']}")
        print(f"   頻率: {default_params['f']}")
        
        return result
    
    def _full_josephson_model(self, Phi_ext, **params):
        """完整的 Josephson 模型"""
        Ic, phi_0, f, T, k, r, C, d = [params[key] for key in 
                                       ['Ic', 'phi_0', 'f', 'T', 'k', 'r', 'C', 'd']]
        
        phase_term = 2 * np.pi * f * (Phi_ext - d) - phi_0
        
        term1 = Ic * np.sin(phase_term)
        
        # 避免除零和負數開方
        sin_half_phase = np.sin(phase_term / 2)
        denominator_arg = 1 - T * sin_half_phase**2
        
        # 確保分母為正
        denominator_arg = np.maximum(denominator_arg, 1e-10)
        term2 = np.sqrt(denominator_arg)
        
        term3 = k * (Phi_ext - d)**2
        term4 = r * (Phi_ext - d)
        
        return term1 / term2 + term3 + term4 + C
    
    def _simplified_josephson_model(self, Phi_ext, **params):
        """簡化的 Josephson 模型（不含 term2）"""
        Ic, phi_0, f, k, r, C, d = [params[key] for key in 
                                    ['Ic', 'phi_0', 'f', 'k', 'r', 'C', 'd']]
        
        phase_term = 2 * np.pi * f * (Phi_ext - d) - phi_0
        
        term1 = Ic * np.sin(phase_term)
        term3 = k * (Phi_ext - d)**2
        term4 = r * (Phi_ext - d)
        
        return term1 + term3 + term4 + C
    
    def _save_to_csv(self, result, model_type):
        """保存數據到 CSV 文件"""
        df = pd.DataFrame({
            'Phi_ext': result['Phi_ext'],
            'I_theory': result['I_theory'],
            'I_noisy': result['I_noisy'],
            'errors': result['errors']
        })
        
        filename = f'josephson_{model_type}_data.csv'
        df.to_csv(filename, index=False)
        print(f"📁 數據已保存到: {filename}")
    
    def analyze_with_lomb_scargle(self, model_type, detrend_order=1):
        """
        使用 Lomb-Scargle 分析 Josephson 數據
        """
        if model_type not in self.simulation_results:
            print(f"❌ 找不到 {model_type} 模型的模擬數據")
            return None
        
        data = self.simulation_results[model_type]
        times = data['Phi_ext']  # 將外部磁通當作"時間"軸
        values = data['I_noisy']
        errors = data['errors']
        
        print(f"\n🔬 開始 Lomb-Scargle 分析 - {data['model_name']}")
        
        # 去趨勢化
        detrended_values = values.copy()
        trend_coeffs = None
        if detrend_order > 0:
            trend_coeffs = np.polyfit(times, values, detrend_order)
            trend = np.polyval(trend_coeffs, times)
            detrended_values = values - trend
            print(f"✅ 已應用 {detrend_order} 階多項式去趨勢化")
        
        # Lomb-Scargle 分析
        ls = LombScargle(times, detrended_values, dy=errors, 
                        fit_mean=True, center_data=True)
        
        # 自動確定頻率範圍
        time_span = times.max() - times.min()
        min_freq = 0.5 / time_span
        median_dt = np.median(np.diff(np.sort(times)))
        max_freq = 0.5 / median_dt
        
        # 計算週期圖
        frequency, power = ls.autopower(minimum_frequency=min_freq,
                                      maximum_frequency=max_freq,
                                      samples_per_peak=10)
        
        # 找到最佳頻率
        best_idx = np.argmax(power)
        best_frequency = frequency[best_idx]
        best_period = 1.0 / best_frequency
        best_power = power[best_idx]
        
        # 計算模型參數
        model_params = ls.model_parameters(best_frequency)
        amplitude = np.sqrt(model_params[0]**2 + model_params[1]**2)
        phase = np.arctan2(model_params[1], model_params[0])
        offset = ls.offset()
        
        # 計算擬合值
        ls_model_detrended = ls.model(times, best_frequency)
        if trend_coeffs is not None:
            ls_model_original = ls_model_detrended + np.polyval(trend_coeffs, times)
        else:
            ls_model_original = ls_model_detrended
        
        # 統計評估
        stats = ModelStatistics(
            y_true=values,
            y_pred=ls_model_original,
            n_params=3,  # 頻率、振幅、相位
            model_name=f"LS-{data['model_name']}"
        )
        
        # 保存分析結果
        analysis_result = {
            'frequency': frequency,
            'power': power,
            'best_frequency': best_frequency,
            'best_period': best_period,
            'best_power': best_power,
            'amplitude': amplitude,
            'phase': phase,
            'offset': offset,
            'ls_model': ls_model_original,
            'statistics': stats,
            'true_frequency': data['parameters']['f'],  # 真實頻率
            'ls_object': ls
        }
        
        self.analysis_results[model_type] = analysis_result
        
        # 打印結果
        print(f"\n📊 Lomb-Scargle 分析結果:")
        print(f"   真實頻率: {data['parameters']['f']:.6f}")
        print(f"   檢測頻率: {best_frequency:.6f}")
        print(f"   頻率誤差: {abs(best_frequency - data['parameters']['f']):.6f}")
        print(f"   最佳週期: {best_period:.6f}")
        print(f"   檢測振幅: {amplitude:.6f}")
        print(f"   R²: {stats.r_squared:.6f}")
        
        return analysis_result
    
    def fit_custom_model(self, model_type, use_true_model=True):
        """
        使用自定義模型擬合數據
        
        Parameters:
        -----------
        model_type : str
            模型類型
        use_true_model : bool
            是否使用真實模型結構進行擬合
        """
        if model_type not in self.simulation_results:
            print(f"❌ 找不到 {model_type} 模型的模擬數據")
            return None
        
        data = self.simulation_results[model_type]
        Phi_ext = data['Phi_ext']
        I_noisy = data['I_noisy']
        
        print(f"\n🔧 開始自定義模型擬合 - {data['model_name']}")
        
        if use_true_model:
            # 使用真實的模型結構
            if model_type == "full":
                fit_func = self._fit_full_model
                param_names = ['Ic', 'phi_0', 'f', 'T', 'k', 'r', 'C', 'd']
            else:
                fit_func = self._fit_simplified_model
                param_names = ['Ic', 'phi_0', 'f', 'k', 'r', 'C', 'd']
        else:
            # 使用通用的正弦加多項式模型
            fit_func = self._fit_generic_model
            param_names = ['A', 'f', 'phi', 'a2', 'a1', 'a0']
        
        # 設置初始猜測值
        initial_guess = self._get_initial_guess(model_type, use_true_model)
        
        try:
            # 執行擬合使用 lmfit (L-BFGS-B)
            popt, pcov = curve_fit_compatible(fit_func, Phi_ext, I_noisy, 
                                           p0=initial_guess, maxfev=5000)
            
            # 計算擬合值
            fitted_values = fit_func(Phi_ext, *popt)
            
            # 計算統計
            stats = ModelStatistics(
                y_true=I_noisy,
                y_pred=fitted_values,
                n_params=len(popt),
                model_name=f"Custom-{data['model_name']}"
            )
            
            # 打印擬合結果
            print(f"\n📈 自定義模型擬合結果:")
            for i, (name, value) in enumerate(zip(param_names, popt)):
                std_err = np.sqrt(pcov[i, i]) if pcov is not None else 0
                print(f"   {name}: {value:.6f} ± {std_err:.6f}")
            
            print(f"   R²: {stats.r_squared:.6f}")
            
            return {
                'parameters': dict(zip(param_names, popt)),
                'covariance': pcov,
                'fitted_values': fitted_values,
                'statistics': stats
            }
            
        except Exception as e:
            print(f"❌ 擬合失敗: {e}")
            return None
    
    def _fit_full_model(self, Phi_ext, Ic, phi_0, f, T, k, r, C, d):
        """完整模型擬合函數"""
        return self._full_josephson_model(Phi_ext, Ic=Ic, phi_0=phi_0, f=f, T=T, 
                                        k=k, r=r, C=C, d=d)
    
    def _fit_simplified_model(self, Phi_ext, Ic, phi_0, f, k, r, C, d):
        """簡化模型擬合函數"""
        return self._simplified_josephson_model(Phi_ext, Ic=Ic, phi_0=phi_0, f=f, 
                                              k=k, r=r, C=C, d=d)
    
    def _fit_generic_model(self, x, A, f, phi, a2, a1, a0):
        """通用正弦加多項式模型"""
        return A * np.sin(2 * np.pi * f * x + phi) + a2 * x**2 + a1 * x + a0
    
    def _get_initial_guess(self, model_type, use_true_model):
        """獲取初始猜測值"""
        data = self.simulation_results[model_type]
        params = data['parameters']
        
        if use_true_model:
            if model_type == "full":
                return [params['Ic'], params['phi_0'], params['f'], params['T'],
                       params['k'], params['r'], params['C'], params['d']]
            else:
                return [params['Ic'], params['phi_0'], params['f'],
                       params['k'], params['r'], params['C'], params['d']]
        else:
            # 通用模型的初始猜測
            I_range = np.ptp(data['I_noisy'])
            return [I_range/2, params['f'], 0, 0.01, 0.01, np.mean(data['I_noisy'])]
    
    def plot_comprehensive_analysis(self, model_type):
        """
        繪製完整的分析結果
        """
        if model_type not in self.simulation_results:
            print(f"❌ 找不到 {model_type} 模型的數據")
            return
        
        data = self.simulation_results[model_type]
        analysis = self.analysis_results.get(model_type)
        
        # 建立子圖
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f'{data["model_name"]} - 原始數據',
                'Lomb-Scargle 週期圖',
                '模型擬合比較',
                '殘差分析'
            )
        )
        
        # 1. 原始數據
        fig.add_trace(
            go.Scatter(x=data['Phi_ext'], y=data['I_theory'],
                      mode='lines', name='理論值',
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=data['Phi_ext'], y=data['I_noisy'],
                      mode='markers', name='含雜訊數據',
                      marker=dict(size=3, opacity=0.6, color='red'),
                      error_y=dict(type='data', array=data['errors'], visible=True)),
            row=1, col=1
        )
        
        # 2. Lomb-Scargle 週期圖
        if analysis:
            fig.add_trace(
                go.Scatter(x=analysis['frequency'], y=analysis['power'],
                          mode='lines', name='LS 功率',
                          line=dict(color='green')),
                row=1, col=2
            )
            
            # 標記最佳頻率
            fig.add_trace(
                go.Scatter(x=[analysis['best_frequency']], y=[analysis['best_power']],
                          mode='markers', name=f'最佳頻率 ({analysis["best_frequency"]:.3f})',
                          marker=dict(size=10, color='red', symbol='star')),
                row=1, col=2
            )
            
            # 3. 模型擬合比較
            fig.add_trace(
                go.Scatter(x=data['Phi_ext'], y=data['I_noisy'],
                          mode='markers', name='觀測數據',
                          marker=dict(size=3, opacity=0.6, color='gray')),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=data['Phi_ext'], y=analysis['ls_model'],
                          mode='lines', name=f'LS 擬合 (R²={analysis["statistics"].r_squared:.4f})',
                          line=dict(color='orange', width=2)),
                row=2, col=1
            )
            
            # 4. 殘差分析
            residuals = data['I_noisy'] - analysis['ls_model']
            fig.add_trace(
                go.Scatter(x=data['Phi_ext'], y=residuals,
                          mode='markers', name='LS 殘差',
                          marker=dict(size=4, opacity=0.7, color='purple')),
                row=2, col=2
            )
            
            fig.add_hline(y=0, line_dash="dash", line_color="black", row=2, col=2)
        
        # 更新佈局
        fig.update_layout(
            title_text=f"Josephson 結完整分析 - {data['model_name']}",
            height=800,
            showlegend=True
        )
        
        # 更新軸標題
        fig.update_xaxes(title_text="外部磁通 Φ_ext", row=1, col=1)
        fig.update_yaxes(title_text="電流 I_s", row=1, col=1)
        
        fig.update_xaxes(title_text="頻率", row=1, col=2)
        fig.update_yaxes(title_text="LS 功率", row=1, col=2)
        
        fig.update_xaxes(title_text="外部磁通 Φ_ext", row=2, col=1)
        fig.update_yaxes(title_text="電流 I_s", row=2, col=1)
        
        fig.update_xaxes(title_text="外部磁通 Φ_ext", row=2, col=2)
        fig.update_yaxes(title_text="殘差", row=2, col=2)
        
        fig.show()