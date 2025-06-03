import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from astropy.timeseries import LombScargle
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from josephson_analysis.utils.lmfit_tools import curve_fit_compatible
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 設定 Plotly 為非互動模式，避免終端輸出問題
pio.renderers.default = "json"

# 您提供的物理參數
JOSEPHSON_PARAMS = {
    'Ic': 1.0e-6,           # 臨界電流 (A)
    'phi_0': np.pi / 4,     # 相位偏移 (rad)
    'f': 5e4,               # 頻率 (Hz)
    'T': 0.8,               # 參數T
    'k': -0.01,             # 二次項係數
    'r': 5e-3,              # 線性項係數 
    'C': 10.0e-6,           # 常數項 (A)
    'd': -10.0e-3,          # 偏移量
    'noise_level': 2e-7     # 雜訊強度 (A)
}

class RealisticJosephsonAnalyzer:
    """
    現實物理參數的 Josephson 結分析器
    """
    
    def __init__(self):
        self.phi_ext_range = (-20e-5, 0e-5)  # 外部磁通範圍
        self.n_points = 500
        self.data = {}
        self.analysis_results = {}
    
    def generate_data(self):
        """生成兩種模型的數據"""
        print("🔬 生成現實 Josephson 結數據")
        print("="*50)
        
        # 生成外部磁通
        Phi_ext = np.linspace(self.phi_ext_range[0], self.phi_ext_range[1], self.n_points)
        
        # 完整模型
        I_full = self._full_josephson_model(Phi_ext)
        noise = JOSEPHSON_PARAMS['noise_level'] * np.random.normal(size=Phi_ext.shape)
        I_full_noisy = I_full + noise
        
        # 簡化正弦模型
        I_sine = self._sine_josephson_model(Phi_ext)
        noise = JOSEPHSON_PARAMS['noise_level'] * np.random.normal(size=Phi_ext.shape)
        I_sine_noisy = I_sine + noise
        
        # 保存數據
        self.data = {
            'Phi_ext': Phi_ext,
            'full_model': {
                'name': '完整 Josephson 模型',
                'I_theory': I_full,
                'I_noisy': I_full_noisy,
                'errors': np.full_like(I_full, JOSEPHSON_PARAMS['noise_level'])
            },
            'sine_model': {
                'name': '簡化正弦模型',
                'I_theory': I_sine,
                'I_noisy': I_sine_noisy,
                'errors': np.full_like(I_sine, JOSEPHSON_PARAMS['noise_level'])
            }
        }
        
        # 打印基本信息
        print(f"磁通範圍: {self.phi_ext_range[0]:.2e} 到 {self.phi_ext_range[1]:.2e}")
        print(f"數據點數: {self.n_points}")
        print(f"理論頻率: {JOSEPHSON_PARAMS['f']:.2e} Hz")
        print(f"臨界電流: {JOSEPHSON_PARAMS['Ic']:.2e} A")
        print(f"雜訊水平: {JOSEPHSON_PARAMS['noise_level']:.2e} A")
        
        # 創建 CSV 文件
        self._save_to_csv()
        
        return self.data
    
    def _full_josephson_model(self, Phi_ext):
        """完整的 Josephson 模型"""
        p = JOSEPHSON_PARAMS
        
        phase_term = 2 * np.pi * p['f'] * (Phi_ext - p['d']) - p['phi_0']
        
        term1 = p['Ic'] * np.sin(phase_term)
        
        # 計算分母項，確保數值穩定性
        sin_half = np.sin(phase_term / 2)
        denominator_arg = 1 - p['T'] * sin_half**2
        denominator_arg = np.maximum(denominator_arg, 1e-12)  # 防止除零
        term2 = np.sqrt(denominator_arg)
        
        term3 = p['k'] * (Phi_ext - p['d'])**2
        term4 = p['r'] * (Phi_ext - p['d'])
        
        return term1 / term2 + term3 + term4 + p['C']
    
    def _sine_josephson_model(self, Phi_ext):
        """簡化的正弦模型"""
        p = JOSEPHSON_PARAMS
        
        phase_term = 2 * np.pi * p['f'] * (Phi_ext - p['d']) - p['phi_0']
        
        term1 = p['Ic'] * np.sin(phase_term)
        term3 = p['k'] * (Phi_ext - p['d'])**2
        term4 = p['r'] * (Phi_ext - p['d'])
        
        return term1 + term3 + term4 + p['C']
    
    def _save_to_csv(self):
        """保存數據到 CSV"""
        for model_type in ['full_model', 'sine_model']:
            df = pd.DataFrame({
                'Phi_ext': self.data['Phi_ext'],
                'I_theory': self.data[model_type]['I_theory'],
                'I_noisy': self.data[model_type]['I_noisy'],
                'errors': self.data[model_type]['errors']
            })
            
            filename = f'josephson_{model_type}_realistic.csv'
            df.to_csv(filename, index=False)
            print(f"💾 數據已保存: {filename}")
    
    def lomb_scargle_analysis(self, model_type='both'):
        """執行 Lomb-Scargle 分析"""
        print(f"\n🔍 Lomb-Scargle 週期分析")
        print("="*50)
        
        models_to_analyze = ['full_model', 'sine_model'] if model_type == 'both' else [model_type]
        
        for model in models_to_analyze:
            if model not in self.data:
                continue
                
            print(f"\n📊 分析 {self.data[model]['name']}")
            
            # 準備數據
            times = self.data['Phi_ext']
            values = self.data[model]['I_noisy']
            errors = self.data[model]['errors']
            
            # 去趨勢化（移除線性和二次項趨勢）
            detrend_order = 2  # 使用二次多項式
            trend_coeffs = np.polyfit(times, values, detrend_order)
            trend = np.polyval(trend_coeffs, times)
            detrended_values = values - trend
            
            print(f"   ✓ 應用 {detrend_order} 階多項式去趨勢化")
            print(f"   趨勢係數: {trend_coeffs}")
            
            # 創建 Lomb-Scargle 物件
            ls = LombScargle(times, detrended_values, dy=errors, 
                           fit_mean=True, center_data=True)
            
            # 計算頻率範圍
            time_span = times.max() - times.min()
            min_freq = 1.0 / time_span  # 最低頻率
            max_freq = self.n_points / (2 * time_span)  # 奈奎斯特頻率
            
            print(f"   頻率搜索範圍: {min_freq:.2e} 到 {max_freq:.2e}")
            
            # 計算週期圖
            frequency, power = ls.autopower(
                minimum_frequency=min_freq,
                maximum_frequency=max_freq,
                samples_per_peak=20
            )
            
            # 尋找峰值
            peak_indices = self._find_significant_peaks(frequency, power)
            
            # 分析主要峰值
            best_idx = np.argmax(power)
            best_frequency = frequency[best_idx]
            best_power = power[best_idx]
            best_period = 1.0 / best_frequency
            
            # 計算統計顯著性
            try:
                fap = ls.false_alarm_probability(best_power, method='baluev')
                print(f"   最高峰值 FAP: {fap:.2e}")
            except:
                fap = None
                print("   無法計算 FAP")
            
            # 計算模型參數
            model_params = ls.model_parameters(best_frequency)
            amplitude = np.sqrt(model_params[0]**2 + model_params[1]**2)
            phase = np.arctan2(model_params[1], model_params[0])
            offset = ls.offset()
            
            # 生成擬合模型
            ls_model_detrended = ls.model(times, best_frequency)
            ls_model_full = ls_model_detrended + trend
            
            # 計算統計指標
            stats_obj = self._calculate_enhanced_statistics(values, ls_model_full, n_params=3)
            
            # 保存結果
            self.analysis_results[model] = {
                'frequency': frequency,
                'power': power,
                'best_frequency': best_frequency,
                'best_period': best_period,
                'best_power': best_power,
                'amplitude': amplitude,
                'phase': phase,
                'offset': offset,
                'fap': fap,
                'ls_model': ls_model_full,
                'detrended_values': detrended_values,
                'trend': trend,
                'peak_indices': peak_indices,
                'statistics': stats_obj,
                'ls_object': ls
            }
            
            # 打印結果
            self._print_analysis_results(model)
    
    def _find_significant_peaks(self, frequency, power, n_peaks=10):
        """尋找顯著峰值"""
        from scipy.signal import find_peaks
        
        # 尋找峰值
        mean_power = np.mean(power)
        std_power = np.std(power)
        threshold = mean_power + 2 * std_power
        
        peaks, properties = find_peaks(power, height=threshold, distance=20)
        
        # 按功率排序
        if len(peaks) > 0:
            peak_powers = power[peaks]
            sorted_indices = np.argsort(peak_powers)[::-1]
            return peaks[sorted_indices[:n_peaks]]
        else:
            return []
    
    def _calculate_enhanced_statistics(self, y_true, y_pred, n_params):
        """計算增強統計指標"""
        # 移除 NaN
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        n = len(y_true_clean)
        
        if n == 0:
            return None
        
        # 計算統計量
        residuals = y_true_clean - y_pred_clean
        
        # R-squared
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_true_clean - np.mean(y_true_clean))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Adjusted R-squared
        if n > n_params + 1:
            adj_r_squared = 1 - ((ss_res / (n - n_params - 1)) / (ss_tot / (n - 1)))
        else:
            adj_r_squared = r_squared
        
        # 其他統計量
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))
        
        return {
            'r_squared': r_squared,
            'adjusted_r_squared': adj_r_squared,
            'rmse': rmse,
            'mae': mae,
            'sse': ss_res,
            'n_observations': n,
            'residuals': residuals
        }
    
    def _print_analysis_results(self, model):
        """打印分析結果"""
        result = self.analysis_results[model]
        true_freq = JOSEPHSON_PARAMS['f']
        
        print(f"\n   📈 {self.data[model]['name']} 分析結果:")
        print(f"   {'─'*40}")
        print(f"   真實頻率:     {true_freq:.2e} Hz")
        print(f"   檢測頻率:     {result['best_frequency']:.2e}")
        print(f"   頻率誤差:     {abs(result['best_frequency'] - true_freq)/true_freq*100:.2f}%")
        print(f"   檢測週期:     {result['best_period']:.2e}")
        print(f"   檢測振幅:     {result['amplitude']:.2e} A")
        print(f"   相位:        {result['phase']:.3f} rad")
        print(f"   最大功率:     {result['best_power']:.4f}")
        
        if result['statistics']:
            stats = result['statistics']
            print(f"   R²:          {stats['r_squared']:.6f}")
            print(f"   調整後R²:     {stats['adjusted_r_squared']:.6f}")
            print(f"   RMSE:        {stats['rmse']:.2e} A")
            print(f"   MAE:         {stats['mae']:.2e} A")
    
    def plot_comprehensive_analysis(self):
        """繪製完整分析結果"""
        if not self.analysis_results:
            print("❌ 請先執行 Lomb-Scargle 分析")
            return
        
        # 創建大型綜合圖表
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                '完整模型 - 原始數據', '簡化模型 - 原始數據',
                '完整模型 - 週期圖', '簡化模型 - 週期圖', 
                '模型比較', '殘差分析'
            ),
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        colors = {'full_model': 'blue', 'sine_model': 'red'}
        
        # 第一行：原始數據
        for i, model in enumerate(['full_model', 'sine_model']):
            col = i + 1
            
            # 理論曲線
            fig.add_trace(
                go.Scatter(
                    x=self.data['Phi_ext'] * 1e5,  # 轉換為 10^-5 單位
                    y=self.data[model]['I_theory'] * 1e6,  # 轉換為 μA
                    mode='lines',
                    name=f'{self.data[model]["name"]} (理論)',
                    line=dict(color=colors[model], width=2)
                ),
                row=1, col=col
            )
            
            # 含雜訊數據
            fig.add_trace(
                go.Scatter(
                    x=self.data['Phi_ext'] * 1e5,
                    y=self.data[model]['I_noisy'] * 1e6,
                    mode='markers',
                    name=f'{self.data[model]["name"]} (含雜訊)',
                    marker=dict(size=2, opacity=0.6, color=colors[model]),
                    error_y=dict(
                        type='data', 
                        array=self.data[model]['errors'] * 1e6,
                        visible=True
                    )
                ),
                row=1, col=col
            )
        
        # 第二行：週期圖
        for i, model in enumerate(['full_model', 'sine_model']):
            if model in self.analysis_results:
                col = i + 1
                result = self.analysis_results[model]
                
                # 週期圖
                fig.add_trace(
                    go.Scatter(
                        x=result['frequency'],
                        y=result['power'],
                        mode='lines',
                        name=f'{self.data[model]["name"]} 功率譜',
                        line=dict(color=colors[model])
                    ),
                    row=2, col=col
                )
                
                # 標記最佳頻率
                fig.add_trace(
                    go.Scatter(
                        x=[result['best_frequency']],
                        y=[result['best_power']],
                        mode='markers',
                        name=f'最佳頻率 ({result["best_frequency"]:.2e})',
                        marker=dict(size=10, color='red', symbol='star')
                    ),
                    row=2, col=col
                )
                
                # 標記真實頻率
                true_freq = JOSEPHSON_PARAMS['f']
                if true_freq >= result['frequency'].min() and true_freq <= result['frequency'].max():
                    # 找到最接近真實頻率的功率值
                    closest_idx = np.argmin(np.abs(result['frequency'] - true_freq))
                    fig.add_trace(
                        go.Scatter(
                            x=[true_freq],
                            y=[result['power'][closest_idx]],
                            mode='markers',
                            name=f'真實頻率 ({true_freq:.2e})',
                            marker=dict(size=8, color='green', symbol='diamond')
                        ),
                        row=2, col=col
                    )
        
        # 第三行左：模型擬合比較
        for i, model in enumerate(['full_model', 'sine_model']):
            if model in self.analysis_results:
                result = self.analysis_results[model]
                
                # 原始數據
                fig.add_trace(
                    go.Scatter(
                        x=self.data['Phi_ext'] * 1e5,
                        y=self.data[model]['I_noisy'] * 1e6,
                        mode='markers',
                        name=f'{self.data[model]["name"]} 數據',
                        marker=dict(size=2, opacity=0.4, color=colors[model])
                    ),
                    row=3, col=1
                )
                
                # LS 擬合
                fig.add_trace(
                    go.Scatter(
                        x=self.data['Phi_ext'] * 1e5,
                        y=result['ls_model'] * 1e6,
                        mode='lines',
                        name=f'{self.data[model]["name"]} LS擬合 (R²={result["statistics"]["r_squared"]:.4f})',
                        line=dict(color=colors[model], width=2, dash='dash')
                    ),
                    row=3, col=1
                )
        
        # 第三行右：殘差分析
        for i, model in enumerate(['full_model', 'sine_model']):
            if model in self.analysis_results:
                result = self.analysis_results[model]
                residuals = self.data[model]['I_noisy'] - result['ls_model']
                
                fig.add_trace(
                    go.Scatter(
                        x=self.data['Phi_ext'] * 1e5,
                        y=residuals * 1e6,
                        mode='markers',
                        name=f'{self.data[model]["name"]} 殘差',
                        marker=dict(size=3, opacity=0.7, color=colors[model])
                    ),
                    row=3, col=2
                )
        
        # 添加零線到殘差圖
        fig.add_hline(y=0, line_dash="dash", line_color="black", row=3, col=2)
        
        # 更新佈局
        fig.update_layout(
            title_text="現實 Josephson 結完整分析報告",
            height=1200,
            showlegend=True
        )
        
        # 更新軸標題
        fig.update_xaxes(title_text="外部磁通 Φ_ext (×10⁻⁵)", row=1, col=1)
        fig.update_xaxes(title_text="外部磁通 Φ_ext (×10⁻⁵)", row=1, col=2)
        fig.update_yaxes(title_text="電流 I_s (μA)", row=1, col=1)
        fig.update_yaxes(title_text="電流 I_s (μA)", row=1, col=2)
        
        fig.update_xaxes(title_text="頻率 (Hz)", row=2, col=1)
        fig.update_xaxes(title_text="頻率 (Hz)", row=2, col=2)
        fig.update_yaxes(title_text="LS 功率", row=2, col=1)
        fig.update_yaxes(title_text="LS 功率", row=2, col=2)
        
        fig.update_xaxes(title_text="外部磁通 Φ_ext (×10⁻⁵)", row=3, col=1)
        fig.update_xaxes(title_text="外部磁通 Φ_ext (×10⁻⁵)", row=3, col=2)
        fig.update_yaxes(title_text="電流 I_s (μA)", row=3, col=1)
        fig.update_yaxes(title_text="殘差 (μA)", row=3, col=2)
        
        fig.show()
    
    def generate_summary_report(self):
        """生成分析摘要報告"""
        print("\n" + "="*80)
        print("📊 現實 Josephson 結分析摘要報告")
        print("="*80)
        
        print(f"\n🔬 實驗參數:")
        print(f"   臨界電流 Ic:     {JOSEPHSON_PARAMS['Ic']:.2e} A")
        print(f"   理論頻率 f:      {JOSEPHSON_PARAMS['f']:.2e} Hz")
        print(f"   相位偏移 φ₀:     {JOSEPHSON_PARAMS['phi_0']:.3f} rad")
        print(f"   非線性參數 T:    {JOSEPHSON_PARAMS['T']}")
        print(f"   雜訊水平:        {JOSEPHSON_PARAMS['noise_level']:.2e} A")
        print(f"   磁通掃描範圍:    {self.phi_ext_range[0]:.2e} 到 {self.phi_ext_range[1]:.2e}")
        
        if self.analysis_results:
            print(f"\n📈 Lomb-Scargle 分析結果:")
            
            comparison_data = []
            for model in ['full_model', 'sine_model']:
                if model in self.analysis_results:
                    result = self.analysis_results[model]
                    stats = result['statistics']
                    true_freq = JOSEPHSON_PARAMS['f']
                    freq_error = abs(result['best_frequency'] - true_freq) / true_freq * 100
                    
                    comparison_data.append({
                        '模型': self.data[model]['name'],
                        '檢測頻率 (Hz)': f"{result['best_frequency']:.2e}",
                        '頻率誤差 (%)': f"{freq_error:.2f}",
                        'R²': f"{stats['r_squared']:.6f}",
                        '調整後R²': f"{stats['adjusted_r_squared']:.6f}",
                        'RMSE (A)': f"{stats['rmse']:.2e}",
                        '最大功率': f"{result['best_power']:.4f}"
                    })
            
            # 打印比較表
            if comparison_data:
                df = pd.DataFrame(comparison_data)
                print("\n" + df.to_string(index=False))
        
        print(f"\n🎯 分析結論:")
        
        if len(self.analysis_results) == 2:
            full_stats = self.analysis_results['full_model']['statistics']
            sine_stats = self.analysis_results['sine_model']['statistics']
            
            if full_stats['r_squared'] > sine_stats['r_squared']:
                print("   • 完整模型顯示更好的擬合品質")
            else:
                print("   • 簡化模型顯示足夠的擬合品質")
            
            freq_accuracy_full = abs(self.analysis_results['full_model']['best_frequency'] - JOSEPHSON_PARAMS['f']) / JOSEPHSON_PARAMS['f'] * 100
            freq_accuracy_sine = abs(self.analysis_results['sine_model']['best_frequency'] - JOSEPHSON_PARAMS['f']) / JOSEPHSON_PARAMS['f'] * 100
            
            if freq_accuracy_full < 5 and freq_accuracy_sine < 5:
                print("   • 兩個模型都能準確檢測理論頻率")
            elif min(freq_accuracy_full, freq_accuracy_sine) < 10:
                print("   • 頻率檢測具有合理準確性")
            else:
                print("   • 頻率檢測可能受到雜訊或非線性效應影響")
        
        print("="*80)

# 執行分析
def run_complete_analysis():
    """執行完整的分析流程"""
    
    # 創建分析器
    analyzer = RealisticJosephsonAnalyzer()
    
    # 生成數據
    data = analyzer.generate_data()
    
    # 執行 Lomb-Scargle 分析
    analyzer.lomb_scargle_analysis('both')
    
    # 繪製結果
    analyzer.plot_comprehensive_analysis()
    
    # 生成摘要報告
    analyzer.generate_summary_report()
    
    return analyzer

# 執行分析
print("🚀 開始現實 Josephson 結分析")
analyzer = run_complete_analysis()