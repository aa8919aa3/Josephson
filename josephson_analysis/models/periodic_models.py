"""
週期性信號模型

提供各種週期性信號分析和建模工具。
"""

import numpy as np
from scipy import signal
from ..utils.lmfit_tools import lmfit_curve_fit, curve_fit_compatible

class PeriodicSignalAnalyzer:
    """
    週期性信號分析器
    
    提供多種週期性信號的檢測和分析方法。
    """
    
    def __init__(self):
        self.analysis_results = {}
    
    def extract_dominant_frequency(self, x, y, method='fft'):
        """
        提取主要頻率成分
        
        Parameters:
        -----------
        x : array-like
            自變量陣列
        y : array-like
            因變量陣列
        method : str
            分析方法 ('fft', 'lomb_scargle', 'autocorr')
        
        Returns:
        --------
        dict : 頻率分析結果
        """
        x = np.array(x)
        y = np.array(y)
        
        if method == 'fft':
            return self._fft_frequency_analysis(x, y)
        elif method == 'autocorr':
            return self._autocorr_frequency_analysis(x, y)
        else:
            raise ValueError(f"不支援的方法: {method}")
    
    def _fft_frequency_analysis(self, x, y):
        """FFT 頻率分析"""
        # 確保等間距
        if not np.allclose(np.diff(x), np.diff(x)[0], rtol=1e-3):
            x_uniform = np.linspace(x.min(), x.max(), len(x))
            y = np.interp(x_uniform, x, y)
            x = x_uniform
        
        # 計算 FFT
        fft_vals = np.fft.fft(y - np.mean(y))
        freqs = np.fft.fftfreq(len(y), np.mean(np.diff(x)))
        
        # 只取正頻率
        pos_mask = freqs > 0
        freqs = freqs[pos_mask]
        power = np.abs(fft_vals[pos_mask])**2
        
        # 找主要頻率
        dominant_idx = np.argmax(power)
        dominant_freq = freqs[dominant_idx]
        
        return {
            'dominant_frequency': dominant_freq,
            'dominant_period': 1.0 / dominant_freq,
            'frequencies': freqs,
            'power_spectrum': power,
            'method': 'fft'
        }
    
    def _autocorr_frequency_analysis(self, x, y):
        """自相關頻率分析"""
        y_centered = y - np.mean(y)
        autocorr = np.correlate(y_centered, y_centered, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]  # 歸一化
        
        # 找第一個非零峰值
        peaks, _ = signal.find_peaks(autocorr[1:], height=0.1)
        
        if len(peaks) > 0:
            first_peak = peaks[0] + 1  # 修正索引
            dx = np.mean(np.diff(x))
            period = first_peak * dx
            frequency = 1.0 / period
        else:
            frequency = np.nan
            period = np.nan
        
        return {
            'dominant_frequency': frequency,
            'dominant_period': period,
            'autocorr': autocorr,
            'peaks': peaks,
            'method': 'autocorr'
        }

def extract_periodic_components(x, y, n_components=3):
    """
    提取多個週期性成分
    
    Parameters:
    -----------
    x : array-like
        自變量陣列
    y : array-like
        因變量陣列
    n_components : int
        要提取的成分數量
    
    Returns:
    --------
    dict : 週期性成分分析結果
    """
    x = np.array(x)
    y = np.array(y)
    
    # 確保等間距
    if not np.allclose(np.diff(x), np.diff(x)[0], rtol=1e-3):
        x_uniform = np.linspace(x.min(), x.max(), len(x))
        y = np.interp(x_uniform, x, y)
        x = x_uniform
    
    # FFT 分析
    y_centered = y - np.mean(y)
    fft_vals = np.fft.fft(y_centered)
    freqs = np.fft.fftfreq(len(y), np.mean(np.diff(x)))
    
    # 只取正頻率
    pos_mask = freqs > 0
    freqs = freqs[pos_mask]
    fft_vals = fft_vals[pos_mask]
    power = np.abs(fft_vals)**2
    
    # 找前 n 個峰值
    peak_indices = signal.find_peaks(power, height=np.max(power)*0.05)[0]
    top_peaks = peak_indices[np.argsort(power[peak_indices])[-n_components:]]
    
    components = []
    for i, peak_idx in enumerate(top_peaks):
        freq = freqs[peak_idx]
        amplitude = np.abs(fft_vals[peak_idx]) * 2 / len(y)
        phase = np.angle(fft_vals[peak_idx])
        
        components.append({
            'frequency': freq,
            'period': 1.0 / freq,
            'amplitude': amplitude,
            'phase': phase,
            'power': power[peak_idx]
        })
    
    # 按功率排序
    components.sort(key=lambda c: c['power'], reverse=True)
    
    return {
        'components': components,
        'n_components': len(components),
        'total_power': np.sum(power),
        'explained_power': np.sum([c['power'] for c in components])
    }

def fit_sinusoidal_model(x, y, n_harmonics=1):
    """
    擬合正弦模型
    
    Parameters:
    -----------
    x : array-like
        自變量陣列
    y : array-like
        因變量陣列
    n_harmonics : int
        諧波數量
    
    Returns:
    --------
    dict : 擬合結果
    """
    x = np.array(x)
    y = np.array(y)
    
    # 估計基本頻率
    analyzer = PeriodicSignalAnalyzer()
    freq_result = analyzer.extract_dominant_frequency(x, y, method='fft')
    base_freq = freq_result['dominant_frequency']
    
    # 定義多諧波正弦模型
    def multi_sine_model(x, *params):
        """多諧波正弦模型"""
        offset = params[0]
        result = np.full_like(x, offset)
        
        for i in range(n_harmonics):
            A = params[1 + i*3]      # 振幅
            f = params[2 + i*3]      # 頻率
            phi = params[3 + i*3]    # 相位
            result += A * np.sin(2 * np.pi * f * x + phi)
        
        return result
    
    # 初始猜測
    initial_guess = [np.mean(y)]  # 偏移
    for i in range(n_harmonics):
        initial_guess.extend([
            np.std(y),                    # 振幅
            base_freq * (i + 1),         # 頻率 (基頻和諧波)
            0.0                          # 相位
        ])
    
    try:
        # 轉換為 float 列表
        initial_guess_float = [float(x) for x in initial_guess]
        
        # 執行擬合使用 lmfit (L-BFGS-B)
        popt, pcov = curve_fit_compatible(multi_sine_model, x, y, p0=initial_guess_float, maxfev=5000)
        
        # 計算擬合值
        y_fit = multi_sine_model(x, *popt)
        
        # 提取參數
        offset = popt[0]
        harmonics = []
        for i in range(n_harmonics):
            harmonics.append({
                'amplitude': popt[1 + i*3],
                'frequency': popt[2 + i*3],
                'phase': popt[3 + i*3],
                'period': 1.0 / popt[2 + i*3]
            })
        
        # 計算擬合品質
        residuals = y - y_fit
        r_squared = 1 - np.sum(residuals**2) / np.sum((y - np.mean(y))**2)
        rmse = np.sqrt(np.mean(residuals**2))
        
        return {
            'success': True,
            'offset': offset,
            'harmonics': harmonics,
            'fitted_values': y_fit,
            'residuals': residuals,
            'r_squared': r_squared,
            'rmse': rmse,
            'parameters': popt,
            'covariance': pcov
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def detect_period_modulation(x, y, base_period=None):
    """
    檢測週期調制
    
    Parameters:
    -----------
    x : array-like
        自變量陣列
    y : array-like
        因變量陣列
    base_period : float, optional
        基本週期
    
    Returns:
    --------
    dict : 週期調制分析結果
    """
    x = np.array(x)
    y = np.array(y)
    
    if base_period is None:
        # 自動檢測基本週期
        analyzer = PeriodicSignalAnalyzer()
        freq_result = analyzer.extract_dominant_frequency(x, y)
        base_period = freq_result['dominant_period']
    
    # 計算局部週期
    n_windows = 10
    window_size = len(x) // n_windows
    local_periods = []
    window_centers = []
    
    for i in range(n_windows):
        start_idx = i * window_size
        end_idx = min((i + 1) * window_size, len(x))
        
        if end_idx - start_idx < 10:  # 跳過太小的窗口
            continue
        
        x_window = x[start_idx:end_idx]
        y_window = y[start_idx:end_idx]
        
        try:
            analyzer = PeriodicSignalAnalyzer()
            freq_result = analyzer.extract_dominant_frequency(x_window, y_window)
            local_periods.append(freq_result['dominant_period'])
            window_centers.append(np.mean(x_window))
        except:
            continue
    
    local_periods = np.array(local_periods)
    window_centers = np.array(window_centers)
    
    # 計算週期變化
    if len(local_periods) > 2:
        period_variation = np.std(local_periods) / np.mean(local_periods)
        period_trend = np.polyfit(window_centers, local_periods, 1)[0]
    else:
        period_variation = 0
        period_trend = 0
    
    return {
        'base_period': base_period,
        'local_periods': local_periods,
        'window_centers': window_centers,
        'period_variation': period_variation,
        'period_trend': period_trend,
        'is_modulated': period_variation > 0.05  # 5% 變化閾值
    }