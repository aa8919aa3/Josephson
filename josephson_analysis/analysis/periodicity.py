"""
週期性分析工具

提供多種週期檢測方法，包括 Lomb-Scargle 週期圖、FFT 分析等。
適用於 Josephson 結磁通響應的週期性特徵提取。
"""

import numpy as np
from astropy.timeseries import LombScargle
from scipy import signal
from .statistics import ModelStatistics

def enhanced_lomb_scargle_analysis(phi_ext, current, errors=None, detrend_order=2):
    """
    增強版 Lomb-Scargle 週期分析
    
    Parameters:
    -----------
    phi_ext : array-like
        外部磁通陣列
    current : array-like  
        電流響應陣列
    errors : array-like, optional
        測量誤差陣列
    detrend_order : int
        去趨勢多項式階數
    
    Returns:
    --------
    dict : 包含分析結果的字典
    """
    print(f"🔍 執行 Lomb-Scargle 週期分析")
    
    # 數據預處理
    phi_ext = np.array(phi_ext)
    current = np.array(current)
    
    if errors is None:
        errors = np.std(current) * 0.1 * np.ones_like(current)
    
    # 去趨勢化
    detrended_current = current.copy()
    trend_coeffs = None
    
    if detrend_order > 0:
        trend_coeffs = np.polyfit(phi_ext, current, detrend_order)
        trend = np.polyval(trend_coeffs, phi_ext)
        detrended_current = current - trend
        print(f"✅ 應用 {detrend_order} 階多項式去趨勢化")
    
    # 建立 Lomb-Scargle 物件
    ls = LombScargle(phi_ext, detrended_current, dy=errors,
                    fit_mean=True, center_data=True)
    
    # 計算頻率範圍
    phi_span = phi_ext.max() - phi_ext.min()
    min_freq = 0.5 / phi_span
    median_dphi = np.median(np.diff(np.sort(phi_ext)))
    max_freq = 0.5 / median_dphi
    
    print(f"頻率搜索範圍: {min_freq:.2e} 到 {max_freq:.2e}")
    
    # 計算週期圖
    frequency, power = ls.autopower(
        minimum_frequency=min_freq,
        maximum_frequency=max_freq,
        samples_per_peak=20
    )
    
    # 尋找最佳頻率
    best_idx = np.argmax(power)
    best_frequency = frequency[best_idx]
    best_power = power[best_idx]
    best_period = 1.0 / best_frequency
    
    # 計算統計顯著性
    try:
        fap = ls.false_alarm_probability(best_power, method='baluev')
    except:
        fap = None
    
    # 計算模型參數
    model_params = ls.model_parameters(best_frequency)
    amplitude = np.sqrt(model_params[0]**2 + model_params[1]**2)
    phase = np.arctan2(model_params[1], model_params[0])
    offset = ls.offset()
    
    # 生成擬合模型
    ls_model_detrended = ls.model(phi_ext, best_frequency)
    
    if trend_coeffs is not None:
        ls_model_full = ls_model_detrended + trend
    else:
        ls_model_full = ls_model_detrended
    
    # 計算統計指標
    stats = ModelStatistics(
        y_true=current,
        y_pred=ls_model_full,
        n_params=3,
        model_name="Lomb-Scargle"
    )
    
    # 尋找其他顯著峰值
    significant_peaks = find_significant_peaks(frequency, power, n_peaks=5)
    
    results = {
        'frequency': frequency,
        'power': power,
        'best_frequency': best_frequency,
        'best_period': best_period,
        'best_power': best_power,
        'amplitude': amplitude,
        'phase': phase,
        'offset': offset,
        'false_alarm_probability': fap,
        'ls_model': ls_model_full,
        'detrended_current': detrended_current,
        'trend_coeffs': trend_coeffs,
        'significant_peaks': significant_peaks,
        'statistics': stats.get_summary_dict(),
        'ls_object': ls
    }
    
    print(f"✅ 檢測到主要頻率: {best_frequency:.2e}")
    print(f"   對應週期: {best_period:.2e}")
    print(f"   最大功率: {best_power:.4f}")
    if fap is not None:
        print(f"   虛警概率: {fap:.2e}")
    
    return results

def fft_period_analysis(phi_ext, current, window='hann'):
    """
    FFT 週期分析
    
    Parameters:
    -----------
    phi_ext : array-like
        外部磁通陣列
    current : array-like
        電流響應陣列
    window : str
        窗函數類型
    
    Returns:
    --------
    dict : FFT 分析結果
    """
    print(f"⚡ 執行 FFT 週期分析")
    
    phi_ext = np.array(phi_ext)
    current = np.array(current)
    
    # 確保等間距（插值如果需要）
    if not np.allclose(np.diff(phi_ext), np.diff(phi_ext)[0], rtol=1e-3):
        print("⚠️ 磁通數據不等間距，執行插值")
        phi_uniform = np.linspace(phi_ext.min(), phi_ext.max(), len(phi_ext))
        current = np.interp(phi_uniform, phi_ext, current)
        phi_ext = phi_uniform
    
    # 應用窗函數
    if window:
        window_func = signal.get_window(window, len(current))
        windowed_current = current * window_func
        print(f"✅ 應用 {window} 窗函數")
    else:
        windowed_current = current
    
    # 計算 FFT
    fft_values = np.fft.fft(windowed_current)
    fft_power = np.abs(fft_values)**2
    
    # 頻率陣列
    dphi = np.mean(np.diff(phi_ext))
    frequencies = np.fft.fftfreq(len(current), dphi)
    
    # 只取正頻率部分
    positive_mask = frequencies > 0
    frequencies = frequencies[positive_mask]
    fft_power = fft_power[positive_mask]
    
    # 尋找主要頻率
    dominant_idx = np.argmax(fft_power)
    dominant_frequency = frequencies[dominant_idx]
    dominant_power = fft_power[dominant_idx]
    
    # 計算功率比（主峰與平均功率的比值）
    mean_power = np.mean(fft_power)
    power_ratio = dominant_power / mean_power
    
    # 尋找前幾個峰值
    peak_indices = signal.find_peaks(fft_power, height=mean_power*2)[0]
    top_peaks = peak_indices[np.argsort(fft_power[peak_indices])[-5:]]
    
    results = {
        'frequencies': frequencies,
        'power_spectrum': fft_power,
        'dominant_frequency': dominant_frequency,
        'dominant_period': 1.0 / dominant_frequency,
        'dominant_power': dominant_power,
        'power_ratio': power_ratio,
        'peak_frequencies': frequencies[top_peaks],
        'peak_powers': fft_power[top_peaks],
        'window_used': window
    }
    
    print(f"✅ 主要頻率: {dominant_frequency:.2e}")
    print(f"   對應週期: {1.0/dominant_frequency:.2e}")
    print(f"   功率比: {power_ratio:.2f}")
    
    return results

def find_significant_peaks(frequency, power, n_peaks=10):
    """
    尋找功率譜中的顯著峰值
    
    Parameters:
    -----------
    frequency : array-like
        頻率陣列
    power : array-like
        功率陣列
    n_peaks : int
        返回的峰值數量
    
    Returns:
    --------
    list : 顯著峰值的信息
    """
    mean_power = np.mean(power)
    std_power = np.std(power)
    threshold = mean_power + 2 * std_power
    
    # 尋找峰值
    peaks, properties = signal.find_peaks(
        power, 
        height=threshold, 
        distance=len(power)//50  # 最小間距
    )
    
    if len(peaks) == 0:
        return []
    
    # 按功率排序
    peak_powers = power[peaks]
    sorted_indices = np.argsort(peak_powers)[::-1]
    top_peaks = peaks[sorted_indices[:n_peaks]]
    
    significant_peaks = []
    for peak_idx in top_peaks:
        significant_peaks.append({
            'frequency': frequency[peak_idx],
            'period': 1.0 / frequency[peak_idx],
            'power': power[peak_idx],
            'significance': power[peak_idx] / mean_power
        })
    
    return significant_peaks

def autocorrelation_analysis(phi_ext, current, max_lag=None):
    """
    自相關函數分析
    
    Parameters:
    -----------
    phi_ext : array-like
        外部磁通陣列
    current : array-like
        電流響應陣列
    max_lag : int, optional
        最大滯後數
    
    Returns:
    --------
    dict : 自相關分析結果
    """
    print(f"🔄 執行自相關分析")
    
    current = np.array(current)
    n = len(current)
    
    if max_lag is None:
        max_lag = n // 4
    
    # 計算自相關
    autocorr = np.correlate(current - np.mean(current), 
                           current - np.mean(current), 
                           mode='full')
    
    # 歸一化
    autocorr = autocorr / autocorr[n-1]
    
    # 取正滯後部分
    autocorr = autocorr[n-1:n-1+max_lag]
    lags = np.arange(max_lag)
    
    # 尋找自相關峰值（除了零滯後）
    if len(autocorr) > 1:
        peaks, _ = signal.find_peaks(autocorr[1:], height=0.1)
        peaks = peaks + 1  # 修正索引
    else:
        peaks = []
    
    results = {
        'lags': lags,
        'autocorrelation': autocorr,
        'peaks': peaks,
        'max_autocorr_lag': lags[np.argmax(autocorr[1:])+1] if len(autocorr) > 1 else 0
    }
    
    if len(peaks) > 0:
        first_peak_lag = peaks[0]
        phi_spacing = np.mean(np.diff(phi_ext))
        estimated_period = first_peak_lag * phi_spacing
        print(f"✅ 估計週期: {estimated_period:.2e} (從自相關)")
    
    return results