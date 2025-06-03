"""
數據處理工具

提供數據載入、保存、預處理等功能。
"""

import numpy as np
import pandas as pd
from scipy import interpolate

def load_csv_data(filename, phi_column='phi_ext', current_column='current', 
                 error_column=None):
    """
    從 CSV 文件載入數據
    
    Parameters:
    -----------
    filename : str
        文件名
    phi_column : str
        磁通列名
    current_column : str
        電流列名
    error_column : str, optional
        誤差列名
    
    Returns:
    --------
    tuple : (phi_ext, current, errors)
    """
    
    try:
        df = pd.read_csv(filename)
        
        phi_ext = df[phi_column].values
        current = df[current_column].values
        
        if error_column and error_column in df.columns:
            errors = df[error_column].values
        else:
            errors = None
        
        print(f"✅ 成功載入數據: {filename}")
        print(f"   數據點數: {len(phi_ext)}")
        print(f"   磁通範圍: {phi_ext.min():.2e} 到 {phi_ext.max():.2e}")
        print(f"   電流範圍: {current.min():.2e} 到 {current.max():.2e}")
        
        return phi_ext, current, errors
        
    except Exception as e:
        print(f"❌ 載入數據失敗: {e}")
        return None, None, None

def save_analysis_results(results, filename):
    """
    保存分析結果到文件
    
    Parameters:
    -----------
    results : dict
        分析結果字典
    filename : str
        輸出文件名
    """
    
    try:
        # 轉換為可序列化的格式
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, dict):
                serializable_results[key] = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                           for k, v in value.items()}
            else:
                serializable_results[key] = value
        
        # 保存為 JSON
        import json
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"✅ 分析結果已保存: {filename}")
        
    except Exception as e:
        print(f"❌ 保存失敗: {e}")

def interpolate_uniform_grid(x, y, n_points=None):
    """
    將數據插值到均勻網格
    
    Parameters:
    -----------
    x : array-like
        自變量陣列
    y : array-like
        因變量陣列
    n_points : int, optional
        輸出點數（默認與輸入相同）
    
    Returns:
    --------
    tuple : (x_uniform, y_uniform)
    """
    
    x = np.array(x)
    y = np.array(y)
    
    if n_points is None:
        n_points = len(x)
    
    # 檢查是否已經是均勻的
    if np.allclose(np.diff(x), np.diff(x)[0], rtol=1e-6):
        print("數據已經是均勻間距")
        return x, y
    
    # 創建插值函數
    f_interp = interpolate.interp1d(x, y, kind='linear', bounds_error=False, fill_value='extrapolate')
    
    # 創建均勻網格
    x_uniform = np.linspace(x.min(), x.max(), n_points)
    y_uniform = f_interp(x_uniform)
    
    print(f"✅ 數據已插值到 {n_points} 個均勻點")
    
    return x_uniform, y_uniform

def remove_outliers(x, y, method='iqr', threshold=1.5):
    """
    移除異常值
    
    Parameters:
    -----------
    x : array-like
        自變量陣列
    y : array-like
        因變量陣列
    method : str
        異常值檢測方法 ('iqr', 'zscore')
    threshold : float
        閾值
    
    Returns:
    --------
    tuple : (x_clean, y_clean, outlier_mask)
    """
    
    x = np.array(x)
    y = np.array(y)
    
    if method == 'iqr':
        Q1 = np.percentile(y, 25)
        Q3 = np.percentile(y, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outlier_mask = (y >= lower_bound) & (y <= upper_bound)
        
    elif method == 'zscore':
        z_scores = np.abs((y - np.mean(y)) / np.std(y))
        outlier_mask = z_scores <= threshold
        
    else:
        raise ValueError(f"不支援的方法: {method}")
    
    x_clean = x[outlier_mask]
    y_clean = y[outlier_mask]
    
    n_outliers = len(x) - len(x_clean)
    print(f"✅ 移除 {n_outliers} 個異常值 ({n_outliers/len(x)*100:.1f}%)")
    
    return x_clean, y_clean, outlier_mask

def smooth_data(x, y, window_size=5, method='moving_average'):
    """
    平滑數據
    
    Parameters:
    -----------
    x : array-like
        自變量陣列
    y : array-like
        因變量陣列
    window_size : int
        窗口大小
    method : str
        平滑方法 ('moving_average', 'savgol')
    
    Returns:
    --------
    tuple : (x, y_smoothed)
    """
    
    y = np.array(y)
    
    if method == 'moving_average':
        y_smoothed = np.convolve(y, np.ones(window_size)/window_size, mode='same')
        
    elif method == 'savgol':
        from scipy.signal import savgol_filter
        y_smoothed = savgol_filter(y, window_size, 2)
        
    else:
        raise ValueError(f"不支援的平滑方法: {method}")
    
    print(f"✅ 數據已平滑 (窗口大小: {window_size})")
    
    return x, y_smoothed