# API 參考文檔

## 主要模組概覽

### `josephson_analysis`
主要分析套件，提供完整的約瑟夫森結週期性分析功能。

### `josephson_analysis.models`
物理模型模組，包含各種約瑟夫森結和 SQUID 的數學模型。

### `josephson_analysis.analysis`
分析工具模組，提供週期性檢測、參數擬合和統計分析功能。

### `josephson_analysis.visualization`
視覺化模組，提供專業的科學繪圖功能。

### `josephson_analysis.utils`
工具模組，包含數據處理和參數估計的輔助函數。

---

## 核心類別

### `JosephsonPeriodicAnalyzer`

主要的分析類別，整合所有分析功能。

```python
class JosephsonPeriodicAnalyzer:
    """
    約瑟夫森結週期性分析器
    
    提供完整的磁通調制分析功能，包括週期檢測、參數擬合、
    統計評估和視覺化。
    """
```

#### 初始化

```python
def __init__(self, flux_quantum: float = 2.067833831e-15, 
             temperature: float = 0.01, 
             noise_level: float = 1e-9):
    """
    初始化分析器
    
    Parameters:
    -----------
    flux_quantum : float, optional
        磁通量子，預設值為 2.067833831e-15 Wb
    temperature : float, optional
        工作溫度，預設值為 0.01 K
    noise_level : float, optional
        預期雜訊水準，預設值為 1e-9 A
    """
```

#### 主要方法

##### `generate_flux_sweep_data()`

```python
def generate_flux_sweep_data(self, Ic: float, phi_range: Tuple[float, float],
                           n_points: int = 500, noise_level: float = None,
                           model_type: str = 'simplified') -> Tuple[np.ndarray, np.ndarray]:
    """
    生成模擬的磁通掃描數據
    
    Parameters:
    -----------
    Ic : float
        臨界電流 (A)
    phi_range : Tuple[float, float]
        磁通掃描範圍 (Wb)
    n_points : int, optional
        數據點數，預設 500
    noise_level : float, optional
        雜訊水準，如未指定則使用類別預設值
    model_type : str, optional
        物理模型類型 ('simplified' 或 'full')
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        (磁通值, 電流值)
    """
```

##### `analyze_periodicity()`

```python
def analyze_periodicity(self, phi_ext: np.ndarray, current: np.ndarray,
                       methods: List[str] = ['fft', 'lomb_scargle', 'autocorr']
                       ) -> Dict[str, Any]:
    """
    執行完整的週期性分析
    
    Parameters:
    -----------
    phi_ext : np.ndarray
        外部磁通值 (Wb)
    current : np.ndarray
        測量電流值 (A)
    methods : List[str], optional
        使用的分析方法列表
        
    Returns:
    --------
    Dict[str, Any]
        包含所有分析結果的字典:
        - 'primary_period': 主要週期
        - 'confidence': 置信度
        - 'signal_to_noise_ratio': 信噪比
        - 'fft_results': FFT 分析結果
        - 'lomb_scargle_results': Lomb-Scargle 分析結果
        - 'autocorr_results': 自相關分析結果
    """
```

##### `fit_josephson_model()`

```python
def fit_josephson_model(self, phi_ext: np.ndarray, current: np.ndarray,
                       model_type: str = 'simplified',
                       initial_guess: Dict[str, float] = None,
                       bounds: Dict[str, Tuple[float, float]] = None
                       ) -> Dict[str, Any]:
    """
    擬合約瑟夫森結模型
    
    Parameters:
    -----------
    phi_ext : np.ndarray
        外部磁通值
    current : np.ndarray
        測量電流值
    model_type : str, optional
        模型類型 ('simplified', 'full', 'polynomial')
    initial_guess : Dict[str, float], optional
        初始參數猜測
    bounds : Dict[str, Tuple[float, float]], optional
        參數界限
        
    Returns:
    --------
    Dict[str, Any]
        擬合結果字典:
        - 'parameters': 擬合參數
        - 'parameter_errors': 參數誤差
        - 'fitted_values': 擬合值
        - 'residuals': 殘差
        - 'r_squared': 決定係數
        - 'rmse': 均方根誤差
        - 'aic': 阿凱克信息準則
        - 'bic': 貝葉斯信息準則
    """
```

---

## 模型模組 (`josephson_analysis.models`)

### `JosephsonPhysics`

物理模型類別，實現各種約瑟夫森結模型。

#### 主要方法

##### `simplified_josephson_model()`

```python
def simplified_josephson_model(self, phi_ext: np.ndarray, Ic: float, phi_0: float,
                             freq: float, k: float = 0, r: float = 0,
                             C: float = 0, d: float = 0) -> np.ndarray:
    """
    簡化的約瑟夫森結模型
    
    I_s = Ic * sin(2π*freq*(Φ-d) - φ₀) + k*(Φ-d)² + r*(Φ-d) + C
    
    Parameters:
    -----------
    phi_ext : np.ndarray
        外部磁通值
    Ic : float
        臨界電流
    phi_0 : float
        相位偏移
    freq : float
        頻率參數
    k : float, optional
        二次項係數
    r : float, optional
        線性項係數
    C : float, optional
        常數項
    d : float, optional
        磁通偏移
        
    Returns:
    --------
    np.ndarray
        計算的電流值
    """
```

##### `full_josephson_model()`

```python
def full_josephson_model(self, phi_ext: np.ndarray, Ic: float, phi_0: float,
                        freq: float, T: float, k: float = 0, r: float = 0,
                        C: float = 0, d: float = 0) -> np.ndarray:
    """
    完整的非線性約瑟夫森結模型
    
    I_s = (Ic * sin(2π*freq*(Φ-d) - φ₀)) / √(1 - T*sin²(...)) + k*(Φ-d)² + r*(Φ-d) + C
    
    Parameters:
    -----------
    phi_ext : np.ndarray
        外部磁通值
    Ic : float
        臨界電流
    phi_0 : float
        相位偏移
    freq : float
        頻率參數
    T : float
        非線性參數 (0 < T < 1)
    k : float, optional
        二次項係數
    r : float, optional
        線性項係數
    C : float, optional
        常數項
    d : float, optional
        磁通偏移
        
    Returns:
    --------
    np.ndarray
        計算的電流值
    """
```

##### `dc_squid_response()`

```python
def dc_squid_response(self, phi_ext: np.ndarray, Ic1: float, Ic2: float,
                     asymmetry: float = 0.0, inductance: float = None
                     ) -> np.ndarray:
    """
    DC SQUID 響應模型
    
    Parameters:
    -----------
    phi_ext : np.ndarray
        外部磁通值
    Ic1 : float
        第一個約瑟夫森結的臨界電流
    Ic2 : float
        第二個約瑟夫森結的臨界電流
    asymmetry : float, optional
        結的不對稱性
    inductance : float, optional
        SQUID 迴路電感
        
    Returns:
    --------
    np.ndarray
        SQUID 臨界電流響應
    """
```

---

## 分析模組 (`josephson_analysis.analysis`)

### 週期性分析 (`periodicity`)

#### `fft_period_analysis()`

```python
def fft_period_analysis(phi_ext: np.ndarray, current: np.ndarray,
                       window: str = 'hann', detrend: bool = True,
                       zero_padding: int = 1) -> Dict[str, Any]:
    """
    FFT 週期性分析
    
    Parameters:
    -----------
    phi_ext : np.ndarray
        磁通值
    current : np.ndarray
        電流值
    window : str, optional
        窗函數類型
    detrend : bool, optional
        是否去除趨勢
    zero_padding : int, optional
        零填充倍數
        
    Returns:
    --------
    Dict[str, Any]
        FFT 分析結果
    """
```

#### `lomb_scargle_analysis()`

```python
def lomb_scargle_analysis(phi_ext: np.ndarray, current: np.ndarray,
                         frequency_range: Tuple[float, float] = None,
                         n_frequencies: int = 1000) -> Dict[str, Any]:
    """
    Lomb-Scargle 週期圖分析
    
    適用於非均勻採樣數據的週期檢測
    
    Parameters:
    -----------
    phi_ext : np.ndarray
        磁通值
    current : np.ndarray
        電流值
    frequency_range : Tuple[float, float], optional
        頻率範圍
    n_frequencies : int, optional
        頻率點數
        
    Returns:
    --------
    Dict[str, Any]
        Lomb-Scargle 分析結果
    """
```

#### `autocorrelation_analysis()`

```python
def autocorrelation_analysis(phi_ext: np.ndarray, current: np.ndarray,
                           max_lag_fraction: float = 0.5) -> Dict[str, Any]:
    """
    自相關函數週期分析
    
    Parameters:
    -----------
    phi_ext : np.ndarray
        磁通值
    current : np.ndarray
        電流值
    max_lag_fraction : float, optional
        最大延遲比例
        
    Returns:
    --------
    Dict[str, Any]
        自相關分析結果
    """
```

### 參數擬合 (`fitting`)

#### `estimate_initial_parameters()`

```python
def estimate_initial_parameters(phi_ext: np.ndarray, current: np.ndarray
                              ) -> Dict[str, float]:
    """
    自動估計初始參數
    
    Parameters:
    -----------
    phi_ext : np.ndarray
        磁通值
    current : np.ndarray
        電流值
        
    Returns:
    --------
    Dict[str, float]
        估計的初始參數
    """
```

#### `fit_with_bounds()`

```python
def fit_with_bounds(phi_ext: np.ndarray, current: np.ndarray,
                   model_func: callable, initial_params: Dict[str, float],
                   bounds: Dict[str, Tuple[float, float]] = None,
                   method: str = 'lm') -> Dict[str, Any]:
    """
    帶界限的非線性擬合
    
    Parameters:
    -----------
    phi_ext : np.ndarray
        自變數
    current : np.ndarray
        因變數
    model_func : callable
        模型函數
    initial_params : Dict[str, float]
        初始參數
    bounds : Dict[str, Tuple[float, float]], optional
        參數界限
    method : str, optional
        優化方法
        
    Returns:
    --------
    Dict[str, Any]
        擬合結果
    """
```

### 統計分析 (`statistics`)

#### `calculate_fit_statistics()`

```python
def calculate_fit_statistics(observed: np.ndarray, predicted: np.ndarray,
                           n_parameters: int) -> Dict[str, float]:
    """
    計算擬合統計指標
    
    Parameters:
    -----------
    observed : np.ndarray
        觀測值
    predicted : np.ndarray
        預測值
    n_parameters : int
        模型參數數量
        
    Returns:
    --------
    Dict[str, float]
        統計指標字典
    """
```

#### `bootstrap_confidence_intervals()`

```python
def bootstrap_confidence_intervals(phi_ext: np.ndarray, current: np.ndarray,
                                 fit_function: callable, n_bootstrap: int = 1000,
                                 confidence_level: float = 0.95) -> Dict[str, Any]:
    """
    Bootstrap 置信區間估計
    
    Parameters:
    -----------
    phi_ext : np.ndarray
        自變數
    current : np.ndarray
        因變數
    fit_function : callable
        擬合函數
    n_bootstrap : int, optional
        Bootstrap 樣本數
    confidence_level : float, optional
        置信水準
        
    Returns:
    --------
    Dict[str, Any]
        置信區間結果
    """
```

---

## 視覺化模組 (`josephson_analysis.visualization`)

### 磁響應繪圖 (`magnetic_plots`)

#### `plot_flux_response()`

```python
def plot_flux_response(phi_ext: np.ndarray, current: np.ndarray,
                      fitted_current: np.ndarray = None,
                      title: str = "Josephson Junction Flux Response",
                      show_residuals: bool = False,
                      interactive: bool = True) -> Any:
    """
    繪製磁通響應曲線
    
    Parameters:
    -----------
    phi_ext : np.ndarray
        磁通值
    current : np.ndarray
        電流值
    fitted_current : np.ndarray, optional
        擬合電流值
    title : str, optional
        圖表標題
    show_residuals : bool, optional
        是否顯示殘差
    interactive : bool, optional
        是否使用互動式圖表
        
    Returns:
    --------
    Figure
        圖表物件
    """
```

### 週期分析繪圖 (`period_analysis`)

#### `plot_period_spectrum()`

```python
def plot_period_spectrum(frequency: np.ndarray, power: np.ndarray,
                        method: str = "FFT",
                        peak_frequencies: np.ndarray = None,
                        title: str = None,
                        interactive: bool = True) -> Any:
    """
    繪製功率頻譜
    
    Parameters:
    -----------
    frequency : np.ndarray
        頻率值
    power : np.ndarray
        功率值
    method : str, optional
        分析方法
    peak_frequencies : np.ndarray, optional
        峰值頻率
    title : str, optional
        圖表標題
    interactive : bool, optional
        是否使用互動式圖表
        
    Returns:
    --------
    Figure
        圖表物件
    """
```

---

## 工具模組 (`josephson_analysis.utils`)

### 數據處理 (`data_processing`)

#### `process_magnetic_sweep_data()`

```python
def process_magnetic_sweep_data(phi_ext: np.ndarray, current: np.ndarray,
                               remove_offset: bool = True,
                               smooth_data: bool = False,
                               smooth_window: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    處理磁通掃描數據
    
    Parameters:
    -----------
    phi_ext : np.ndarray
        磁通值
    current : np.ndarray
        電流值
    remove_offset : bool, optional
        是否移除直流偏移
    smooth_data : bool, optional
        是否平滑數據
    smooth_window : int, optional
        平滑窗口大小
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        處理後的數據
    """
```

### 參數估計 (`parameter_estimation`)

#### `estimate_josephson_parameters()`

```python
def estimate_josephson_parameters(phi_ext: np.ndarray, current: np.ndarray
                                ) -> Dict[str, float]:
    """
    估計約瑟夫森結參數
    
    Parameters:
    -----------
    phi_ext : np.ndarray
        磁通值
    current : np.ndarray
        電流值
        
    Returns:
    --------
    Dict[str, float]
        估計參數
    """
```

---

## 常數和單位

### 物理常數

```python
# 磁通量子
FLUX_QUANTUM = 2.067833831e-15  # Wb

# 基本電荷
ELEMENTARY_CHARGE = 1.602176634e-19  # C

# 普朗克常數
PLANCK_CONSTANT = 6.62607015e-34  # J⋅s

# 約化普朗克常數
REDUCED_PLANCK = PLANCK_CONSTANT / (2 * np.pi)  # J⋅s

# 波茲曼常數
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K
```

### 典型參數範圍

```python
# 約瑟夫森結典型參數
TYPICAL_RANGES = {
    'Ic': (1e-9, 1e-3),          # 臨界電流 (A)
    'frequency': (1e10, 1e15),    # 頻率參數 (Hz⁻¹)
    'phase_offset': (-np.pi, np.pi),  # 相位偏移 (rad)
    'temperature': (0.001, 10),   # 溫度 (K)
    'noise_level': (1e-12, 1e-6)  # 雜訊水準 (A)
}
```

---

## 異常處理

### 自定義異常

```python
class JosephsonAnalysisError(Exception):
    """約瑟夫森分析基礎異常"""
    pass

class FittingError(JosephsonAnalysisError):
    """參數擬合異常"""
    pass

class DataProcessingError(JosephsonAnalysisError):
    """數據處理異常"""
    pass

class VisualizationError(JosephsonAnalysisError):
    """視覺化異常"""
    pass
```

### 常見異常情況

1. **數據格式錯誤**
   - `ValueError`: 輸入數據格式不正確
   - `IndexError`: 數據長度不匹配

2. **擬合失敗**
   - `FittingError`: 擬合算法不收斂
   - `RuntimeError`: 數值計算錯誤

3. **參數範圍錯誤**
   - `ValueError`: 參數超出物理合理範圍
   - `OverflowError`: 數值溢出

---

*更多詳細資訊和使用範例請參考 [使用者指南](user_guide.md) 和 [範例程式碼](../examples/)。*
