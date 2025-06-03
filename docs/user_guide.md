# 使用者指南

## 快速開始

### 安裝

1. **克隆專案**
```bash
git clone https://github.com/aa8919aa3/josephson-periodic-analysis.git
cd josephson-periodic-analysis
```

2. **安裝依賴**
```bash
pip install -r requirements.txt
```

3. **安裝套件**
```bash
pip install -e .
```

### 基本使用範例

#### 1. 載入套件

```python
import numpy as np
from josephson_analysis import JosephsonPeriodicAnalyzer
from josephson_analysis.models import JosephsonPhysics
from josephson_analysis.visualization import plot_flux_response
```

#### 2. 創建分析器

```python
# 創建分析器實例
analyzer = JosephsonPeriodicAnalyzer()

# 設定分析參數
analyzer.set_analysis_parameters(
    flux_quantum=2.07e-15,  # 磁通量子 (Wb)
    temperature=0.01,       # 溫度 (K)
    noise_level=1e-8       # 雜訊水準
)
```

#### 3. 生成測試數據

```python
# 生成模擬的磁通掃描數據
phi_ext, current = analyzer.generate_flux_sweep_data(
    Ic=1.0e-6,              # 臨界電流 1 μA
    phi_range=(-2e-14, 2e-14),  # 磁通範圍
    n_points=500,           # 數據點數
    noise_level=2e-8        # 雜訊水準
)
```

#### 4. 週期性分析

```python
# 執行完整的週期性分析
results = analyzer.analyze_periodicity(phi_ext, current)

print(f"檢測到的主要週期: {results['primary_period']:.3e}")
print(f"信噪比: {results['signal_to_noise_ratio']:.2f}")
```

## 主要功能模組

### 1. 物理模型 (`josephson_analysis.models`)

#### 基本約瑟夫森模型

```python
from josephson_analysis.models import JosephsonPhysics

physics = JosephsonPhysics()

# 簡化正弦模型
current = physics.simplified_josephson_model(
    phi_ext=phi_values,
    Ic=1e-6,        # 臨界電流
    phi_0=0,        # 相位偏移
    freq=1e13,      # 頻率參數
    k=0,            # 二次項係數
    r=0,            # 線性項係數
    C=0,            # 常數項
    d=0             # 磁通偏移
)

# 完整非線性模型
current_full = physics.full_josephson_model(
    phi_ext=phi_values,
    Ic=1e-6,
    phi_0=0,
    freq=1e13,
    T=0.1,          # 非線性參數
    k=0,
    r=0,
    C=0,
    d=0
)
```

#### SQUID 模型

```python
# DC SQUID 響應
squid_current = physics.dc_squid_response(
    phi_ext=phi_values,
    Ic1=1e-6,       # 第一個結的臨界電流
    Ic2=1e-6,       # 第二個結的臨界電流
    asymmetry=0.05  # 不對稱性參數
)
```

### 2. 週期性分析 (`josephson_analysis.analysis`)

#### FFT 分析

```python
from josephson_analysis.analysis import periodicity

# 執行 FFT 分析
fft_results = periodicity.fft_period_analysis(
    phi_ext, current,
    window='hann',      # 窗函數
    detrend=True        # 是否去趨勢
)

print(f"主要頻率: {fft_results['primary_frequency']:.3e} Hz⁻¹")
print(f"對應週期: {fft_results['primary_period']:.3e}")
```

#### Lomb-Scargle 分析

```python
# 適用於非均勻採樣數據
ls_results = periodicity.lomb_scargle_analysis(
    phi_ext, current,
    frequency_range=(1e10, 1e15),  # 頻率範圍
    n_frequencies=1000             # 頻率點數
)

# 檢查統計顯著性
significance = periodicity.calculate_false_alarm_probability(
    ls_results['power'],
    n_independent=len(current)
)
```

#### 自相關分析

```python
# 自相關函數分析
autocorr_results = periodicity.autocorrelation_analysis(
    phi_ext, current,
    max_lag_fraction=0.5    # 最大延遲比例
)

print(f"自相關檢測的週期: {autocorr_results['period_estimate']:.3e}")
```

### 3. 參數擬合 (`josephson_analysis.analysis.fitting`)

#### 自動參數估計

```python
from josephson_analysis.analysis import fitting

# 自動估計初始參數
initial_params = fitting.estimate_initial_parameters(phi_ext, current)
print("初始參數估計:", initial_params)
```

#### 非線性擬合

```python
# 執行非線性最小二乘法擬合
fit_results = fitting.fit_josephson_model(
    phi_ext, current,
    model_type='simplified',    # 'simplified' 或 'full'
    initial_guess=initial_params,
    bounds=None,                # 參數界限
    method='lm'                 # 優化方法
)

print("擬合參數:")
for param, value in fit_results['parameters'].items():
    print(f"  {param}: {value:.3e} ± {fit_results['parameter_errors'][param]:.3e}")

print(f"R²: {fit_results['r_squared']:.4f}")
print(f"RMSE: {fit_results['rmse']:.3e}")
```

#### 參數不確定性分析

```python
# Bootstrap 不確定性估計
uncertainty_results = fitting.bootstrap_parameter_uncertainty(
    phi_ext, current,
    n_bootstrap=1000,
    confidence_level=0.95
)

print("參數不確定性 (95% 信賴區間):")
for param, interval in uncertainty_results['confidence_intervals'].items():
    print(f"  {param}: [{interval[0]:.3e}, {interval[1]:.3e}]")
```

### 4. 統計評估 (`josephson_analysis.analysis.statistics`)

#### 擬合品質評估

```python
from josephson_analysis.analysis import statistics

# 計算詳細的統計指標
stats = statistics.calculate_fit_statistics(
    observed=current,
    predicted=fitted_current,
    n_parameters=len(fit_results['parameters'])
)

print(f"R²: {stats['r_squared']:.4f}")
print(f"Adjusted R²: {stats['adjusted_r_squared']:.4f}")
print(f"AIC: {stats['aic']:.2f}")
print(f"BIC: {stats['bic']:.2f}")
print(f"RMSE: {stats['rmse']:.3e}")
```

#### 殘差分析

```python
# 殘差分析
residual_stats = statistics.analyze_residuals(
    observed=current,
    predicted=fitted_current
)

print(f"殘差正態性檢驗 p-value: {residual_stats['normality_test_p']:.4f}")
print(f"殘差自相關: {residual_stats['autocorrelation']:.4f}")
print(f"異方差性檢驗 p-value: {residual_stats['heteroscedasticity_p']:.4f}")
```

### 5. 視覺化 (`josephson_analysis.visualization`)

#### 磁通響應圖

```python
from josephson_analysis.visualization import magnetic_plots

# 繪製磁通響應曲線
fig = magnetic_plots.plot_flux_response(
    phi_ext, current,
    fitted_current=fitted_current,
    show_residuals=True,
    interactive=True
)
fig.show()
```

#### 週期分析圖

```python
from josephson_analysis.visualization import period_analysis

# 繪製功率譜
fig = period_analysis.plot_period_spectrum(
    frequency=fft_results['frequency'],
    power=fft_results['power'],
    method="FFT",
    peak_frequencies=fft_results['peak_frequencies']
)
fig.show()

# 繪製 Lomb-Scargle 週期圖
fig_ls = period_analysis.plot_lomb_scargle_periodogram(
    frequency=ls_results['frequency'],
    power=ls_results['power'],
    false_alarm_levels={'1%': 0.01, '5%': 0.05}
)
fig_ls.show()
```

#### 參數擬合結果

```python
# 繪製擬合結果
fig = period_analysis.plot_parameter_fitting_results(
    phi_ext, current, fitted_current,
    fit_params=fit_results['parameters']
)
fig.show()
```

## 實際應用案例

### 案例 1: SQUID 磁力計數據分析

```python
import pandas as pd

# 載入實驗數據
data = pd.read_csv('squid_measurement.csv')
phi_ext = data['magnetic_flux'].values
voltage = data['voltage_response'].values

# 分析 SQUID 響應
squid_analyzer = JosephsonPeriodicAnalyzer()
squid_results = squid_analyzer.analyze_squid_response(phi_ext, voltage)

# 計算磁場靈敏度
sensitivity = squid_analyzer.calculate_field_sensitivity(squid_results)
print(f"磁場靈敏度: {sensitivity:.2e} T/√Hz")
```

### 案例 2: 量子位元特性分析

```python
# 量子位元頻譜分析
def analyze_qubit_spectroscopy(flux_bias, frequency_data):
    analyzer = JosephsonPeriodicAnalyzer()
    
    # 找到最佳工作點 (sweet spots)
    sweet_spots = analyzer.find_sweet_spots(flux_bias, frequency_data)
    
    # 分析相干時間
    coherence_analysis = analyzer.analyze_coherence_properties(
        flux_bias, frequency_data
    )
    
    return sweet_spots, coherence_analysis

# 使用範例
sweet_spots, coherence = analyze_qubit_spectroscopy(bias_flux, qubit_freq)
```

### 案例 3: 參數掃描優化

```python
# 參數空間搜索
param_sweep_results = analyzer.parameter_sweep_analysis(
    phi_ext, current,
    param_ranges={
        'Ic': (1e-7, 1e-5),      # 臨界電流範圍
        'freq': (1e12, 1e14),    # 頻率範圍
    },
    n_points=50,                 # 每個參數的點數
    metric='r_squared'           # 優化指標
)

# 找到最佳參數組合
optimal_params = param_sweep_results['optimal_parameters']
print("最佳參數組合:", optimal_params)
```

## 高級功能

### 多模型比較

```python
# 比較不同物理模型的表現
models_to_compare = ['simplified', 'full', 'polynomial']
comparison_results = analyzer.compare_models(
    phi_ext, current,
    models=models_to_compare,
    cross_validation=True
)

# 選擇最佳模型
best_model = comparison_results['best_model']
print(f"最佳模型: {best_model}")
```

### 雜訊特性分析

```python
# 分析測量雜訊
noise_analysis = analyzer.analyze_noise_characteristics(current)
print(f"雜訊功率譜密度: {noise_analysis['psd_mean']:.3e} A²/Hz")
print(f"信噪比: {noise_analysis['snr']:.2f} dB")
```

### 時間序列分析

```python
# 時間相關的磁通調制分析
time_series_results = analyzer.analyze_time_dependent_modulation(
    time_array, phi_ext_time, current_time
)
```

## 故障排除

### 常見問題

1. **擬合不收斂**
   - 檢查初始參數估計
   - 調整參數界限
   - 嘗試不同的優化方法

2. **週期檢測失敗**
   - 增加數據點數
   - 檢查雜訊水準
   - 使用不同的週期分析方法

3. **記憶體不足**
   - 減少數據點數
   - 使用分批處理
   - 優化計算參數

### 效能優化

```python
# 設定並行計算
analyzer.set_computation_parameters(
    n_cores=4,              # 使用的核心數
    chunk_size=1000,        # 分批大小
    use_gpu=False           # 是否使用 GPU
)
```

## 參考文獻和資源

- [物理背景文檔](physics_background.md)
- [API 參考](api_reference.md)
- [範例程式碼](../examples/)
- [測試案例](../tests/)

---

*如有問題或建議，請在 GitHub repository 中提交 issue。*
