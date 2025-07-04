# Josephson 結週期性信號分析工具

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Physics](https://img.shields.io/badge/Physics-Superconductor-purple.svg)]()

## 📋 項目簡介

這是一個專門用於分析 Josephson 結在外部磁通調制下的週期性電流響應的 Python 工具包。主要應用於超導量子器件的特性分析和參數提取。

### 🔬 物理背景

Josephson 結的超導電流 `I_s` 隨外部磁通 `Φ_ext` 呈現週期性變化：

```
I_s(Φ_ext) = Ic * f(2πΦ_ext/Φ₀) + background_terms
```

其中 `Φ₀ = h/2e` 是磁通量子，這種週期性是超導量子干涉的直接體現。

### 🌟 主要特點

- **🧲 磁通響應分析**: 分析電流隨外部磁通的週期性變化
- **📊 週期檢測**: 使用 Lomb-Scargle 和 FFT 方法檢測週期性
- **⚗️ 物理建模**: 完整和簡化的 Josephson 結模型
- **📈 參數估計**: 自動提取臨界電流、週期等物理參數
- **🎨 專業視覺化**: 磁通-電流特性曲線和週期分析圖
- **🔍 統計評估**: 完整的擬合品質評估

## 🚀 快速開始

### 安裝

```bash
git clone https://github.com/aa8919aa3/josephson-periodic-analysis.git
cd josephson-periodic-analysis
pip install -r requirements.txt
```

### 基本使用

```python
from josephson_analysis import JosephsonPeriodicAnalyzer

# 創建分析器
analyzer = JosephsonPeriodicAnalyzer()

# 生成模擬的磁通掃描數據
phi_ext, current = analyzer.generate_flux_sweep_data(
    Ic=1.0e-6,          # 臨界電流 1 μA
    phi_range=(-20e-5, 0),  # 磁通範圍
    n_points=500,
    noise_level=2e-7
)

# 分析週期性
period_results = analyzer.analyze_periodicity(phi_ext, current)

# 繪製結果
analyzer.plot_flux_response(phi_ext, current)
analyzer.plot_period_analysis(period_results)
```

## 🔬 核心功能

### 1. Josephson 結物理模型

#### 完整非線性模型
```python
def full_josephson_model(phi_ext, Ic, phi_0, freq, T, k, r, C, d):
    """
    完整的 Josephson 結模型，包含非線性效應
    
    I_s = (Ic * sin(2πf(Φ-d) - φ₀)) / √(1 - T*sin²(...)) + k(Φ-d)² + r(Φ-d) + C
    """
```

#### 簡化線性模型
```python
def simplified_josephson_model(phi_ext, Ic, phi_0, freq, k, r, C, d):
    """
    簡化的正弦模型
    
    I_s = Ic * sin(2πf(Φ-d) - φ₀) + k(Φ-d)² + r(Φ-d) + C
    """
```

### 2. 週期性分析

```python
# FFT 週期分析
fft_results = analyzer.fft_period_analysis(phi_ext, current)

# Lomb-Scargle 週期圖（適用於不均勻採樣）
ls_results = analyzer.lomb_scargle_analysis(phi_ext, current)

# 自相關函數分析
autocorr_results = analyzer.autocorrelation_analysis(phi_ext, current)
```

### 3. 參數估計和擬合

```python
# 自動參數估計
params = analyzer.estimate_parameters(phi_ext, current)

# 非線性擬合
fit_results = analyzer.fit_josephson_model(
    phi_ext, current,
    model_type='full',  # 'full' 或 'simplified'
    initial_guess=params
)

# 擬合品質評估
stats = analyzer.evaluate_fit_quality(fit_results)
```

## 📊 視覺化功能

### 磁通響應曲線
```python
analyzer.plot_flux_response(
    phi_ext, current,
    show_fit=True,
    show_residuals=True
)
```

### 週期分析圖
```python
analyzer.plot_period_spectrum(
    frequency, power,
    highlight_peaks=True
)
```

### 參數掃描圖
```python
analyzer.plot_parameter_sweep(
    param_name='Ic',
    param_values=np.logspace(-7, -5, 20),
    metric='r_squared'
)
```

## 🧪 應用案例

### 1. SQUID 磁力計分析

```python
# SQUID 響應分析
squid_analyzer = SQUIDAnalyzer()

# 載入實驗數據
phi_ext, voltage = squid_analyzer.load_data('squid_data.csv')

# 分析磁通週期性
flux_quantum = squid_analyzer.extract_flux_quantum(phi_ext, voltage)
sensitivity = squid_analyzer.calculate_sensitivity()
```

### 2. 超導量子位元特性

```python
# 量子位元分析
qubit_analyzer = QubitAnalyzer()

# 分析控制曲線
control_flux, energy = qubit_analyzer.load_spectroscopy_data()
sweet_spots = qubit_analyzer.find_sweet_spots(control_flux, energy)
```

### 3. 磁通量子位元

```python
# 磁通量子位元分析
flux_qubit = FluxQubitAnalyzer()

# 分析磁通依賴性
flux_bias, frequency = flux_qubit.measure_frequency_vs_flux()
persistent_current = flux_qubit.extract_persistent_current()
```

## 📈 統計評估

本工具提供完整的統計指標：

- **R² (決定係數)**: 模型解釋變異的比例
- **Adjusted R²**: 考慮參數數量的修正 R²
- **RMSE**: 均方根誤差
- **SSE**: 誤差平方和
- **AIC/BIC**: 模型選擇準則
- **殘差分析**: 檢測系統性偏差

## 🔧 高級功能

### 多模型比較
```python
# 比較不同物理模型
models = ['linear_sine', 'nonlinear_full', 'polynomial']
comparison = analyzer.compare_models(phi_ext, current, models)
```

### 雜訊分析
```python
# 分析測量雜訊特性
noise_stats = analyzer.analyze_noise_characteristics(current)
snr = analyzer.calculate_signal_to_noise_ratio()
```

### 參數不確定性
```python
# Bootstrap 參數不確定性估計
param_uncertainty = analyzer.bootstrap_parameter_uncertainty(
    phi_ext, current, n_bootstrap=1000
)
```

## 📚 物理應用

### 超導量子干涉器件 (SQUID)
- 磁通量子檢測
- 磁場靈敏度分析
- 線性度評估

### 超導量子位元
- 控制參數優化
- 相干時間分析
- 甜點 (Sweet Spot) 識別

### 約瑟夫森參量放大器
- 增益特性分析
- 頻寬優化
- 雜訊性能評估

## 📁 項目結構

```
josephson-periodic-analysis/
├── josephson_analysis/
│   ├── models/                 # 物理模型
│   │   ├── josephson_physics.py
│   │   └── periodic_models.py
│   ├── analysis/               # 分析工具
│   │   ├── periodicity.py
│   │   ├── fitting.py
│   │   └── statistics.py
│   ├── visualization/          # 視覺化
│   │   ├── magnetic_plots.py
│   │   └── period_analysis.py
│   └── applications/           # 應用模組
│       ├── squid.py
│       ├── qubit.py
│       └── amplifier.py
├── examples/                   # 使用範例
├── tests/                      # 測試套件
└── docs/                       # 文檔
```

## 🛠️ 依賴項

```
numpy >= 1.19.0       # 數值計算
scipy >= 1.7.0        # 科學計算
pandas >= 1.3.0       # 數據處理
plotly >= 5.0.0       # 互動視覺化
matplotlib >= 3.3.0   # 靜態圖表
astropy >= 4.0.0      # Lomb-Scargle 分析
lmfit >= 1.0.0        # 非線性擬合
```

## 📖 相關文獻

1. **Josephson, B. D.** (1962). "Possible new effects in superconductive tunnelling"
2. **Clarke, J. & Braginski, A. I.** (2004). "The SQUID Handbook"
3. **Tinkham, M.** (1996). "Introduction to Superconductivity"
4. **Devoret, M. H. & Schoelkopf, R. J.** (2013). "Superconducting circuits for quantum information"

## 🤝 貢獻

歡迎對超導物理和量子器件感興趣的研究者貢獻代碼！

## 📄 許可證

MIT License - 詳見 [LICENSE](LICENSE) 文件

---

🧲 **專注於超導量子器件的週期性響應分析** 🧲