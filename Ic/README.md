# 超導電流與磁通量關係模擬項目

## 🔥 最新更新: Current-Phase Relation Heatmap

新增完整的電流-相位關係熱力圖功能，包含：
- 四模型對比熱力圖 (`josephson_current_phase_heatmap.png`)
- 3D進階分析圖 (`josephson_advanced_phase_analysis.png`) 
- 高解析度單模型分析和互動式切片分析

運行 `python demo_heatmap.py` 查看演示效果。

---

## 項目概述

本項目實現了一個完整的 SciPy 模擬程序，用於分析 Josephson 結中超導電流與外加磁通量的關係，基於多種理論模型並考慮透明度效應。

## 理論基礎

### 包含的理論模型

1. **Ambegaokar-Baratoff (AB) 理論**
   - 適用於低透明度隧道結 (T << 1)
   - 基於隧道效應的經典理論
   - 公式：I_c = (π/2) × (Δ/eR_N) × T

2. **Kulik-Omelyanchuk (KO) 理論**
   - 適用於高透明度短結 (T ≈ 1)
   - 考慮安德列夫反射效應
   - 需要複雜的橢圓積分計算

3. **內插公式**
   - 適用於任意透明度範圍
   - 公式：eI_c(0)R_N/Δ = π/(1+√(1-τ))
   - 在實際應用中具有最佳平衡性

4. **散射理論模型**
   - 考慮量子散射效應
   - 包含相移和勢壘強度參數
   - 在中等透明度下表現獨特

### Fraunhofer 繞射圖案

所有模型都遵循 Fraunhofer 繞射公式：
```
I_c(Φ_ext) = I_c(0) × |sin(πΦ_ext/Φ₀)| / |πΦ_ext/Φ₀|
```

### Current-Phase Relation (電流-相位關係)

新增的電流-相位關係分析功能：

1. **理論關係**
   - 基本正弦關係：I = I_c sin(φ)
   - 不同模型和透明度下的修正
   - 相位範圍：0 到 2π

2. **諧波分析**
   - 傅立葉分解分析各次諧波成分
   - 計算總諧波失真 (THD)
   - 分析基波和高次諧波的相對振幅

3. **物理意義**
   - 揭示不同透明度下的電流相位非線性特性
   - 分析各理論模型的諧波特徵差異
   - 提供量子相干性的重要指標

## 新增功能: Current-Phase Relation Heatmap 🔥

### 熱力圖視覺化功能

項目現已包含完整的 **Current-Phase Relation Heatmap** 功能，提供多維度的電流-相位關係分析：

#### 🎯 主要特徵
- **四模型對比熱力圖**: 展示 AB、KO、內插、散射四個理論模型
- **3D 視角分析**: 相位-透明度-電流的三維關係展示
- **高解析度單模型圖**: 詳細的內插模型分析
- **互動式切片分析**: 多面板對比不同參數下的特性

#### 📊 生成的圖表文件
- `josephson_current_phase_heatmap.png` - 主要的四模型熱力圖
- `josephson_advanced_phase_analysis.png` - 進階3D分析圖
- `demo_current_phase_heatmap.png` - 演示用熱力圖
- `demo_high_res_interpolation_heatmap.png` - 高解析度單模型圖
- `demo_interactive_analysis.png` - 互動式分析圖

#### 🔬 物理洞察
- 電流-相位關係的週期性特徵清晰可見
- 不同透明度下的電流調制效應
- 理論模型間的差異在高透明度區域尤為明顯
- 最佳透明度工作點的視覺化識別

## 項目結構

```
Ic/
├── Is_Sim.py                           # 主模擬程序
├── analyze_results.py                  # 結果分析工具
├── 超導電流與磁場模擬.md               # 理論文檔1 (519行)
├── Josephson結模型T與I_c.md           # 理論文檔2 (420行)
├── josephson_flux_dependence.png       # 磁通量依賴性圖
├── josephson_transparency_dependence.png # 透明度依賴性圖
├── josephson_3d_analysis.png           # 3D分析圖
├── detailed_model_comparison.png       # 詳細模型比較圖
├── josephson_simulation_results.csv    # 完整數據 (20,000點)
├── josephson_complete_results.json     # JSON格式結果
├── simulation_analysis_report.md       # 分析報告
└── README.md                           # 本文件
```

## 主要功能

### 1. JosephsonCurrentSimulator 類
- 實現4種理論模型的臨界電流計算
- Fraunhofer 圖案生成
- 溫度效應模擬
- 完整的參數化模擬

### 2. 可視化功能
- 2D 圖表：磁通量依賴性、透明度依賴性
- 3D 分析：透明度 vs 磁通量 vs 電流
- 詳細模型比較圖
- 高質量 PNG 輸出

### 3. 數據分析
- CSV 數據導出 (20,000 數據點)
- JSON 完整結果保存
- Fraunhofer 圖案特徵分析
- 模型性能比較

## 模擬結果摘要

### 關鍵發現

1. **透明度效應**：
   - 內插模型：最大電流 2.8559，最佳透明度 0.990
   - KO 模型：最大電流 3.1415，適用於高透明度
   - AB 模型：適用於低透明度，電流值較低
   - 散射模型：在中等透明度下有獨特優勢

2. **Fraunhofer 圖案**：
   - 所有模型展現典型 sinc 函數包絡線
   - 零點位置：±1Φ₀, ±2Φ₀, ±3Φ₀...
   - 次峰位置符合理論預期
   - 圖案對稱性良好

3. **模型比較**：
   - 內插模型在全範圍內表現均衡，推薦實用
   - KO 模型在高透明度下最準確
   - AB 模型適用於傳統隧道結
   - 散射模型考慮量子效應

## 使用方法

### 運行主模擬
```bash
cd /Users/albert-mac/Code/GitHub/Josephson/Ic
python Is_Sim.py
```

### 分析結果
```bash
python analyze_results.py
```

### 查看結果
生成的圖片和數據文件可直接查看，分析報告在 `simulation_analysis_report.md`。

## 技術特點

- **科學計算**：基於 NumPy, SciPy
- **可視化**：Matplotlib, Seaborn
- **數據處理**：Pandas
- **非交互式**：適合服務器環境運行
- **中文支持**：完整的中文注釋和輸出

## 物理參數

- **超導體**：鋁 (Al)
- **能隙**：Δ₀ = 1.76 meV
- **臨界溫度**：T_c = 1.2 K
- **模擬溫度**：T = 0.1 K
- **透明度範圍**：0.1 - 0.9
- **磁通量範圍**：-3Φ₀ 至 +3Φ₀

## 實驗驗證建議

1. 在透明度 T = 0.3-0.8 範圍內進行實驗驗證
2. 重點測量零磁場臨界電流精度
3. 驗證 Fraunhofer 圖案對稱性和週期性
4. 比較不同溫度下的實驗結果

## 應用前景

- 超導量子計算設備設計
- Josephson 結參數優化
- SQUID 磁強計性能分析
- 超導電子學器件建模

## 作者

AI Assistant  
日期：2025年6月4日

## 版本

v1.0 - 初始完整版本

## 版本更新記錄

### v1.1 - Current-Phase Relation 功能新增

**新增功能：**
1. **電流-相位關係分析 (Current-Phase Relation)**
   - 實現了完整的電流相位關係計算
   - 支持所有4種理論模型 (AB, KO, 內插, 散射)
   - 相位範圍: 0 到 2π，分析不同透明度下的關係

2. **諧波分析 (Harmonic Analysis)**
   - 傅立葉分解分析各次諧波成分 (最多5次諧波)
   - 計算總諧波失真 (THD)
   - 分析基波和高次諧波的相對振幅

3. **新增可視化**
   - `josephson_current_phase_relation.png` - 電流相位關係圖
   - `josephson_harmonic_analysis.png` - 諧波分析圖

4. **數據完整性提升**
   - JSON結果檔案現包含 `current_phase_relation` 和 `harmonic_analysis` 數據
   - 結果摘要包含電流相位分析統計

**物理意義：**
- 揭示不同透明度下的電流相位非線性特性
- 分析各理論模型的諧波特徵差異
- 提供量子相干性和非線性效應的重要指標
- 為SQUID和約瑟夫結量子器件設計提供關鍵參數

**使用示例：**
```python
# 運行完整模擬（包含Current-Phase Relation）
python Is_Sim.py

# 生成的新文件
# - josephson_current_phase_relation.png
# - josephson_harmonic_analysis.png
# - 更新的 josephson_complete_results.json
```

**技術實現：**
- 使用離散傅立葉變換進行諧波分析
- 自動選擇代表性透明度值進行可視化
- 計算總諧波失真(THD)作為非線性度量
- 支持所有現有理論模型的相位關係分析

---

*此項目為超導電流與磁通量關係的理論模擬研究，基於多種 Josephson 結理論模型實現。*
