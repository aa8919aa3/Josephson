# Josephson 結週期性信號分析 Repository 設置

## 項目概述
建立一個專門用於 Josephson 結週期性信號分析的 GitHub repository，重點是：
- 外部磁通對電流的週期性影響分析
- Lomb-Scargle 週期圖分析（用於檢測磁通-電流關係中的週期性）
- 非線性 Josephson 效應的統計建模
- 物理參數估計和模型驗證

## 正確的物理概念
- **Phi_ext**: 外部磁通/磁場強度
- **I_s**: 超導電流響應
- **分析目標**: 電流隨磁通變化的週期性特徵
- **應用**: 超導量子干涉器件(SQUID)、量子位元等

## Repository 結構
```
josephson-periodic-analysis/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── LICENSE
├── docs/
│   ├── physics_background.md
│   ├── user_guide.md
│   └── api_reference.md
├── josephson_analysis/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── josephson_physics.py
│   │   └── periodic_models.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── periodicity.py
│   │   ├── fitting.py
│   │   └── statistics.py
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── magnetic_plots.py
│   │   └── period_analysis.py
│   └── utils/
│       ├── __init__.py
│       ├── data_processing.py
│       └── parameter_estimation.py
├── examples/
│   ├── squid_analysis.py
│   ├── josephson_modeling.py
│   └── parameter_sweep.py
├── tests/
│   ├── test_josephson_models.py
│   ├── test_periodicity.py
│   └── test_fitting.py
└── data/
    ├── experimental/
    └── simulated/
```

## 主要功能
1. **Josephson 結物理建模**
   - 完整非線性模型
   - 簡化線性近似
   - 磁通量子化效應

2. **週期性分析**
   - 磁通依賴的週期檢測
   - Fourier 分析
   - Lomb-Scargle 方法（適用於不均勻磁場掃描）

3. **參數估計**
   - 臨界電流 Ic 估計
   - 週期性參數提取
   - 雜訊特性分析

4. **實驗數據分析**
   - SQUID 磁力計數據
   - 量子位元控制曲線
   - 磁通調制特性

## 物理應用領域
- 超導量子干涉器件 (SQUID)
- 超導量子位元
- 磁通量子位元 (Flux Qubit)
- 約瑟夫森參量放大器
