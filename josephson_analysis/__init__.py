"""
Josephson 結週期性信號分析工具包

這個套件提供了完整的 Josephson 結磁通響應分析功能，包括：
- 週期性信號檢測
- 物理參數估計
- 統計評估
- 視覺化工具

作者: aa8919aa3
版本: 1.0.0
許可證: MIT
"""

__version__ = "1.0.0"
__author__ = "aa8919aa3"
__email__ = ""

from .models.josephson_physics import (
    JosephsonPeriodicAnalyzer,
    full_josephson_model,
    simplified_josephson_model
)
from .analysis.statistics import (
    ModelStatistics,
    compare_multiple_models,
    plot_comprehensive_model_diagnostics
)
from .analysis.periodicity import (
    enhanced_lomb_scargle_analysis,
    fft_period_analysis
)
from .visualization.magnetic_plots import (
    plot_flux_response,
    plot_period_analysis,
    plot_comprehensive_analysis
)

__all__ = [
    'JosephsonPeriodicAnalyzer',
    'full_josephson_model',
    'simplified_josephson_model',
    'ModelStatistics',
    'compare_multiple_models',
    'enhanced_lomb_scargle_analysis',
    'fft_period_analysis',
    'plot_flux_response',
    'plot_period_analysis',
    'plot_comprehensive_analysis',
    'plot_comprehensive_model_diagnostics'
]