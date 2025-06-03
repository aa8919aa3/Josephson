"""
基本 Josephson 結週期性分析範例

這個範例展示如何使用工具包進行完整的 Josephson 結分析。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from josephson_analysis import JosephsonPeriodicAnalyzer
from josephson_analysis.analysis.statistics import compare_multiple_models
import numpy as np

def main():
    """執行基本的 Josephson 結分析"""
    
    print("🚀 開始 Josephson 結週期性信號分析")
    print("="*60)
    
    # 創建分析器
    analyzer = JosephsonPeriodicAnalyzer(save_data=True)
    
    # 設置物理參數
    params = {
        'Ic': 1.0e-6,           # 臨界電流 1 μA
        'phi_0': np.pi / 4,     # 相位偏移
        'f': 5e4,               # 週期頻率 50 kHz
        'T': 0.8,               # 非線性參數
        'k': -0.00,             # 二次項係數
        'r': 5e-3,              # 線性項係數
        'C': 10.0e-6,           # 常數項 10 μA
        'd': -10.0e-3,          # 偏移量
        'noise_level': 2e-7     # 雜訊水平 0.2 μA
    }
    
    print(f"📋 分析參數:")
    for key, value in params.items():
        print(f"   {key}: {value:.2e}")
    
    # 1. 生成模擬數據
    print(f"\n🔬 步驟 1: 生成模擬數據")
    data = analyzer.generate_flux_sweep_data(
        phi_range=(-20e-5, 0e-5),
        n_points=1001,
        model_type="both",
        **params
    )
    
    # 2. 週期性分析
    print(f"\n🔍 步驟 2: 週期性分析")
    period_results = analyzer.analyze_periodicity(model_type="both")
    
    # 3. 參數擬合
    print(f"\n🔧 步驟 3: 參數擬合")
    
    # 擬合完整模型
    full_fit = analyzer.fit_model_parameters(model_type='full')
    
    # 擬合簡化模型
    simple_fit = analyzer.fit_model_parameters(model_type='simplified')
    
    # 4. 模型比較
    if full_fit and simple_fit:
        print(f"\n🏆 步驟 4: 模型比較")
        comparison = compare_multiple_models(
            full_fit['statistics'],
            simple_fit['statistics'],
            plot_comparison=False
        )
    
    # 5. 生成完整報告
    print(f"\n📊 步驟 5: 生成分析報告")
    analyzer.generate_summary_report()
    
    # 6. 視覺化（如果需要）
    print(f"\n🎨 步驟 6: 生成視覺化圖表")
    try:
        from josephson_analysis.visualization.magnetic_plots import plot_comprehensive_analysis
        plot_comprehensive_analysis(analyzer)
        print("✅ 視覺化圖表已生成")
    except ImportError:
        print("⚠️ 視覺化模組未完成，跳過圖表生成")
    
    print(f"\n✅ 分析完成！")
    
    return analyzer

if __name__ == "__main__":
    analyzer = main()