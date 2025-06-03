"""
命令行接口

提供簡單的命令行工具來執行 Josephson 結分析。
"""

import argparse
import sys
from .models.josephson_physics import JosephsonPeriodicAnalyzer

def main():
    """主要的 CLI 入口點"""
    parser = argparse.ArgumentParser(
        description='Josephson 結週期性信號分析工具'
    )
    
    parser.add_argument(
        '--generate', 
        action='store_true',
        help='生成模擬數據'
    )
    
    parser.add_argument(
        '--analyze',
        action='store_true', 
        help='執行週期性分析'
    )
    
    parser.add_argument(
        '--model',
        choices=['full', 'simplified', 'both'],
        default='both',
        help='選擇模型類型'
    )
    
    parser.add_argument(
        '--noise',
        type=float,
        default=2e-7,
        help='雜訊水平 (A)'
    )
    
    parser.add_argument(
        '--points',
        type=int,
        default=500,
        help='數據點數'
    )
    
    args = parser.parse_args()
    
    print("🧲 Josephson 結分析工具")
    print("="*40)
    
    analyzer = JosephsonPeriodicAnalyzer()
    
    if args.generate:
        print("生成模擬數據...")
        data = analyzer.generate_flux_sweep_data(
            n_points=args.points,
            model_type=args.model,
            noise_level=args.noise
        )
        print("✅ 數據生成完成")
    
    if args.analyze:
        print("執行週期性分析...")
        results = analyzer.analyze_periodicity(model_type=args.model)
        print("✅ 分析完成")
        
        analyzer.generate_summary_report()

if __name__ == "__main__":
    main()