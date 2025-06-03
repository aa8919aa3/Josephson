#!/usr/bin/env python3
"""
實時優化系統應用範例：對實驗數據進行自動化參數優化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
from scipy.stats import pearsonr

# 導入我們的實時優化器
import sys
sys.path.append('/Users/albert-mac/Code/GitHub/Josephson')
from josephson_analysis.optimization.real_time_optimizer import RealTimeOptimizer

def load_experimental_data(filename: str) -> tuple:
    """
    載入實驗數據
    
    Args:
        filename: 實驗數據檔案名
        
    Returns:
        (field_data, current_data) 元組
    """
    data_dir = Path("/Users/albert-mac/Code/GitHub/Josephson/data/experimental")
    file_path = data_dir / filename
    
    if not file_path.exists():
        raise FileNotFoundError(f"找不到檔案: {file_path}")
    
    data = pd.read_csv(file_path)
    field_data = data['y_field'].values
    current_data = data['Ic'].values
    
    return field_data, current_data

def run_optimization_comparison(filename: str, methods: list = None):
    """
    對實驗數據執行多種優化方法的比較
    
    Args:
        filename: 實驗數據檔案名
        methods: 要比較的優化方法列表
    """
    if methods is None:
        methods = ['bayesian', 'differential', 'gradient']
    
    print(f"\\n=== 對 {filename} 進行實時優化分析 ===")
    
    # 載入實驗數據
    try:
        field_data, experimental_data = load_experimental_data(filename)
        print(f"成功載入數據：{len(experimental_data)} 個數據點")
    except Exception as e:
        print(f"載入數據失敗：{e}")
        return None
    
    results = {}
    
    # 對每種方法進行優化
    for method in methods:
        print(f"\\n--- 使用 {method.upper()} 方法優化 ---")
        
        # 創建優化器
        optimizer = RealTimeOptimizer(
            target_correlation=0.5,  # 設定較為實際的目標
            max_iterations=30,       # 減少迭代次數以加快演示
            optimization_method=method
        )
        
        # 執行優化
        start_time = time.time()
        try:
            result = optimizer.optimize_parameters(experimental_data, field_data)
            optimization_time = time.time() - start_time
            
            # 驗證結果
            best_params = result['best_parameters']
            simulated_data = optimizer.generate_josephson_response(
                field_data,
                best_params['Ic_base'],
                best_params['field_scale'], 
                best_params['phase_offset'],
                best_params['asymmetry'],
                best_params['noise_level']
            )
            
            # 重新計算相關係數確認
            final_correlation, _ = pearsonr(experimental_data, simulated_data)
            
            results[method] = {
                'optimization_result': result,
                'final_correlation': final_correlation,
                'optimization_time': optimization_time,
                'simulated_data': simulated_data,
                'success': True
            }
            
            print(f"✅ {method.upper()} 優化完成")
            print(f"   最佳相關係數: {final_correlation:.6f}")
            print(f"   優化時間: {optimization_time:.2f} 秒")
            print(f"   迭代次數: {result['total_iterations']}")
            
        except Exception as e:
            print(f"❌ {method.upper()} 優化失敗：{e}")
            results[method] = {'success': False, 'error': str(e)}
    
    # 生成比較報告
    generate_optimization_report(filename, results, field_data, experimental_data)
    
    return results

def generate_optimization_report(filename: str, results: dict, 
                               field_data: np.ndarray, experimental_data: np.ndarray):
    """
    生成優化結果報告和視覺化
    """
    results_dir = Path("/Users/albert-mac/Code/GitHub/Josephson/results")
    results_dir.mkdir(exist_ok=True)
    
    # 創建比較圖表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'實時優化結果比較：{filename}', fontsize=16, fontweight='bold')
    
    successful_methods = [method for method, result in results.items() if result.get('success', False)]
    
    if not successful_methods:
        print("⚠️  沒有成功的優化結果，跳過報告生成")
        return
    
    # 1. 原始數據與優化結果比較
    ax1 = axes[0, 0]
    ax1.plot(field_data, experimental_data, 'k.-', label='實驗數據', alpha=0.8, markersize=4)
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, method in enumerate(successful_methods):
        result = results[method]
        simulated_data = result['simulated_data']
        correlation = result['final_correlation']
        
        ax1.plot(field_data, simulated_data, color=colors[i % len(colors)], 
                alpha=0.7, label=f'{method.upper()} (r={correlation:.4f})')
    
    ax1.set_xlabel('y_field')
    ax1.set_ylabel('Ic (A)')
    ax1.set_title('數據擬合比較')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 相關係數比較
    ax2 = axes[0, 1]
    method_names = list(successful_methods)
    correlations = [results[method]['final_correlation'] for method in method_names]
    
    bars = ax2.bar(method_names, correlations, alpha=0.7, color=colors[:len(method_names)])
    ax2.set_ylabel('相關係數')
    ax2.set_title('方法性能比較')
    ax2.grid(True, alpha=0.3)
    
    # 標記最佳方法
    if correlations:
        best_idx = np.argmax(correlations)
        bars[best_idx].set_color('gold')
        ax2.text(best_idx, correlations[best_idx] + 0.01, 'BEST', 
                ha='center', va='bottom', fontweight='bold')
    
    # 3. 優化時間比較
    ax3 = axes[1, 0]
    times = [results[method]['optimization_time'] for method in method_names]
    
    ax3.bar(method_names, times, alpha=0.7, color=colors[:len(method_names)])
    ax3.set_ylabel('優化時間 (秒)')
    ax3.set_title('計算效率比較')
    ax3.grid(True, alpha=0.3)
    
    # 4. 統計摘要
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # 準備統計文字
    stats_text = f"""優化統計摘要：

檔案：{filename}
數據點數：{len(experimental_data)}
實驗數據統計：
  平均值：{np.mean(experimental_data):.2e} A
  標準差：{np.std(experimental_data):.2e} A
  範圍：[{np.min(experimental_data):.2e}, {np.max(experimental_data):.2e}]

優化結果："""
    
    for method in method_names:
        result = results[method]
        stats_text += f"""
{method.upper()}：
  相關係數：{result['final_correlation']:.6f}
  優化時間：{result['optimization_time']:.2f} 秒
  迭代次數：{result['optimization_result']['total_iterations']}"""
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # 保存圖表
    plot_file = results_dir / f"realtime_optimization_{filename.replace('.csv', '.png')}"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"\\n📊 比較圖表已保存：{plot_file}")
    
    plt.show()
    
    # 保存詳細結果
    detailed_results = {}
    for method, result in results.items():
        if result.get('success', False):
            detailed_results[method] = {
                'final_correlation': float(result['final_correlation']),
                'optimization_time': float(result['optimization_time']),
                'best_parameters': result['optimization_result']['best_parameters'],
                'total_iterations': int(result['optimization_result']['total_iterations']),
                'convergence_achieved': bool(result['optimization_result']['convergence_achieved'])
            }
    
    json_file = results_dir / f"realtime_optimization_{filename.replace('.csv', '.json')}"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    print(f"📄 詳細結果已保存：{json_file}")

def main():
    """
    主要執行函數
    """
    print("🚀 實時參數優化系統演示")
    print("=" * 50)
    
    # 選擇要分析的檔案（基於之前的分析結果）
    test_files = ['317Ic.csv', '337Ic.csv', '335Ic.csv']  # 選擇幾個代表性檔案
    
    # 比較不同優化方法
    optimization_methods = ['bayesian', 'differential']  # 暫時使用兩種方法以節省時間
    
    all_results = {}
    
    for filename in test_files:
        print(f"\\n{'='*60}")
        print(f"正在處理：{filename}")
        print(f"{'='*60}")
        
        try:
            results = run_optimization_comparison(filename, optimization_methods)
            all_results[filename] = results
        except Exception as e:
            print(f"❌ 處理 {filename} 時發生錯誤：{e}")
            continue
    
    # 生成總體摘要
    print("\\n" + "="*60)
    print("📋 總體優化結果摘要")
    print("="*60)
    
    for filename, file_results in all_results.items():
        if file_results:
            print(f"\\n📁 {filename}:")
            successful_methods = [method for method, result in file_results.items() 
                                if result.get('success', False)]
            
            if successful_methods:
                best_method = max(successful_methods, 
                                key=lambda m: file_results[m]['final_correlation'])
                best_corr = file_results[best_method]['final_correlation']
                print(f"   🏆 最佳方法：{best_method.upper()} (r = {best_corr:.6f})")
                
                for method in successful_methods:
                    result = file_results[method]
                    print(f"   • {method.upper()}: r = {result['final_correlation']:.6f}, "
                          f"時間 = {result['optimization_time']:.2f}s")
            else:
                print("   ❌ 所有優化方法都失敗了")

if __name__ == "__main__":
    main()
