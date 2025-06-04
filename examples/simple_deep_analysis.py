#!/usr/bin/env python3
"""
簡化深度分析工具：對表現最佳的實驗與模擬數據進行對比
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
import json

try:
    import matplotlib.pyplot as plt
    plt.style.use('default')
    plt.rcParams['font.size'] = 10
    plt.rcParams['figure.dpi'] = 100
except ImportError:
    print("警告：matplotlib 不可用，將跳過視覺化功能")
    plt = None

plt.style.use('default')
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 100

def analyze_best_file():
    """分析表現最佳的檔案"""
    print("=== 約瑟夫森結深度分析工具 ===\n")
    
    # 設置路徑
    exp_data_dir = Path("/Users/albert-mac/Code/GitHub/Josephson/data/experimental")
    sim_data_dir = Path("/Users/albert-mac/Code/GitHub/Josephson/data/simulated")
    results_dir = Path("/Users/albert-mac/Code/GitHub/Josephson/results")
    results_dir.mkdir(exist_ok=True)
    
    # 根據之前的結果，317Ic.csv 表現最佳
    filename = "317Ic.csv"
    
    # 載入數據
    exp_file = exp_data_dir / filename
    sim_file = sim_data_dir / f"improved_sim_{filename}"
    
    if not exp_file.exists() or not sim_file.exists():
        print(f"錯誤：找不到檔案 {exp_file} 或 {sim_file}")
        return
    
    exp_data = pd.read_csv(exp_file)
    sim_data = pd.read_csv(sim_file)
    
    exp_y = exp_data['y_field'].values
    exp_Ic = exp_data['Ic'].values
    sim_y = sim_data['y_field'].values
    sim_Ic = sim_data['Ic'].values
    
    # 數據對齊
    min_len = min(len(exp_y), len(sim_y))
    exp_y, exp_Ic = exp_y[:min_len], exp_Ic[:min_len]
    sim_y, sim_Ic = sim_y[:min_len], sim_Ic[:min_len]
    
    # 計算統計量
    correlation, p_value = pearsonr(exp_Ic, sim_Ic)
    
    if plt is None:
        print(f"分析結果：相關係數 = {correlation:.4f}, p-值 = {p_value:.4f}")
        return
    
    # 創建分析圖表
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'深度分析：{filename} (相關係數: {correlation:.4f})', fontsize=16, fontweight='bold')
    
    # 1. 原始數據比較
    axes[0, 0].plot(exp_y, exp_Ic, 'b.-', label='實驗數據', alpha=0.7, markersize=3)
    axes[0, 0].plot(sim_y, sim_Ic, 'r.-', label='模擬數據', alpha=0.7, markersize=3)
    axes[0, 0].set_xlabel('y_field')
    axes[0, 0].set_ylabel('Ic (A)')
    axes[0, 0].set_title('原始數據比較')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 正規化數據比較
    exp_norm = (exp_Ic - np.mean(exp_Ic)) / np.std(exp_Ic)
    sim_norm = (sim_Ic - np.mean(sim_Ic)) / np.std(sim_Ic)
    axes[0, 1].plot(exp_y, exp_norm, 'b.-', label='實驗（正規化）', alpha=0.7, markersize=3)
    axes[0, 1].plot(sim_y, sim_norm, 'r.-', label='模擬（正規化）', alpha=0.7, markersize=3)
    axes[0, 1].set_xlabel('y_field')
    axes[0, 1].set_ylabel('正規化 Ic')
    axes[0, 1].set_title('正規化數據比較')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 散點圖和相關性
    axes[0, 2].scatter(exp_Ic, sim_Ic, alpha=0.6, s=20)
    min_val = min(np.min(exp_Ic), np.min(sim_Ic))
    max_val = max(np.max(exp_Ic), np.max(sim_Ic))
    axes[0, 2].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    axes[0, 2].set_xlabel('實驗 Ic (A)')
    axes[0, 2].set_ylabel('模擬 Ic (A)')
    axes[0, 2].set_title(f'相關性分析\\nr = {correlation:.4f}, p = {p_value:.4f}')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. 殘差分析
    residuals = sim_Ic - exp_Ic
    axes[1, 0].plot(exp_y, residuals, 'g.-', alpha=0.7, markersize=3)
    axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.8)
    axes[1, 0].fill_between(exp_y, residuals, alpha=0.3, color='green')
    axes[1, 0].set_xlabel('y_field')
    axes[1, 0].set_ylabel('殘差 (模擬 - 實驗)')
    axes[1, 0].set_title('殘差分析')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 分佈直方圖
    axes[1, 1].hist(exp_Ic, bins=20, alpha=0.6, label='實驗', density=True, color='blue')
    axes[1, 1].hist(sim_Ic, bins=20, alpha=0.6, label='模擬', density=True, color='red')
    axes[1, 1].set_xlabel('Ic (A)')
    axes[1, 1].set_ylabel('密度')
    axes[1, 1].set_title('數值分佈')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 統計摘要
    axes[1, 2].axis('off')
    
    stats_text = f"""統計摘要：
    
實驗數據：
  平均值: {np.mean(exp_Ic):.2e} A
  標準差: {np.std(exp_Ic):.2e} A
  變異係數: {np.std(exp_Ic)/np.mean(exp_Ic):.4f}
  範圍: [{np.min(exp_Ic):.2e}, {np.max(exp_Ic):.2e}]

模擬數據：
  平均值: {np.mean(sim_Ic):.2e} A
  標準差: {np.std(sim_Ic):.2e} A
  變異係數: {np.std(sim_Ic)/np.mean(sim_Ic):.4f}
  範圍: [{np.min(sim_Ic):.2e}, {np.max(sim_Ic):.2e}]

比較：
  相關係數: {correlation:.4f}
  p-值: {p_value:.4f}
  平均值比率: {np.mean(sim_Ic)/np.mean(exp_Ic):.4f}
  標準差比率: {np.std(sim_Ic)/np.std(exp_Ic):.4f}
"""
    
    axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes, fontsize=9,
                    verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # 保存圖表
    output_file = results_dir / f"deep_analysis_{filename.replace('.csv', '.png')}"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"深度分析圖表已保存到: {output_file}")
    
    plt.show()
    
    # 保存分析結果
    analysis_results = {
        'filename': filename,
        'correlation': float(correlation),
        'p_value': float(p_value),
        'exp_stats': {
            'mean': float(np.mean(exp_Ic)),
            'std': float(np.std(exp_Ic)),
            'cv': float(np.std(exp_Ic)/np.mean(exp_Ic))
        },
        'sim_stats': {
            'mean': float(np.mean(sim_Ic)),
            'std': float(np.std(sim_Ic)),
            'cv': float(np.std(sim_Ic)/np.mean(sim_Ic))
        }
    }
    
    analysis_file = results_dir / f"deep_analysis_{filename.replace('.csv', '.json')}"
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    print(f"分析結果已保存到: {analysis_file}")

def compare_all_improvements():
    """比較所有檔案的改進情況"""
    print("\\n=== 比較改進情況 ===")
    
    # 載入原始結果和改進結果
    results_dir = Path("/Users/albert-mac/Code/GitHub/Josephson/results")
    
    original_file = results_dir / "comparison_report.md"
    improved_file = results_dir / "improved_simulation_results.json"
    
    if improved_file.exists():
        with open(improved_file, 'r', encoding='utf-8') as f:
            improved_results = json.load(f)
        
        print("改進後的相關係數：")
        correlations = []
        for filename, results in improved_results.items():
            corr = results['correlation']
            if not np.isnan(corr):
                correlations.append(corr)
                print(f"  {filename}: {corr:.4f}")
        
        if correlations:
            print(f"\\n總結：")
            print(f"  平均相關係數: {np.mean(correlations):.4f}")
            print(f"  最高相關係數: {np.max(correlations):.4f}")
            print(f"  最低相關係數: {np.min(correlations):.4f}")
            print(f"  標準差: {np.std(correlations):.4f}")
            
            # 創建比較圖表
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            filenames = list(improved_results.keys())
            correlations_all = [improved_results[f]['correlation'] for f in filenames]
            
            bars = ax.bar(range(len(filenames)), correlations_all, alpha=0.7, color='skyblue')
            ax.set_xlabel('檔案')
            ax.set_ylabel('相關係數')
            ax.set_title('改進後各檔案相關係數比較')
            ax.set_xticks(range(len(filenames)))
            ax.set_xticklabels([f.replace('.csv', '') for f in filenames], rotation=45)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.8)
            
            # 標記最佳檔案
            best_idx = np.argmax(correlations_all)
            bars[best_idx].set_color('orange')
            ax.text(best_idx, correlations_all[best_idx] + 0.01, 'BEST', 
                   ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            comparison_file = results_dir / "correlation_comparison.png"
            plt.savefig(comparison_file, dpi=150, bbox_inches='tight')
            print(f"\\n比較圖表已保存到: {comparison_file}")
            plt.show()

if __name__ == "__main__":
    analyze_best_file()
    compare_all_improvements()
