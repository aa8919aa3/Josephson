"""
Detailed comparison plotting for single file

This script demonstrates how to perform detailed experimental-simulated data comparison for a single data file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project path
sys.path.append(str(Path(__file__).parent.parent))

# Configure matplotlib with English fonts only
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def normalize_magnetic_field(y_field):
    """
    Convert y_field to normalized magnetic field
    
    Parameters:
    -----------
    y_field : array-like
        Original magnetic field values
        
    Returns:
    --------
    array-like
        Normalized magnetic field values
    """
    # 將磁場正規化到磁通量子的單位
    field_range = y_field.max() - y_field.min()
    field_center = (y_field.max() + y_field.min()) / 2
    normalized_field = (y_field - field_center) / field_range
    return normalized_field


def align_data_arrays(exp_data, sim_data):
    """
    對齊實驗和模擬數據陣列，處理長度不匹配問題
    
    Parameters:
    -----------
    exp_data : DataFrame
        實驗數據
    sim_data : DataFrame
        模擬數據
        
    Returns:
    --------
    tuple
        對齊後的 (exp_data_aligned, sim_data_aligned)
    """
    min_length = min(len(exp_data), len(sim_data))
    
    # 截取到相同長度
    exp_aligned = exp_data.iloc[:min_length].copy()
    sim_aligned = sim_data.iloc[:min_length].copy()
    
    return exp_aligned, sim_aligned


def plot_detailed_comparison(filename="317Ic.csv"):
    """
    繪制詳細的比較圖
    
    Parameters:
    -----------
    filename : str
        要分析的檔案名稱
    """
    # 設定路徑
    exp_data_dir = Path("/Users/albert-mac/Code/GitHub/Josephson/data/experimental")
    sim_data_dir = Path("/Users/albert-mac/Code/GitHub/Josephson/data/simulated")
    results_dir = Path("/Users/albert-mac/Code/GitHub/Josephson/results")
    
    # 載入數據
    try:
        exp_data = pd.read_csv(exp_data_dir / filename)
        sim_data = pd.read_csv(sim_data_dir / filename)
        
        # 統一列名
        if len(exp_data.columns) >= 2:
            exp_data.columns = ['y_field', 'Ic']
        
        # 對齊數據長度
        exp_data, sim_data = align_data_arrays(exp_data, sim_data)
        
        # 計算正規化磁場
        exp_data['normalized_field'] = normalize_magnetic_field(exp_data['y_field'])
        sim_data['normalized_field'] = normalize_magnetic_field(sim_data['y_field'])
        
        print(f"成功載入數據: {filename}")
        print(f"數據長度: 實驗={len(exp_data)}, 模擬={len(sim_data)}")
        print(f"實驗數據範圍: Ic from {exp_data['Ic'].min():.2e} to {exp_data['Ic'].max():.2e} A")
        print(f"模擬數據範圍: Ic from {sim_data['Ic'].min():.2e} to {sim_data['Ic'].max():.2e} A")
        print(f"磁場範圍: {exp_data['y_field'].min():.4f} to {exp_data['y_field'].max():.4f}")
        
    except Exception as e:
        print(f"載入數據失敗: {e}")
        return
    
    # 創建比較圖
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'詳細數據比較: {filename}', fontsize=16, fontweight='bold')
    
    # 1. 原始數據重疊圖
    ax1 = axes[0, 0]
    ax1.plot(exp_data['y_field'], exp_data['Ic'], 'b.-', 
             label='Experimental Data', alpha=0.8, markersize=3)
    ax1.plot(sim_data['y_field'], sim_data['Ic'], 'r.-', 
             label='Simulated Data', alpha=0.8, markersize=3)
    ax1.set_xlabel('Magnetic Field (T)')
    ax1.set_ylabel('Critical Current (A)')
    ax1.set_title('Raw Data Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 正規化磁場比較
    ax2 = axes[0, 1]
    ax2.plot(exp_data['normalized_field'], exp_data['Ic'], 'b.-', 
             label='Experimental Data', alpha=0.8, markersize=3)
    ax2.plot(sim_data['normalized_field'], sim_data['Ic'], 'r.-', 
             label='Simulated Data', alpha=0.8, markersize=3)
    ax2.set_xlabel('Normalized Field')
    ax2.set_ylabel('Critical Current (A)')
    ax2.set_title('Normalized Field Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 正規化電流比較
    ax3 = axes[0, 2]
    # 正規化到 [0, 1] 範圍
    exp_norm = (exp_data['Ic'] - exp_data['Ic'].min()) / (exp_data['Ic'].max() - exp_data['Ic'].min())
    sim_norm = (sim_data['Ic'] - sim_data['Ic'].min()) / (sim_data['Ic'].max() - sim_data['Ic'].min())
    
    ax3.plot(exp_data['normalized_field'], exp_norm, 'b.-', 
             label='Experimental Data (Normalized)', alpha=0.8, markersize=3)
    ax3.plot(sim_data['normalized_field'], sim_norm, 'r.-', 
             label='Simulated Data (Normalized)', alpha=0.8, markersize=3)
    ax3.set_xlabel('Normalized Field')
    ax3.set_ylabel('Normalized Critical Current')
    ax3.set_title('Full Normalization Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 散點圖比較
    ax4 = axes[1, 0]
    ax4.scatter(exp_data['Ic'], sim_data['Ic'], alpha=0.6, s=30)
    
    # 繪制理想相關線
    min_val = min(exp_data['Ic'].min(), sim_data['Ic'].min())
    max_val = max(exp_data['Ic'].max(), sim_data['Ic'].max())
    ax4.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='理想相關 (y=x)')
    
    # 計算實際相關係數
    correlation = np.corrcoef(exp_data['Ic'], sim_data['Ic'])[0, 1]
    ax4.set_xlabel('實驗臨界電流 (A)')
    ax4.set_ylabel('模擬臨界電流 (A)')
    ax4.set_title(f'點對點比較 (r={correlation:.3f})')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 統計分布比較
    ax5 = axes[1, 1]
    ax5.hist(exp_data['Ic'], bins=20, alpha=0.6, label='實驗數據', 
             color='blue', density=True)
    ax5.hist(sim_data['Ic'], bins=20, alpha=0.6, label='模擬數據', 
             color='red', density=True)
    ax5.set_xlabel('臨界電流 (A)')
    ax5.set_ylabel('機率密度')
    ax5.set_title('分布比較')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 差異分析
    ax6 = axes[1, 2]
    diff = exp_data['Ic'].values - sim_data['Ic'].values
    ax6.plot(exp_data['normalized_field'], diff, 'g.-', alpha=0.7, markersize=3)
    ax6.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax6.set_xlabel('正規化磁場')
    ax6.set_ylabel('差異 (實驗 - 模擬)')
    ax6.set_title('逐點差異分析')
    ax6.grid(True, alpha=0.3)
    
    # 添加統計信息
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    ax6.text(0.02, 0.98, f'平均差異: {mean_diff:.2e}\n標準差: {std_diff:.2e}', 
             transform=ax6.transAxes, va='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    
    # 保存圖表
    save_path = results_dir / f'detailed_comparison_{filename.replace(".csv", ".png")}'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"詳細比較圖已保存: {save_path}")
    
    plt.show()
    
    # 打印統計摘要
    print(f"\n=== {filename} 統計摘要 ===")
    print(f"實驗數據:")
    print(f"  平均值: {exp_data['Ic'].mean():.2e} A")
    print(f"  標準差: {exp_data['Ic'].std():.2e} A")
    print(f"  變異係數: {exp_data['Ic'].std()/exp_data['Ic'].mean():.3f}")
    
    print(f"模擬數據:")
    print(f"  平均值: {sim_data['Ic'].mean():.2e} A")
    print(f"  標準差: {sim_data['Ic'].std():.2e} A")
    print(f"  變異係數: {sim_data['Ic'].std()/sim_data['Ic'].mean():.3f}")
    
    # 計算更詳細的相關係數
    if len(exp_data) == len(sim_data):
        correlation = np.corrcoef(exp_data['Ic'], sim_data['Ic'])[0, 1]
        # 計算正規化數據的相關係數
        norm_correlation = np.corrcoef(exp_norm, sim_norm)[0, 1]
        print(f"原始數據相關係數: {correlation:.4f}")
        print(f"正規化數據相關係數: {norm_correlation:.4f}")
    else:
        correlation = 0
        norm_correlation = 0
        print("數據長度不匹配，無法計算相關係數")


def compare_multiple_files():
    """比較多個檔案的快速概覽"""
    exp_data_dir = Path("/Users/albert-mac/Code/GitHub/Josephson/data/experimental")
    sim_data_dir = Path("/Users/albert-mac/Code/GitHub/Josephson/data/simulated")
    
    # 獲取所有對應的檔案
    exp_files = [f.name for f in exp_data_dir.glob("*.csv")]
    sim_files = [f.name for f in sim_data_dir.glob("*.csv")]
    common_files = list(set(exp_files) & set(sim_files))
    
    print(f"找到 {len(common_files)} 個對應的檔案:")
    for filename in sorted(common_files):
        print(f"  - {filename}")
    
    # 創建多檔案比較圖
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle('多檔案數據概覽', fontsize=16, fontweight='bold')
    
    for idx, filename in enumerate(sorted(common_files)):
        if idx >= 12:  # 限制在12個子圖內
            break
            
        row = idx // 4
        col = idx % 4
        ax = axes[row, col]
        
        try:
            # 載入數據
            exp_data = pd.read_csv(exp_data_dir / filename)
            sim_data = pd.read_csv(sim_data_dir / filename)
            
            # 對齊數據長度
            exp_data, sim_data = align_data_arrays(exp_data, sim_data)
            
            # 繪制比較
            ax.plot(exp_data['y_field'], exp_data['Ic'], 'b-', 
                   label='實驗', alpha=0.7, linewidth=1.5)
            ax.plot(sim_data['y_field'], sim_data['Ic'], 'r-', 
                   label='模擬', alpha=0.7, linewidth=1.5)
            
            ax.set_title(filename.replace('.csv', ''), fontsize=10)
            ax.set_xlabel('磁場 (T)', fontsize=8)
            ax.set_ylabel('Ic (A)', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            if idx == 0:  # 只在第一個子圖顯示圖例
                ax.legend(fontsize=8)
                
        except Exception as e:
            ax.text(0.5, 0.5, f'載入失敗:\n{filename}', 
                   transform=ax.transAxes, ha='center', va='center')
    
    # 隱藏多餘的子圖
    for idx in range(len(common_files), 12):
        row = idx // 4
        col = idx % 4
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    
    # 保存多檔案比較圖
    results_dir = Path("/Users/albert-mac/Code/GitHub/Josephson/results")
    save_path = results_dir / 'multi_file_overview.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"多檔案概覽圖已保存: {save_path}")
    
    plt.show()


def main():
    """主函數"""
    print("=== 約瑟夫森結數據比較分析 ===\n")
    
    # 1. 多檔案概覽
    print("1. 生成多檔案概覽...")
    compare_multiple_files()
    
    # 2. 單檔案詳細分析
    print("\n2. 單檔案詳細分析...")
    
    # 選擇一個具有較好擬合結果的檔案進行詳細分析
    plot_detailed_comparison("317Ic.csv")  # 新的實驗數據檔案
    
    print("\n3. 可以嘗試其他檔案的詳細分析:")
    print("   - 335Ic.csv")
    print("   - 336Ic.csv")
    print("   - kay164Ic-.csv")
    print("   - kay164Ic+.csv")
    
    print("\n分析完成！")


if __name__ == "__main__":
    main()
