#!/usr/bin/env python3
"""
綜合性能比較工具：比較所有分析方法的效果
Comprehensive Performance Comparison Tool
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
from scipy.stats import pearsonr
from datetime import datetime
import seaborn as sns

# Configure matplotlib with English fonts only
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 100

class ComprehensiveComparator:
    """綜合性能比較器"""
    
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.exp_dir = self.base_dir / "data" / "experimental"
        self.sim_dir = self.base_dir / "data" / "simulated"
        self.advanced_sim_dir = self.sim_dir / "advanced"
        self.results_dir = self.base_dir / "results"
        
        # 確保目錄存在
        self.advanced_sim_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        # 載入所有結果
        self.basic_results = self._load_json_results("improved_simulation_results.json")
        self.advanced_results = self._load_json_results("advanced_simulation_results.json")
        self.ml_params = self._load_json_results("ml_optimized_parameters.json")
        
    def _load_json_results(self, filename):
        """載入JSON結果檔案"""
        filepath = self.results_dir / filename
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def calculate_correlations_with_ml_params(self):
        """使用ML優化參數計算相關係數"""
        print("🤖 使用ML優化參數計算相關係數...")
        
        ml_correlations = {}
        
        for filename, params in self.ml_params.items():
            exp_file = self.exp_dir / filename
            if not exp_file.exists():
                continue
                
            try:
                # 載入實驗數據
                exp_data = pd.read_csv(exp_file)
                exp_y = exp_data['y_field'].values
                exp_Ic = exp_data['Ic'].values
                
                # 使用ML優化參數生成模擬數據
                sim_Ic = self._generate_ml_simulation(exp_y, params)
                
                # 計算相關係數
                correlation, p_value = pearsonr(exp_Ic, sim_Ic)
                
                ml_correlations[filename] = {
                    'correlation': correlation,
                    'p_value': p_value,
                    'parameters': params
                }
                
                print(f"   ✅ {filename}: r = {correlation:.4f}")
                
            except Exception as e:
                print(f"   ❌ {filename}: 錯誤 - {e}")
                ml_correlations[filename] = {
                    'correlation': np.nan,
                    'p_value': np.nan,
                    'error': str(e)
                }
        
        return ml_correlations
    
    def _generate_ml_simulation(self, y_field, params):
        """使用ML參數生成模擬數據"""
        flux_quantum = 2.067e-15
        phi_ext = y_field * params['field_scale']
        normalized_flux = np.pi * phi_ext / flux_quantum
        
        with np.errstate(divide='ignore', invalid='ignore'):
            sinc_term = np.sin(normalized_flux + params['phase_offset']) / (normalized_flux + 1e-10)
            sinc_term = np.where(np.abs(normalized_flux) < 1e-10, 1.0, sinc_term)
        
        pattern = np.abs(sinc_term) * (1 + params['asymmetry'] * np.cos(2 * normalized_flux))
        sim_Ic = params['Ic_base'] * pattern
        
        # 添加噪聲
        if 'noise_level' in params:
            noise = np.random.normal(0, params['noise_level'], len(sim_Ic))
            sim_Ic += noise
            
        return sim_Ic
    
    def create_comprehensive_comparison(self):
        """創建綜合比較分析"""
        print("\n📊 創建綜合性能比較...")
        
        # 計算ML相關係數
        ml_correlations = self.calculate_correlations_with_ml_params()
        
        # 收集所有數據
        comparison_data = []
        
        # 獲取所有實驗檔案
        exp_files = list(self.exp_dir.glob("*.csv"))
        
        for exp_file in exp_files:
            filename = exp_file.name
            
            row = {
                'filename': filename,
                'basic_correlation': self.basic_results.get(filename, {}).get('correlation', np.nan),
                'advanced_correlation': self.advanced_results.get(filename, {}).get('correlation', np.nan),
                'ml_correlation': ml_correlations.get(filename, {}).get('correlation', np.nan)
            }
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # 計算改進度
        df['basic_to_advanced'] = df['advanced_correlation'] - df['basic_correlation']
        df['basic_to_ml'] = df['ml_correlation'] - df['basic_correlation']
        df['advanced_to_ml'] = df['ml_correlation'] - df['advanced_correlation']
        
        return df, ml_correlations
    
    def create_visualization(self, df):
        """創建可視化圖表"""
        print("🎨 創建可視化圖表...")
        
        # 創建綜合比較圖
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('約瑟夫森結分析方法綜合性能比較', fontsize=16, fontweight='bold')
        
        # 1. 相關係數比較（條形圖）
        ax = axes[0, 0]
        x = np.arange(len(df))
        width = 0.25
        
        ax.bar(x - width, df['basic_correlation'], width, label='Basic Model', alpha=0.8, color='blue')
        ax.bar(x, df['advanced_correlation'], width, label='Advanced Physics Model', alpha=0.8, color='red')
        ax.bar(x + width, df['ml_correlation'], width, label='ML Optimized Model', alpha=0.8, color='green')
        
        ax.set_xlabel('Experimental Files')
        ax.set_ylabel('Correlation Coefficient')
        ax.set_title('Correlation Coefficient Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([f.replace('.csv', '') for f in df['filename']], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # 2. 改進度分析（條形圖）
        ax = axes[0, 1]
        ax.bar(x - width/2, df['basic_to_advanced'], width, label='Basic→Advanced', alpha=0.8, color='orange')
        ax.bar(x + width/2, df['basic_to_ml'], width, label='Basic→ML', alpha=0.8, color='purple')
        
        ax.set_xlabel('Experimental Files')
        ax.set_ylabel('Correlation Improvement')
        ax.set_title('Method Improvement Effect')
        ax.set_xticks(x)
        ax.set_xticklabels([f.replace('.csv', '') for f in df['filename']], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # 3. 散點圖比較
        ax = axes[0, 2]
        valid_mask = ~(np.isnan(df['basic_correlation']) | np.isnan(df['ml_correlation']))
        if valid_mask.sum() > 0:
            ax.scatter(df.loc[valid_mask, 'basic_correlation'], 
                      df.loc[valid_mask, 'ml_correlation'], 
                      alpha=0.7, s=100, color='green')
            
            # 添加對角線
            min_val = min(df.loc[valid_mask, 'basic_correlation'].min(), 
                         df.loc[valid_mask, 'ml_correlation'].min())
            max_val = max(df.loc[valid_mask, 'basic_correlation'].max(), 
                         df.loc[valid_mask, 'ml_correlation'].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            ax.set_xlabel('基礎模型相關係數')
            ax.set_ylabel('ML優化模型相關係數')
            ax.set_title('基礎 vs ML優化 散點圖')
            ax.grid(True, alpha=0.3)
        
        # 4. 統計摘要（直方圖）
        ax = axes[1, 0]
        methods = ['基礎模型', '進階物理模型', 'ML優化模型']
        means = [df['basic_correlation'].mean(), 
                df['advanced_correlation'].mean(), 
                df['ml_correlation'].mean()]
        stds = [df['basic_correlation'].std(), 
               df['advanced_correlation'].std(), 
               df['ml_correlation'].std()]
        
        x_pos = np.arange(len(methods))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.8, 
                     color=['blue', 'red', 'green'])
        
        ax.set_xlabel('分析方法')
        ax.set_ylabel('平均相關係數')
        ax.set_title('方法性能統計摘要')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # 添加數值標籤
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                   f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 5. 熱圖比較
        ax = axes[1, 1]
        correlation_matrix = df[['basic_correlation', 'advanced_correlation', 'ml_correlation']].T
        correlation_matrix.columns = [f.replace('.csv', '') for f in df['filename']]
        
        im = ax.imshow(correlation_matrix.values, cmap='RdBu_r', aspect='auto', vmin=-0.2, vmax=0.2)
        ax.set_xticks(range(len(correlation_matrix.columns)))
        ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha='right')
        ax.set_yticks(range(len(correlation_matrix.index)))
        ax.set_yticklabels(['Basic', 'Advanced', 'ML'])
        ax.set_title('Correlation Coefficient Heatmap')
        
        # 添加顏色條
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation Coefficient')
        
        # 添加數值標籤
        for i in range(len(correlation_matrix.index)):
            for j in range(len(correlation_matrix.columns)):
                text = ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.3f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        # 6. 改進幅度排序
        ax = axes[1, 2]
        improvement_df = df[['filename', 'basic_to_ml']].sort_values('basic_to_ml', ascending=True)
        y_pos = np.arange(len(improvement_df))
        
        colors = ['red' if x < 0 else 'green' for x in improvement_df['basic_to_ml']]
        bars = ax.barh(y_pos, improvement_df['basic_to_ml'], color=colors, alpha=0.7)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f.replace('.csv', '') for f in improvement_df['filename']])
        ax.set_xlabel('相關係數改進度')
        ax.set_title('ML優化改進幅度排序')
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        # 保存圖表
        output_file = self.results_dir / "comprehensive_performance_comparison.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✅ 綜合比較圖表已保存至: {output_file}")
        
        plt.show()
        
        return output_file
    
    def generate_comprehensive_report(self, df, ml_correlations):
        """生成綜合分析報告"""
        print("📝 生成綜合分析報告...")
        
        timestamp = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
        
        report = f"""# 約瑟夫森結數據分析綜合性能比較報告

生成時間: {timestamp}

## 執行摘要

本報告比較了三種不同的約瑟夫森結數據分析方法：
1. **基礎模型**: 標準約瑟夫森結物理模型
2. **進階物理模型**: 包含溫度效應、非線性效應的進階物理模型  
3. **ML優化模型**: 基於機器學習優化參數的模型

## 性能統計摘要

### 平均相關係數
- 基礎模型: {df['basic_correlation'].mean():.4f} ± {df['basic_correlation'].std():.4f}
- 進階物理模型: {df['advanced_correlation'].mean():.4f} ± {df['advanced_correlation'].std():.4f}
- ML優化模型: {df['ml_correlation'].mean():.4f} ± {df['ml_correlation'].std():.4f}

### 改進效果分析
- 基礎→進階物理: 平均改進 {df['basic_to_advanced'].mean():.4f}
- 基礎→ML優化: 平均改進 {df['basic_to_ml'].mean():.4f}
- 進階→ML優化: 平均改進 {df['advanced_to_ml'].mean():.4f}

## 詳細結果

| 實驗檔案 | 基礎模型 | 進階物理模型 | ML優化模型 | 基礎→ML改進 |
|---------|---------|-------------|-----------|------------|
"""
        
        for _, row in df.iterrows():
            report += f"| {row['filename'].replace('.csv', '')} | {row['basic_correlation']:.4f} | {row['advanced_correlation']:.4f} | {row['ml_correlation']:.4f} | {row['basic_to_ml']:.4f} |\n"
        
        # 添加最佳和最差表現分析
        best_basic = df.loc[df['basic_correlation'].idxmax()]
        best_advanced = df.loc[df['advanced_correlation'].idxmax()]
        best_ml = df.loc[df['ml_correlation'].idxmax()]
        
        report += f"""
## 性能亮點

### 最佳表現
- **基礎模型最佳**: {best_basic['filename']} (r = {best_basic['basic_correlation']:.4f})
- **進階物理模型最佳**: {best_advanced['filename']} (r = {best_advanced['advanced_correlation']:.4f})
- **ML優化模型最佳**: {best_ml['filename']} (r = {best_ml['ml_correlation']:.4f})

### 改進幅度最大
"""
        
        max_improvement = df.loc[df['basic_to_ml'].idxmax()]
        min_improvement = df.loc[df['basic_to_ml'].idxmin()]
        
        report += f"""- **最大改進**: {max_improvement['filename']} (+{max_improvement['basic_to_ml']:.4f})
- **最大退化**: {min_improvement['filename']} ({min_improvement['basic_to_ml']:.4f})

## ML優化參數分析

"""
        
        for filename, data in ml_correlations.items():
            if 'parameters' in data:
                params = data['parameters']
                report += f"""### {filename.replace('.csv', '')}
- 預測相關係數: {data['correlation']:.4f}
- 優化參數:
  - Ic_base: {params['Ic_base']:.2e}
  - field_scale: {params['field_scale']:.2e}
  - phase_offset: {params['phase_offset']:.4f}
  - asymmetry: {params['asymmetry']:.4f}
  - noise_level: {params.get('noise_level', 'N/A')}

"""
        
        report += f"""
## 結論與建議

### 方法效果評估
1. **進階物理模型** 相比基礎模型平均改進 {df['basic_to_advanced'].mean():.4f}
2. **ML優化模型** 相比基礎模型平均改進 {df['basic_to_ml'].mean():.4f}
3. **最佳整體表現**: {"ML優化模型" if df['ml_correlation'].mean() > df['advanced_correlation'].mean() else "進階物理模型"}

### 未來發展方向
1. **混合模型**: 結合進階物理模型和ML優化的優點
2. **深度學習**: 探索神經網路在模式識別方面的潛力
3. **多溫度建模**: 擴展到溫度相關效應的分析
4. **實時優化**: 開發自適應參數調整系統
5. **更多實驗數據**: 收集更大規模的實驗數據集以改善模型訓練

### 技術建議
- 對於高雜訊數據，推薦使用ML優化模型
- 對於物理理解需求，推薦使用進階物理模型
- 對於快速分析，基礎模型仍然具有價值

---
*報告由綜合性能比較系統自動生成*
"""
        
        # 保存報告
        report_file = self.results_dir / "comprehensive_performance_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✅ 綜合分析報告已保存至: {report_file}")
        
        return report_file

def main():
    """主要執行函數"""
    print("=== 約瑟夫森結綜合性能比較系統 ===\n")
    
    base_dir = "/Users/albert-mac/Code/GitHub/Josephson"
    comparator = ComprehensiveComparator(base_dir)
    
    # 執行綜合比較
    df, ml_correlations = comparator.create_comprehensive_comparison()
    
    # 創建可視化
    chart_file = comparator.create_visualization(df)
    
    # 生成報告
    report_file = comparator.generate_comprehensive_report(df, ml_correlations)
    
    # 保存比較數據
    comparison_file = comparator.results_dir / "comprehensive_comparison_data.json"
    comparison_data = {
        'comparison_table': df.to_dict('records'),
        'ml_correlations': ml_correlations,
        'summary_stats': {
            'basic_mean': float(df['basic_correlation'].mean()),
            'advanced_mean': float(df['advanced_correlation'].mean()),
            'ml_mean': float(df['ml_correlation'].mean()),
            'basic_std': float(df['basic_correlation'].std()),
            'advanced_std': float(df['advanced_correlation'].std()),
            'ml_std': float(df['ml_correlation'].std())
        }
    }
    
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 比較數據已保存至: {comparison_file}")
    print("\n🎉 綜合性能比較分析完成！")
    
    # 輸出關鍵統計
    print(f"\n📊 關鍵統計:")
    print(f"   基礎模型平均相關係數: {df['basic_correlation'].mean():.4f}")
    print(f"   進階物理模型平均相關係數: {df['advanced_correlation'].mean():.4f}")
    print(f"   ML優化模型平均相關係數: {df['ml_correlation'].mean():.4f}")
    print(f"   最佳整體方法: {'ML優化模型' if df['ml_correlation'].mean() > df['advanced_correlation'].mean() else '進階物理模型'}")

if __name__ == "__main__":
    main()
