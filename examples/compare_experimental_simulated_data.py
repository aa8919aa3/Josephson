"""
比較實驗數據與模擬數據的分析程式

這個腳本比較真實實驗數據和模擬數據的週期性模式、參數擬合結果和統計特性。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import signal, optimize
from scipy.stats import pearsonr, ks_2samp
import warnings
warnings.filterwarnings('ignore')

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class DataComparator:
    """數據比較分析類"""
    
    def __init__(self, exp_data_dir, sim_data_dir):
        """
        初始化數據比較器
        
        Parameters:
        -----------
        exp_data_dir : str or Path
            實驗數據目錄路徑
        sim_data_dir : str or Path
            模擬數據目錄路徑
        """
        self.exp_data_dir = Path(exp_data_dir)
        self.sim_data_dir = Path(sim_data_dir)
        self.exp_data = {}
        self.sim_data = {}
        self.comparison_results = {}
        
    def align_data_for_comparison(self, exp_data, sim_data):
        """
        對齊實驗和模擬數據以進行比較
        
        Parameters:
        -----------
        exp_data : DataFrame
            實驗數據
        sim_data : DataFrame
            模擬數據
            
        Returns:
        --------
        tuple
            對齊後的數據 (exp_aligned, sim_aligned)
        """
        min_length = min(len(exp_data), len(sim_data))
        
        # 截取到相同長度
        exp_aligned = exp_data.iloc[:min_length].copy()
        sim_aligned = sim_data.iloc[:min_length].copy()
        
        return exp_aligned, sim_aligned
    
    def normalize_magnetic_field(self, y_field):
        """
        將y_field轉換為正規化的磁場
        
        Parameters:
        -----------
        y_field : array-like
            原始磁場值
            
        Returns:
        --------
        array-like
            正規化的磁場值
        """
        field_range = y_field.max() - y_field.min()
        field_center = (y_field.max() + y_field.min()) / 2
        normalized_field = (y_field - field_center) / field_range
        return normalized_field
    
    def load_data(self):
        """載入所有實驗和模擬數據"""
        print("載入數據...")
        
        # 載入實驗數據
        exp_files = [f for f in self.exp_data_dir.glob("*.csv")]
        for file_path in exp_files:
            try:
                data = pd.read_csv(file_path)
                if len(data.columns) >= 2:
                    # 統一列名
                    data.columns = ['y_field', 'Ic']
                    self.exp_data[file_path.name] = data
                    print(f"✓ 載入實驗數據: {file_path.name}")
            except Exception as e:
                print(f"✗ 載入實驗數據失敗 {file_path.name}: {e}")
        
        # 載入模擬數據
        sim_files = [f for f in self.sim_data_dir.glob("*.csv") if f.name != "generate_simulated_data.py"]
        for file_path in sim_files:
            try:
                data = pd.read_csv(file_path)
                self.sim_data[file_path.name] = data
                print(f"✓ 載入模擬數據: {file_path.name}")
            except Exception as e:
                print(f"✗ 載入模擬數據失敗 {file_path.name}: {e}")
        
        print(f"\n總計載入: {len(self.exp_data)} 個實驗數據檔案, {len(self.sim_data)} 個模擬數據檔案")
    
    def sinc_function(self, y_field, Ic_max, field_scale, offset):
        """
        約瑟夫森結的 sinc 函數模型
        
        Parameters:
        -----------
        y_field : array-like
            磁場值
        Ic_max : float
            最大臨界電流
        field_scale : float
            磁場尺度參數
        offset : float
            偏移量
            
        Returns:
        --------
        array-like
            計算的臨界電流值
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            x = (y_field - offset) / field_scale
            result = Ic_max * np.abs(np.sin(np.pi * x) / (np.pi * x))
            result = np.where(np.abs(x) < 1e-10, Ic_max, result)
            return result
    
    def fit_sinc_pattern(self, y_field, Ic):
        """
        擬合 sinc 週期性模式
        
        Parameters:
        -----------
        y_field : array-like
            磁場值
        Ic : array-like
            臨界電流值
            
        Returns:
        --------
        dict
            擬合結果字典
        """
        try:
            # 初始參數猜測
            Ic_max_guess = np.max(Ic)
            field_range = np.max(y_field) - np.min(y_field)
            field_scale_guess = field_range / 4  # 假設在範圍內有約4個週期
            offset_guess = np.mean(y_field)
            
            initial_guess = [Ic_max_guess, field_scale_guess, offset_guess]
            
            # 執行擬合
            popt, pcov = optimize.curve_fit(
                self.sinc_function, 
                y_field, 
                Ic, 
                p0=initial_guess,
                maxfev=10000
            )
            
            # 計算擬合品質
            y_fit = self.sinc_function(y_field, *popt)
            r_squared = 1 - np.sum((Ic - y_fit)**2) / np.sum((Ic - np.mean(Ic))**2)
            
            # 計算參數不確定度
            param_errors = np.sqrt(np.diag(pcov)) if pcov is not None else [0, 0, 0]
            
            return {
                'Ic_max': popt[0],
                'field_scale': popt[1],
                'offset': popt[2],
                'Ic_max_error': param_errors[0],
                'field_scale_error': param_errors[1],
                'offset_error': param_errors[2],
                'r_squared': r_squared,
                'fitted_curve': y_fit,
                'success': True
            }
            
        except Exception as e:
            print(f"擬合失敗: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def analyze_periodicity(self, y_field, Ic):
        """
        分析週期性特徵
        
        Parameters:
        -----------
        y_field : array-like
            磁場值
        Ic : array-like
            臨界電流值
            
        Returns:
        --------
        dict
            週期性分析結果
        """
        try:
            # 進行FFT分析
            dt = np.mean(np.diff(y_field))
            frequencies = np.fft.fftfreq(len(Ic), dt)
            fft_Ic = np.fft.fft(Ic - np.mean(Ic))
            power_spectrum = np.abs(fft_Ic)**2
            
            # 找主頻率（除去DC分量）
            positive_freqs = frequencies[1:len(frequencies)//2]
            positive_power = power_spectrum[1:len(power_spectrum)//2]
            
            if len(positive_power) > 0:
                dominant_freq_idx = np.argmax(positive_power)
                dominant_frequency = positive_freqs[dominant_freq_idx]
                period = 1 / abs(dominant_frequency) if dominant_frequency != 0 else float('inf')
            else:
                dominant_frequency = 0
                period = float('inf')
            
            return {
                'dominant_frequency': dominant_frequency,
                'period': period,
                'power_spectrum': positive_power,
                'frequencies': positive_freqs,
                'success': True
            }
            
        except Exception as e:
            print(f"週期性分析失敗: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def compare_datasets(self):
        """比較實驗和模擬數據集"""
        print("\n開始數據比較分析...")
        
        for filename in self.exp_data.keys():
            if filename in self.sim_data:
                print(f"\n分析 {filename}...")
                
                exp_data = self.exp_data[filename]
                sim_data = self.sim_data[filename]
                
                # 確保數據格式一致
                exp_y = exp_data['y_field'].values
                exp_Ic = exp_data['Ic'].values
                sim_y = sim_data['y_field'].values
                sim_Ic = sim_data['Ic'].values
                
                # 擬合分析
                exp_fit = self.fit_sinc_pattern(exp_y, exp_Ic)
                sim_fit = self.fit_sinc_pattern(sim_y, sim_Ic)
                
                # 週期性分析
                exp_period = self.analyze_periodicity(exp_y, exp_Ic)
                sim_period = self.analyze_periodicity(sim_y, sim_Ic)
                
                # 統計比較
                if len(exp_Ic) == len(sim_Ic):
                    correlation, p_value = pearsonr(exp_Ic, sim_Ic)
                    ks_stat, ks_p_value = ks_2samp(exp_Ic, sim_Ic)
                else:
                    correlation, p_value = 0, 1
                    ks_stat, ks_p_value = 0, 1
                
                # 儲存比較結果
                self.comparison_results[filename] = {
                    'experimental': {
                        'fit': exp_fit,
                        'periodicity': exp_period,
                        'statistics': {
                            'mean': np.mean(exp_Ic),
                            'std': np.std(exp_Ic),
                            'min': np.min(exp_Ic),
                            'max': np.max(exp_Ic)
                        }
                    },
                    'simulated': {
                        'fit': sim_fit,
                        'periodicity': sim_period,
                        'statistics': {
                            'mean': np.mean(sim_Ic),
                            'std': np.std(sim_Ic),
                            'min': np.min(sim_Ic),
                            'max': np.max(sim_Ic)
                        }
                    },
                    'comparison': {
                        'correlation': correlation,
                        'correlation_p_value': p_value,
                        'ks_statistic': ks_stat,
                        'ks_p_value': ks_p_value
                    }
                }
                
                print(f"  相關係數: {correlation:.4f} (p={p_value:.4f})")
                if exp_fit['success'] and sim_fit['success']:
                    print(f"  實驗 R²: {exp_fit['r_squared']:.4f}")
                    print(f"  模擬 R²: {sim_fit['r_squared']:.4f}")
            else:
                print(f"⚠️ 找不到對應的模擬數據: {filename}")
    
    def plot_individual_comparison(self, filename, save_dir=None):
        """
        繪制單個檔案的比較圖
        
        Parameters:
        -----------
        filename : str
            檔案名稱
        save_dir : str or Path, optional
            保存目錄
        """
        if filename not in self.comparison_results:
            print(f"沒有找到 {filename} 的比較結果")
            return
        
        exp_data = self.exp_data[filename]
        sim_data = self.sim_data[filename]
        results = self.comparison_results[filename]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'數據比較分析: {filename}', fontsize=16, fontweight='bold')
        
        # 原始數據比較
        ax1 = axes[0, 0]
        ax1.plot(exp_data['y_field'], exp_data['Ic'], 'b.-', label='實驗數據', alpha=0.7)
        ax1.plot(sim_data['y_field'], sim_data['Ic'], 'r.-', label='模擬數據', alpha=0.7)
        ax1.set_xlabel('磁場 (T)')
        ax1.set_ylabel('臨界電流 (A)')
        ax1.set_title('原始數據比較')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 擬合結果比較
        ax2 = axes[0, 1]
        if results['experimental']['fit']['success']:
            exp_fit_curve = results['experimental']['fit']['fitted_curve']
            ax2.plot(exp_data['y_field'], exp_fit_curve, 'b-', label=f"實驗擬合 (R²={results['experimental']['fit']['r_squared']:.3f})", linewidth=2)
        
        if results['simulated']['fit']['success']:
            sim_fit_curve = results['simulated']['fit']['fitted_curve']
            ax2.plot(sim_data['y_field'], sim_fit_curve, 'r-', label=f"模擬擬合 (R²={results['simulated']['fit']['r_squared']:.3f})", linewidth=2)
        
        ax2.set_xlabel('磁場 (T)')
        ax2.set_ylabel('臨界電流 (A)')
        ax2.set_title('擬合曲線比較')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 殘差分析
        ax3 = axes[1, 0]
        if results['experimental']['fit']['success']:
            exp_residuals = exp_data['Ic'] - results['experimental']['fit']['fitted_curve']
            ax3.plot(exp_data['y_field'], exp_residuals, 'b.', label='實驗殘差', alpha=0.7)
        
        if results['simulated']['fit']['success']:
            sim_residuals = sim_data['Ic'] - results['simulated']['fit']['fitted_curve']
            ax3.plot(sim_data['y_field'], sim_residuals, 'r.', label='模擬殘差', alpha=0.7)
        
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax3.set_xlabel('磁場 (T)')
        ax3.set_ylabel('殘差 (A)')
        ax3.set_title('擬合殘差')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 統計比較
        ax4 = axes[1, 1]
        exp_stats = results['experimental']['statistics']
        sim_stats = results['simulated']['statistics']
        
        categories = ['平均值', '標準差', '最小值', '最大值']
        exp_values = [exp_stats['mean'], exp_stats['std'], exp_stats['min'], exp_stats['max']]
        sim_values = [sim_stats['mean'], sim_stats['std'], sim_stats['min'], sim_stats['max']]
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax4.bar(x - width/2, exp_values, width, label='實驗', alpha=0.8)
        ax4.bar(x + width/2, sim_values, width, label='模擬', alpha=0.8)
        ax4.set_xlabel('統計量')
        ax4.set_ylabel('值')
        ax4.set_title('統計特性比較')
        ax4.set_xticks(x)
        ax4.set_xticklabels(categories)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 添加相關性信息
        corr = results['comparison']['correlation']
        corr_p = results['comparison']['correlation_p_value']
        fig.text(0.02, 0.02, f'相關係數: {corr:.4f} (p={corr_p:.4f})', fontsize=10)
        
        plt.tight_layout()
        
        if save_dir:
            save_path = Path(save_dir) / f'comparison_{filename.replace(".csv", ".png")}'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"圖表已保存: {save_path}")
        
        plt.show()
    
    def plot_summary_comparison(self, save_dir=None):
        """
        繪制總結比較圖
        
        Parameters:
        -----------
        save_dir : str or Path, optional
            保存目錄
        """
        if not self.comparison_results:
            print("沒有比較結果可以繪制")
            return
        
        # 收集所有數據進行總結
        filenames = list(self.comparison_results.keys())
        correlations = []
        exp_r_squared = []
        sim_r_squared = []
        
        for filename in filenames:
            results = self.comparison_results[filename]
            correlations.append(results['comparison']['correlation'])
            
            if results['experimental']['fit']['success']:
                exp_r_squared.append(results['experimental']['fit']['r_squared'])
            
            if results['simulated']['fit']['success']:
                sim_r_squared.append(results['simulated']['fit']['r_squared'])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('實驗與模擬數據總結比較', fontsize=16, fontweight='bold')
        
        # 相關係數分布
        ax1 = axes[0, 0]
        ax1.hist(correlations, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('相關係數')
        ax1.set_ylabel('頻次')
        ax1.set_title('實驗-模擬數據相關係數分布')
        ax1.axvline(np.mean(correlations), color='red', linestyle='--', 
                   label=f'平均: {np.mean(correlations):.3f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # R²值比較
        ax2 = axes[0, 1]
        if exp_r_squared and sim_r_squared:
            ax2.scatter(exp_r_squared, sim_r_squared, alpha=0.7, s=60)
            ax2.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='y=x')
            ax2.set_xlabel('實驗數據 R²')
            ax2.set_ylabel('模擬數據 R²')
            ax2.set_title('擬合品質比較')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 相關係數vs檔案
        ax3 = axes[1, 0]
        x_pos = range(len(filenames))
        ax3.bar(x_pos, correlations, alpha=0.7, color='lightcoral')
        ax3.set_xlabel('檔案')
        ax3.set_ylabel('相關係數')
        ax3.set_title('各檔案相關係數')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([f.replace('.csv', '') for f in filenames], rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 擬合品質分布
        ax4 = axes[1, 1]
        if exp_r_squared and sim_r_squared:
            ax4.hist(exp_r_squared, bins=8, alpha=0.7, label='實驗', color='blue')
            ax4.hist(sim_r_squared, bins=8, alpha=0.7, label='模擬', color='red')
            ax4.set_xlabel('R² 值')
            ax4.set_ylabel('頻次')
            ax4.set_title('擬合品質分布')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_dir:
            save_path = Path(save_dir) / 'summary_comparison.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"總結圖表已保存: {save_path}")
        
        plt.show()
    
    def generate_report(self, save_path=None):
        """
        生成比較分析報告
        
        Parameters:
        -----------
        save_path : str or Path, optional
            報告保存路徑
        """
        if not self.comparison_results:
            print("沒有比較結果可以生成報告")
            return
        
        report = []
        report.append("# 實驗與模擬數據比較分析報告\n")
        report.append(f"生成時間: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"分析檔案數量: {len(self.comparison_results)}\n")
        
        # 總結統計
        correlations = [results['comparison']['correlation'] 
                       for results in self.comparison_results.values()]
        
        report.append("## 總結統計\n")
        report.append(f"- 平均相關係數: {np.mean(correlations):.4f}")
        report.append(f"- 相關係數標準差: {np.std(correlations):.4f}")
        report.append(f"- 最高相關係數: {np.max(correlations):.4f}")
        report.append(f"- 最低相關係數: {np.min(correlations):.4f}\n")
        
        # 各檔案詳細結果
        report.append("## 各檔案詳細結果\n")
        
        for filename, results in self.comparison_results.items():
            report.append(f"### {filename}\n")
            
            # 統計比較
            exp_stats = results['experimental']['statistics']
            sim_stats = results['simulated']['statistics']
            comp = results['comparison']
            
            report.append("**統計特性:**")
            report.append(f"- 實驗平均值: {exp_stats['mean']:.2e} A")
            report.append(f"- 模擬平均值: {sim_stats['mean']:.2e} A")
            report.append(f"- 相關係數: {comp['correlation']:.4f} (p={comp['correlation_p_value']:.4f})")
            
            # 擬合結果
            exp_fit = results['experimental']['fit']
            sim_fit = results['simulated']['fit']
            
            if exp_fit['success'] and sim_fit['success']:
                report.append("**擬合結果:**")
                report.append(f"- 實驗 R²: {exp_fit['r_squared']:.4f}")
                report.append(f"- 模擬 R²: {sim_fit['r_squared']:.4f}")
                report.append(f"- 實驗最大電流: {exp_fit['Ic_max']:.2e} A")
                report.append(f"- 模擬最大電流: {sim_fit['Ic_max']:.2e} A")
            
            report.append("")
        
        report_text = "\n".join(report)
        
        # 打印報告
        print(report_text)
        
        # 保存報告
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"\n報告已保存至: {save_path}")
        
        return report_text


def main():
    """主函數"""
    # 設定數據路徑
    exp_data_dir = "/Users/albert-mac/Code/GitHub/Josephson/data/experimental"
    sim_data_dir = "/Users/albert-mac/Code/GitHub/Josephson/data/simulated"
    
    # 創建比較器
    comparator = DataComparator(exp_data_dir, sim_data_dir)
    
    # 載入數據
    comparator.load_data()
    
    # 執行比較分析
    comparator.compare_datasets()
    
    # 創建結果目錄
    results_dir = Path("/Users/albert-mac/Code/GitHub/Josephson/results")
    results_dir.mkdir(exist_ok=True)
    
    # 繪制總結比較圖
    print("\n繪制總結比較圖...")
    comparator.plot_summary_comparison(save_dir=results_dir)
    
    # 生成報告
    print("\n生成分析報告...")
    report_path = results_dir / "comparison_report.md"
    comparator.generate_report(save_path=report_path)
    
    # 詢問是否繪制個別檔案比較圖
    print("\n是否要繪制個別檔案的詳細比較圖？")
    print("可用檔案:", list(comparator.comparison_results.keys()))
    
    return comparator


if __name__ == "__main__":
    comparator = main()
