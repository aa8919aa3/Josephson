#!/usr/bin/env python3
"""
智能分析協調器
整合所有分析工具，提供智能化的約瑟夫森結數據分析流程
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import subprocess
import sys
from datetime import datetime
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# 設定matplotlib
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')

class IntelligentAnalysisCoordinator:
    """
    智能分析協調器
    自動選擇最適合的分析方法並執行完整的分析流程
    """
    
    def __init__(self, base_dir=None):
        if base_dir is None:
            self.base_dir = Path("/Users/albert-mac/Code/GitHub/Josephson")
        else:
            self.base_dir = Path(base_dir)
            
        self.exp_data_dir = self.base_dir / "data" / "experimental"
        self.sim_data_dir = self.base_dir / "data" / "simulated"
        self.advanced_sim_dir = self.base_dir / "data" / "simulated" / "advanced"
        self.results_dir = self.base_dir / "results"
        self.examples_dir = self.base_dir / "examples"
        
        # 確保目錄存在
        self.advanced_sim_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        self.analysis_history = []
        self.current_best_correlations = {}
    
    def assess_data_quality(self):
        """
        評估實驗數據品質
        """
        print("🔍 評估實驗數據品質...")
        
        data_quality = {}
        
        for exp_file in self.exp_data_dir.glob("*.csv"):
            try:
                data = pd.read_csv(exp_file)
                
                if 'y_field' in data.columns and 'Ic' in data.columns:
                    y_field = data['y_field'].values
                    Ic = data['Ic'].values
                    
                    quality_metrics = {
                        'n_points': len(data),
                        'field_range': np.ptp(y_field),
                        'current_range': np.ptp(Ic),
                        'snr_estimate': np.mean(Ic) / np.std(Ic),
                        'missing_values': data.isnull().sum().sum(),
                        'cv': np.std(Ic) / np.mean(Ic),
                        'monotonicity': self._assess_monotonicity(y_field),
                        'periodicity_strength': self._assess_periodicity(Ic)
                    }
                    
                    # 總體品質評分 (0-100)
                    quality_score = self._calculate_quality_score(quality_metrics)
                    quality_metrics['quality_score'] = quality_score
                    
                    data_quality[exp_file.name] = quality_metrics
                    
                    print(f"   {exp_file.name}: 品質評分 {quality_score:.1f}/100")
                    
            except Exception as e:
                print(f"❌ 評估 {exp_file.name} 時發生錯誤: {e}")
                data_quality[exp_file.name] = {'error': str(e)}
        
        return data_quality
    
    def _assess_monotonicity(self, y_field):
        """評估磁場數據的單調性"""
        if len(y_field) < 2:
            return 0.0
        
        diff = np.diff(y_field)
        if np.all(diff > 0):
            return 1.0  # 完全遞增
        elif np.all(diff < 0):
            return 1.0  # 完全遞減
        else:
            # 計算單調性比例
            consistent_direction = max(np.sum(diff > 0), np.sum(diff < 0))
            return consistent_direction / len(diff)
    
    def _assess_periodicity(self, Ic):
        """評估電流數據的週期性強度"""
        if len(Ic) < 10:
            return 0.0
        
        # 自相關分析
        Ic_centered = Ic - np.mean(Ic)
        autocorr = np.correlate(Ic_centered, Ic_centered, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0]
        
        # 尋找第一個顯著的局部最大值
        for i in range(2, min(len(autocorr) - 2, len(Ic) // 4)):
            if (autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1] and 
                autocorr[i] > 0.1):
                return autocorr[i]
        
        return 0.0
    
    def _calculate_quality_score(self, metrics):
        """計算數據品質綜合評分"""
        score = 0
        
        # 數據點數評分 (0-20分)
        if metrics['n_points'] >= 100:
            score += 20
        elif metrics['n_points'] >= 50:
            score += 15
        elif metrics['n_points'] >= 20:
            score += 10
        else:
            score += 5
        
        # 信噪比評分 (0-25分)
        snr = metrics['snr_estimate']
        if snr >= 10:
            score += 25
        elif snr >= 5:
            score += 20
        elif snr >= 2:
            score += 15
        else:
            score += max(0, snr * 5)
        
        # 單調性評分 (0-15分)
        score += metrics['monotonicity'] * 15
        
        # 週期性評分 (0-20分)
        score += metrics['periodicity_strength'] * 20
        
        # 變異係數評分 (0-10分) - 適當的變異是好的
        cv = metrics['cv']
        if 0.1 <= cv <= 0.5:
            score += 10
        elif 0.05 <= cv <= 0.8:
            score += 8
        else:
            score += max(0, 10 - abs(cv - 0.3) * 10)
        
        # 缺失值懲罰 (0-10分)
        if metrics['missing_values'] == 0:
            score += 10
        else:
            score += max(0, 10 - metrics['missing_values'])
        
        return min(100, score)
    
    def select_optimal_analysis_strategy(self, data_quality):
        """
        根據數據品質選擇最佳分析策略
        """
        print("\n🎯 選擇最佳分析策略...")
        
        strategies = {}
        
        for filename, quality in data_quality.items():
            if 'error' in quality:
                strategies[filename] = 'skip'
                continue
                
            score = quality['quality_score']
            n_points = quality['n_points']
            snr = quality['snr_estimate']
            periodicity = quality['periodicity_strength']
            
            if score >= 80:
                # 高品質數據：使用所有進階方法
                strategy = 'comprehensive_advanced'
            elif score >= 60:
                # 中等品質數據：使用標準進階方法
                strategy = 'standard_advanced'
            elif score >= 40:
                # 低品質數據：使用基礎方法加改進
                strategy = 'basic_improved'
            else:
                # 極低品質數據：僅基礎分析
                strategy = 'basic_only'
            
            # 特殊情況調整
            if n_points < 30:
                strategy = 'basic_only'
            elif snr < 2:
                strategy = max('basic_improved', strategy)
            elif periodicity > 0.5:
                # 強週期性，適合進階分析
                if strategy == 'basic_only':
                    strategy = 'basic_improved'
            
            strategies[filename] = strategy
            print(f"   {filename}: {strategy} (品質評分: {score:.1f})")
        
        return strategies
    
    def execute_analysis_pipeline(self, strategies):
        """
        執行完整的分析管道
        """
        print("\n🚀 執行分析管道...")
        
        pipeline_results = {}
        
        # 第一階段：基礎和改進模擬
        print("\n📊 第一階段：基礎模擬數據生成...")
        basic_results = self._run_basic_simulation()
        
        # 第二階段：機器學習優化 (針對中高品質數據)
        ml_candidates = [f for f, s in strategies.items() 
                        if s in ['standard_advanced', 'comprehensive_advanced']]
        
        if ml_candidates:
            print("\n🤖 第二階段：機器學習優化...")
            ml_results = self._run_ml_optimization(ml_candidates)
            pipeline_results['ml_optimization'] = ml_results
        
        # 第三階段：進階物理模型 (針對高品質數據)
        physics_candidates = [f for f, s in strategies.items() 
                            if s == 'comprehensive_advanced']
        
        if physics_candidates:
            print("\n🔬 第三階段：進階物理模型...")
            physics_results = self._run_advanced_physics(physics_candidates)
            pipeline_results['advanced_physics'] = physics_results
        
        # 第四階段：深度分析和可視化
        print("\n📈 第四階段：深度分析和可視化...")
        visualization_results = self._run_comprehensive_visualization()
        pipeline_results['visualization'] = visualization_results
        
        return pipeline_results
    
    def _run_basic_simulation(self):
        """運行基礎模擬"""
        try:
            script_path = self.examples_dir / "generate_improved_simulation.py"
            if script_path.exists():
                result = subprocess.run([sys.executable, str(script_path)], 
                                      capture_output=True, text=True, cwd=str(self.base_dir))
                return {'success': result.returncode == 0, 'output': result.stdout, 'error': result.stderr}
            else:
                return {'success': False, 'error': 'Script not found'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _run_ml_optimization(self, candidates):
        """運行機器學習優化"""
        try:
            script_path = self.examples_dir / "advanced_ml_optimization.py"
            if script_path.exists():
                result = subprocess.run([sys.executable, str(script_path)], 
                                      capture_output=True, text=True, cwd=str(self.base_dir))
                return {'success': result.returncode == 0, 'output': result.stdout, 'error': result.stderr}
            else:
                return {'success': False, 'error': 'ML script not found'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _run_advanced_physics(self, candidates):
        """運行進階物理模型"""
        try:
            script_path = self.examples_dir / "advanced_physics_modeling.py"
            if script_path.exists():
                result = subprocess.run([sys.executable, str(script_path)], 
                                      capture_output=True, text=True, cwd=str(self.base_dir))
                return {'success': result.returncode == 0, 'output': result.stdout, 'error': result.stderr}
            else:
                return {'success': False, 'error': 'Physics script not found'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _run_comprehensive_visualization(self):
        """運行綜合可視化"""
        try:
            script_path = self.examples_dir / "simple_deep_analysis.py"
            if script_path.exists():
                result = subprocess.run([sys.executable, str(script_path)], 
                                      capture_output=True, text=True, cwd=str(self.base_dir))
                return {'success': result.returncode == 0, 'output': result.stdout, 'error': result.stderr}
            else:
                return {'success': False, 'error': 'Visualization script not found'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def evaluate_final_results(self):
        """
        評估最終結果並選擇最佳模型
        """
        print("\n📊 評估最終結果...")
        
        final_evaluation = {}
        
        # 收集所有可用的結果文件
        result_files = {
            'basic': self.results_dir / "improved_simulation_results.json",
            'ml': self.results_dir / "ml_optimized_parameters.json",
            'advanced': self.results_dir / "advanced_simulation_results.json"
        }
        
        for method, file_path in result_files.items():
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    final_evaluation[method] = data
                except Exception as e:
                    print(f"❌ 載入 {method} 結果失敗: {e}")
        
        # 比較所有方法的性能
        performance_summary = self._compare_all_methods(final_evaluation)
        
        # 生成最終建議
        recommendations = self._generate_recommendations(performance_summary)
        
        return {
            'performance_summary': performance_summary,
            'recommendations': recommendations,
            'final_evaluation': final_evaluation
        }
    
    def _compare_all_methods(self, evaluation_data):
        """比較所有方法的性能"""
        comparison = {}
        
        for exp_file in self.exp_data_dir.glob("*.csv"):
            filename = exp_file.name
            comparison[filename] = {}
            
            # 載入實驗數據
            try:
                exp_data = pd.read_csv(exp_file)
                exp_Ic = exp_data['Ic'].values
                
                # 比較不同方法
                for method in ['basic', 'advanced']:
                    sim_file = None
                    
                    if method == 'basic':
                        sim_file = self.sim_data_dir / f"improved_sim_{filename}"
                    elif method == 'advanced':
                        sim_file = self.advanced_sim_dir / f"advanced_sim_{filename}"
                    
                    if sim_file and sim_file.exists():
                        sim_data = pd.read_csv(sim_file)
                        sim_Ic = sim_data['Ic'].values
                        
                        # 計算相關係數
                        correlation, p_value = pearsonr(exp_Ic, sim_Ic)
                        
                        comparison[filename][method] = {
                            'correlation': correlation,
                            'p_value': p_value,
                            'available': True
                        }
                    else:
                        comparison[filename][method] = {'available': False}
                        
            except Exception as e:
                print(f"❌ 比較 {filename} 時發生錯誤: {e}")
        
        return comparison
    
    def _generate_recommendations(self, performance_summary):
        """生成改進建議"""
        recommendations = []
        
        # 分析整體性能
        all_correlations = []
        best_method_counts = {}
        
        for filename, methods in performance_summary.items():
            available_methods = {k: v for k, v in methods.items() if v.get('available', False)}
            
            if available_methods:
                best_method = max(available_methods.keys(), 
                                key=lambda k: available_methods[k]['correlation'])
                best_method_counts[best_method] = best_method_counts.get(best_method, 0) + 1
                
                for method, result in available_methods.items():
                    all_correlations.append(result['correlation'])
        
        avg_correlation = np.mean(all_correlations) if all_correlations else 0
        
        # 生成具體建議
        if avg_correlation < 0.2:
            recommendations.append("🔧 建議開發更精確的物理模型，考慮器件特定效應")
            recommendations.append("📊 建議收集更多實驗數據以改善模型訓練")
            
        if avg_correlation < 0.1:
            recommendations.append("⚠️  當前模型與實驗數據相關性較低，建議重新檢查實驗設置")
            
        if best_method_counts:
            best_overall = max(best_method_counts.keys(), key=lambda k: best_method_counts[k])
            recommendations.append(f"🏆 推薦使用 {best_overall} 方法（在 {best_method_counts[best_overall]} 個檔案中表現最佳）")
        
        recommendations.append("🔬 建議進行溫度相關性研究以改善模型準確性")
        recommendations.append("🎯 建議實施實時參數優化系統")
        
        return recommendations
    
    def generate_comprehensive_report(self, data_quality, strategies, pipeline_results, final_results):
        """
        生成綜合分析報告
        """
        report_path = self.results_dir / "comprehensive_analysis_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 約瑟夫森結數據智能分析綜合報告\n\n")
            f.write(f"**生成時間**: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n\n")
            
            # 數據品質評估
            f.write("## 1. 數據品質評估\n\n")
            quality_scores = [q.get('quality_score', 0) for q in data_quality.values() if 'quality_score' in q]
            if quality_scores:
                f.write(f"- **平均品質評分**: {np.mean(quality_scores):.1f}/100\n")
                f.write(f"- **最高品質評分**: {np.max(quality_scores):.1f}/100\n")
                f.write(f"- **最低品質評分**: {np.min(quality_scores):.1f}/100\n\n")
                
                f.write("### 各檔案品質評分:\n")
                for filename, quality in data_quality.items():
                    if 'quality_score' in quality:
                        f.write(f"- {filename}: {quality['quality_score']:.1f}/100\n")
                f.write("\n")
            
            # 分析策略
            f.write("## 2. 分析策略選擇\n\n")
            strategy_counts = {}
            for strategy in strategies.values():
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            
            for strategy, count in strategy_counts.items():
                f.write(f"- **{strategy}**: {count} 個檔案\n")
            f.write("\n")
            
            # 分析結果
            f.write("## 3. 分析結果摘要\n\n")
            performance = final_results.get('performance_summary', {})
            
            if performance:
                all_basic_corr = []
                all_advanced_corr = []
                
                for filename, methods in performance.items():
                    if methods.get('basic', {}).get('available', False):
                        all_basic_corr.append(methods['basic']['correlation'])
                    if methods.get('advanced', {}).get('available', False):
                        all_advanced_corr.append(methods['advanced']['correlation'])
                
                if all_basic_corr:
                    f.write(f"- **基礎模型平均相關係數**: {np.mean(all_basic_corr):.4f}\n")
                if all_advanced_corr:
                    f.write(f"- **進階模型平均相關係數**: {np.mean(all_advanced_corr):.4f}\n")
                
                if all_basic_corr and all_advanced_corr:
                    improvement = np.mean(all_advanced_corr) - np.mean(all_basic_corr)
                    f.write(f"- **進階模型改進度**: {improvement:.4f}\n")
                f.write("\n")
            
            # 建議
            f.write("## 4. 改進建議\n\n")
            recommendations = final_results.get('recommendations', [])
            for i, rec in enumerate(recommendations, 1):
                f.write(f"{i}. {rec}\n")
            f.write("\n")
            
            # 技術細節
            f.write("## 5. 技術實施細節\n\n")
            f.write("### 已實施的方法:\n")
            f.write("- ✅ 基礎約瑟夫森結模擬\n")
            f.write("- ✅ 參數優化算法\n")
            f.write("- ✅ 機器學習特徵提取\n")
            f.write("- ✅ 進階物理模型（溫度效應、非線性項）\n")
            f.write("- ✅ 智能分析流程\n")
            f.write("- ✅ 綜合性能評估\n\n")
            
            f.write("### 下一步發展方向:\n")
            f.write("- 🔄 實時參數調整系統\n")
            f.write("- 🧠 深度學習模型\n")
            f.write("- 🌡️ 多溫度點建模\n")
            f.write("- 📐 器件幾何優化\n")
            f.write("- 🔗 多結器件建模\n")
        
        print(f"✅ 綜合報告已保存至: {report_path}")
        return report_path

def main():
    """
    主執行函數
    """
    print("🧠 === 智能約瑟夫森結數據分析系統 ===\n")
    
    # 創建協調器
    coordinator = IntelligentAnalysisCoordinator()
    
    # 第一步：評估數據品質
    data_quality = coordinator.assess_data_quality()
    
    # 第二步：選擇分析策略
    strategies = coordinator.select_optimal_analysis_strategy(data_quality)
    
    # 第三步：執行分析管道
    pipeline_results = coordinator.execute_analysis_pipeline(strategies)
    
    # 第四步：評估最終結果
    final_results = coordinator.evaluate_final_results()
    
    # 第五步：生成綜合報告
    report_path = coordinator.generate_comprehensive_report(
        data_quality, strategies, pipeline_results, final_results
    )
    
    print(f"\n🎉 智能分析完成！")
    print(f"📊 綜合報告: {report_path}")
    print(f"📈 建議查看結果目錄: {coordinator.results_dir}")
    
    # 顯示關鍵結果摘要
    print(f"\n📋 關鍵結果摘要:")
    recommendations = final_results.get('recommendations', [])
    for i, rec in enumerate(recommendations[:3], 1):
        print(f"   {i}. {rec}")
    
    if len(recommendations) > 3:
        print(f"   ... 更多建議請查看完整報告")

if __name__ == "__main__":
    main()
