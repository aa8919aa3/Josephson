#!/usr/bin/env python3
"""
進階機器學習優化系統
使用深度學習和特徵工程來改進約瑟夫森結模擬參數
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from scipy.optimize import differential_evolution
from scipy.stats import pearsonr
import json
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib with English fonts only
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')

class AdvancedMLOptimizer:
    """
    進階機器學習優化器，用於約瑟夫森結參數優化
    """
    
    def __init__(self):
        self.feature_scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()
        self.models = {}
        self.feature_importance = {}
        
    def extract_advanced_features(self, y_field, Ic):
        """
        提取進階特徵用於機器學習
        """
        features = {}
        
        # 基本統計特徵
        features['mean'] = np.mean(Ic)
        features['std'] = np.std(Ic)
        features['cv'] = features['std'] / features['mean']
        features['skewness'] = pd.Series(Ic).skew()
        features['kurtosis'] = pd.Series(Ic).kurtosis()
        
        # 磁場相關特徵
        features['field_range'] = np.ptp(y_field)
        features['field_mean'] = np.mean(y_field)
        features['field_std'] = np.std(y_field)
        
        # 頻域特徵
        fft = np.fft.fft(Ic)
        freq = np.fft.fftfreq(len(Ic))
        power_spectrum = np.abs(fft)**2
        
        # 主要頻率
        dominant_freq_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
        features['dominant_frequency'] = freq[dominant_freq_idx]
        features['dominant_power'] = power_spectrum[dominant_freq_idx]
        
        # 週期性特徵
        autocorr = np.correlate(Ic - np.mean(Ic), Ic - np.mean(Ic), mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0]
        
        # 找到第一個局部最大值作為週期指標
        local_maxima = []
        for i in range(1, min(len(autocorr)-1, 50)):
            if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                local_maxima.append((i, autocorr[i]))
        
        if local_maxima:
            features['primary_period'] = local_maxima[0][0]
            features['primary_period_strength'] = local_maxima[0][1]
        else:
            features['primary_period'] = 0
            features['primary_period_strength'] = 0
        
        # 梯度特徵
        gradient = np.gradient(Ic)
        features['gradient_mean'] = np.mean(gradient)
        features['gradient_std'] = np.std(gradient)
        features['gradient_max'] = np.max(np.abs(gradient))
        
        # 局部變異特徵
        local_variations = []
        window_size = min(10, len(Ic) // 10)
        for i in range(0, len(Ic) - window_size, window_size):
            window_data = Ic[i:i+window_size]
            local_variations.append(np.std(window_data))
        
        features['local_variation_mean'] = np.mean(local_variations)
        features['local_variation_std'] = np.std(local_variations)
        
        return features
    
    def create_feature_dataset(self, exp_data_dir, sim_data_dir):
        """
        創建特徵數據集
        """
        feature_data = []
        target_correlations = []
        
        print("🔍 提取特徵數據...")
        
        for exp_file in exp_data_dir.glob("*.csv"):
            try:
                # 載入實驗數據
                exp_df = pd.read_csv(exp_file)
                exp_y = exp_df['y_field'].values
                exp_Ic = exp_df['Ic'].values
                
                # 載入對應的模擬數據
                sim_file = sim_data_dir / f"improved_sim_{exp_file.name}"
                if sim_file.exists():
                    sim_df = pd.read_csv(sim_file)
                    sim_Ic = sim_df['Ic'].values
                    
                    # 計算相關係數作為目標
                    correlation, _ = pearsonr(exp_Ic, sim_Ic)
                    
                    # 提取特徵
                    features = self.extract_advanced_features(exp_y, exp_Ic)
                    
                    feature_data.append(features)
                    target_correlations.append(correlation)
                    
                    print(f"✅ 處理完成: {exp_file.name}, 相關係數: {correlation:.4f}")
                    
            except Exception as e:
                print(f"❌ 處理 {exp_file.name} 時發生錯誤: {e}")
        
        return pd.DataFrame(feature_data), np.array(target_correlations)
    
    def train_ml_models(self, features_df, targets):
        """
        訓練多個機器學習模型
        """
        print("\n🤖 訓練機器學習模型...")
        
        # 準備數據
        X = features_df.values
        y = targets
        
        # 標準化特徵
        X_scaled = self.feature_scaler.fit_transform(X)
        
        # 分割數據（由於數據較少，使用較小的測試集）
        if len(X) > 6:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.3, random_state=42
            )
        else:
            X_train, X_test, y_train, y_test = X_scaled, X_scaled, y, y
        
        # 訓練模型
        models_config = {
            'random_forest': {
                'model': RandomForestRegressor(n_estimators=100, random_state=42),
                'params': {}
            },
            'neural_network': {
                'model': MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=1000, random_state=42),
                'params': {}
            }
        }
        
        results = {}
        
        for name, config in models_config.items():
            print(f"   訓練 {name}...")
            
            model = config['model']
            model.fit(X_train, y_train)
            
            # 評估
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)
            train_mse = mean_squared_error(y_train, train_pred)
            test_mse = mean_squared_error(y_test, test_pred)
            
            results[name] = {
                'model': model,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mse': train_mse,
                'test_mse': test_mse
            }
            
            # 特徵重要性（僅對隨機森林）
            if hasattr(model, 'feature_importances_'):
                importance = dict(zip(features_df.columns, model.feature_importances_))
                self.feature_importance[name] = importance
            
            print(f"      訓練 R²: {train_r2:.4f}, 測試 R²: {test_r2:.4f}")
        
        self.models = results
        return results
    
    def optimize_parameters_with_ml(self, exp_file, feature_target=0.5):
        """
        使用機器學習指導參數優化
        """
        print(f"\n🎯 為 {exp_file.name} 優化參數...")
        
        # 載入實驗數據
        exp_df = pd.read_csv(exp_file)
        exp_y = exp_df['y_field'].values
        exp_Ic = exp_df['Ic'].values
        
        # 提取特徵
        features = self.extract_advanced_features(exp_y, exp_Ic)
        feature_vector = np.array([features[col] for col in features.keys()]).reshape(1, -1)
        feature_vector_scaled = self.feature_scaler.transform(feature_vector)
        
        # 使用最佳模型預測目標相關係數
        best_model_name = max(self.models.keys(), 
                             key=lambda x: self.models[x]['test_r2'])
        best_model = self.models[best_model_name]['model']
        
        predicted_correlation = best_model.predict(feature_vector_scaled)[0]
        print(f"   預測相關係數: {predicted_correlation:.4f}")
        
        # 如果預測相關係數低於目標，使用差分進化算法優化
        if predicted_correlation < feature_target:
            print("   開始參數優化...")
            
            def objective_function(params):
                """優化目標函數"""
                Ic_base, field_scale, phase_offset, asymmetry, noise_level = params
                
                try:
                    # 生成模擬數據
                    flux_quantum = 2.067e-15
                    phi_ext = exp_y * field_scale
                    normalized_flux = np.pi * phi_ext / flux_quantum
                    
                    with np.errstate(divide='ignore', invalid='ignore'):
                        sinc_term = np.sin(normalized_flux + phase_offset) / (normalized_flux + 1e-10)
                        sinc_term = np.where(np.abs(normalized_flux) < 1e-10, 1.0, sinc_term)
                    
                    Ic_pattern = np.abs(sinc_term) * (1 + asymmetry * np.cos(2 * normalized_flux))
                    Ic_sim = Ic_base * Ic_pattern
                    
                    # 添加噪聲
                    noise = np.random.normal(0, noise_level, len(Ic_sim))
                    Ic_sim += noise
                    
                    # 計算相關係數
                    correlation, _ = pearsonr(exp_Ic, Ic_sim)
                    
                    return -correlation if not np.isnan(correlation) else -1e-6
                    
                except:
                    return 1.0  # 懲罰無效參數
            
            # 定義參數邊界
            bounds = [
                (np.mean(exp_Ic) * 0.1, np.mean(exp_Ic) * 3.0),  # Ic_base
                (1e-10, 1e-5),                                    # field_scale
                (0, 2*np.pi),                                     # phase_offset
                (-0.5, 0.5),                                      # asymmetry
                (np.std(exp_Ic) * 0.01, np.std(exp_Ic) * 0.5)   # noise_level
            ]
            
            # 執行優化
            result = differential_evolution(
                objective_function,
                bounds,
                maxiter=100,
                popsize=15,
                seed=42
            )
            
            optimized_params = {
                'Ic_base': result.x[0],
                'field_scale': result.x[1],
                'phase_offset': result.x[2],
                'asymmetry': result.x[3],
                'noise_level': result.x[4],
                'predicted_correlation': -result.fun
            }
            
            print(f"   優化後預期相關係數: {-result.fun:.4f}")
            
        else:
            print("   預測相關係數已足夠，跳過優化")
            optimized_params = None
        
        return optimized_params
    
    def generate_enhanced_report(self, results_dir):
        """
        生成增強分析報告
        """
        report_path = results_dir / "ml_optimization_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 機器學習優化分析報告\n\n")
            f.write(f"生成時間: 2025年6月3日\n\n")
            
            f.write("## 模型性能\n\n")
            for name, results in self.models.items():
                f.write(f"### {name.title()}\n")
                f.write(f"- 訓練 R²: {results['train_r2']:.4f}\n")
                f.write(f"- 測試 R²: {results['test_r2']:.4f}\n")
                f.write(f"- 訓練 MSE: {results['train_mse']:.6f}\n")
                f.write(f"- 測試 MSE: {results['test_mse']:.6f}\n\n")
            
            f.write("## 特徵重要性\n\n")
            for model_name, importance in self.feature_importance.items():
                f.write(f"### {model_name.title()}\n")
                sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                for feature, score in sorted_features[:10]:
                    f.write(f"- {feature}: {score:.4f}\n")
                f.write("\n")
            
            f.write("## 建議\n\n")
            f.write("1. **特徵工程**: 基於重要性分析，重點關注高權重特徵\n")
            f.write("2. **模型選擇**: 選擇測試集表現最佳的模型進行預測\n")
            f.write("3. **參數優化**: 對低相關係數樣品進行重點優化\n")
            f.write("4. **數據擴充**: 考慮收集更多實驗數據以改善模型性能\n")
        
        print(f"✅ 報告已保存至: {report_path}")

def main():
    """
    主執行函數
    """
    print("=== 進階機器學習優化系統 ===\n")
    
    # 設定路徑
    base_dir = Path("/Users/albert-mac/Code/GitHub/Josephson")
    exp_data_dir = base_dir / "data" / "experimental"
    sim_data_dir = base_dir / "data" / "simulated"
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    # 創建優化器
    optimizer = AdvancedMLOptimizer()
    
    # 創建特徵數據集
    features_df, targets = optimizer.create_feature_dataset(exp_data_dir, sim_data_dir)
    
    print(f"\n📊 特徵數據集摘要:")
    print(f"   樣本數量: {len(features_df)}")
    print(f"   特徵數量: {len(features_df.columns)}")
    print(f"   目標範圍: {targets.min():.4f} 到 {targets.max():.4f}")
    
    # 訓練模型
    model_results = optimizer.train_ml_models(features_df, targets)
    
    # 優化參數
    optimization_results = {}
    for exp_file in exp_data_dir.glob("*.csv"):
        if exp_file.name in ['317Ic.csv', '335Ic.csv', '337Ic.csv']:  # 選擇幾個樣本進行優化
            opt_params = optimizer.optimize_parameters_with_ml(exp_file, feature_target=0.3)
            if opt_params:
                optimization_results[exp_file.name] = opt_params
    
    # 保存優化結果
    if optimization_results:
        opt_file = results_dir / "ml_optimized_parameters.json"
        with open(opt_file, 'w', encoding='utf-8') as f:
            json.dump(optimization_results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 優化參數已保存至: {opt_file}")
    
    # 生成報告
    optimizer.generate_enhanced_report(results_dir)
    
    print("\n✅ 機器學習優化完成！")

if __name__ == "__main__":
    main()
