#!/usr/bin/env python3
"""
優化模擬參數生成器
根據實驗數據特性優化模擬參數以提高相關性
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from josephson_analysis.utils.lmfit_tools import curve_fit_compatible
from scipy.stats import pearsonr
import json

def analyze_experimental_data():
    """分析實驗數據特性以指導模擬參數設置"""
    exp_data_dir = Path("/Users/albert-mac/Code/GitHub/Josephson/data/experimental")
    analysis_results = {}
    
    print("分析實驗數據特性...")
    
    for csv_file in exp_data_dir.glob("*.csv"):
        try:
            data = pd.read_csv(csv_file)
            if 'y_field' in data.columns and 'Ic' in data.columns:
                y_field = np.array(data['y_field'].values, dtype=float)
                Ic = np.array(data['Ic'].values, dtype=float)
                
                # 基本統計
                stats = {
                    'mean_Ic': float(np.mean(Ic)),
                    'std_Ic': float(np.std(Ic)),
                    'max_Ic': float(np.max(Ic)),
                    'min_Ic': float(np.min(Ic)),
                    'y_field_range': [float(np.min(y_field)), float(np.max(y_field))],
                    'n_points': len(Ic),
                    'dynamic_range': float(np.max(Ic) / np.mean(Ic)),
                    'noise_estimate': float(np.std(np.diff(Ic)) / np.sqrt(2))
                }
                
                # 檢測週期性模式
                try:
                    # 簡單的週期檢測（基於FFT）
                    Ic_array = np.array(Ic, dtype=float)
                    fft = np.fft.fft(Ic_array - np.mean(Ic_array))
                    freqs = np.fft.fftfreq(len(Ic_array))
                    dominant_freq_idx = np.argmax(np.abs(fft[1:len(fft)//2])) + 1
                    dominant_period = 1.0 / freqs[dominant_freq_idx] if freqs[dominant_freq_idx] != 0 else 0
                    stats['estimated_period'] = float(dominant_period)
                except:
                    stats['estimated_period'] = 0
                
                analysis_results[csv_file.name] = stats
                print(f"  {csv_file.name}: 平均電流 {stats['mean_Ic']:.2e} A, 標準差 {stats['std_Ic']:.2e} A")
                
        except Exception as e:
            print(f"  無法分析 {csv_file.name}: {e}")
    
    return analysis_results

def generate_optimized_parameters(exp_analysis):
    """根據實驗數據分析結果生成優化的模擬參數"""
    optimized_params = {}
    
    for filename, stats in exp_analysis.items():
        # 根據實驗數據特性調整參數
        mean_Ic = stats['mean_Ic']
        std_Ic = stats['std_Ic']
        noise_est = stats['noise_estimate']
        
        # 基礎電流設定為實驗平均值的合理倍數
        Ic_base = mean_Ic * np.random.uniform(0.8, 1.5)
        
        # 噪聲水平基於實驗數據的噪聲估計
        noise_level = max(noise_est * np.random.uniform(0.5, 2.0), std_Ic * 0.1)
        
        # 相位偏移和不對稱性參數
        phase_offset = np.random.uniform(-0.3, 0.3)
        asymmetry = np.random.uniform(-0.05, 0.05)
        
        # 磁場轉換係數調整（根據y_field範圍和估計週期）
        y_field_range = stats['y_field_range']
        field_span = y_field_range[1] - y_field_range[0]
        
        # 調整轉換係數使得在磁場範圍內有合理的週期數
        target_periods = np.random.uniform(1.5, 4.0)  # 目標週期數
        field_to_flux_factor = target_periods * 2.067e-15 / field_span  # 磁通量子
        
        optimized_params[filename] = {
            'Ic_base': float(Ic_base),
            'phase_offset': float(phase_offset),
            'asymmetry': float(asymmetry),
            'noise_level': float(noise_level),
            'field_to_flux_factor': float(field_to_flux_factor),
            'target_mean': float(mean_Ic),
            'target_std': float(std_Ic)
        }
    
    return optimized_params

def josephson_pattern_advanced(y_field, Ic_base, phase_offset, asymmetry, field_to_flux_factor, 
                             harmonic_amp=0.0, harmonic_phase=0.0):
    """
    改進的約瑟夫森結模式，包含諧波成分
    """
    flux_quantum = 2.067e-15  # 磁通量子
    
    # 將 y_field 轉換為磁通
    phi_ext = y_field * field_to_flux_factor
    normalized_flux = np.pi * phi_ext / flux_quantum
    
    # 基本 sinc 模式
    with np.errstate(divide='ignore', invalid='ignore'):
        sinc_term = np.sin(normalized_flux + phase_offset) / normalized_flux
        sinc_term = np.where(np.abs(normalized_flux) < 1e-10, 1.0, sinc_term)
    
    # 添加不對稱性和諧波成分
    pattern = np.abs(sinc_term) * (1 + asymmetry * np.cos(2 * normalized_flux))
    
    # 添加高階諧波（改善擬合）
    if harmonic_amp > 0:
        pattern += harmonic_amp * np.cos(4 * normalized_flux + harmonic_phase)
    
    Ic = Ic_base * pattern
    
    return Ic

def generate_improved_simulated_data(filename, params, exp_info):
    """生成改進的模擬數據"""
    y_field_range = exp_info['y_field_range']
    n_points = exp_info['n_points']
    
    # 生成磁場值
    y_field = np.linspace(y_field_range[0], y_field_range[1], n_points)
    
    # 生成理想約瑟夫森模式
    Ic_ideal = josephson_pattern_advanced(
        y_field, 
        params['Ic_base'],
        params['phase_offset'],
        params['asymmetry'],
        params['field_to_flux_factor']
    )
    
    # 添加噪聲
    noise = np.random.normal(0, params['noise_level'], n_points)
    Ic_measured = Ic_ideal + noise
    
    # 確保電流值為正且在合理範圍內
    Ic_measured = np.maximum(Ic_measured, 0.1 * params['Ic_base'])
    
    # 後處理：調整使平均值接近目標
    current_mean = np.mean(Ic_measured)
    target_mean = params['target_mean']
    scaling_factor = target_mean / current_mean
    Ic_measured *= scaling_factor
    
    # 創建數據框
    data = pd.DataFrame({
        'y_field': y_field,
        'Ic': Ic_measured
    })
    
    return data

def evaluate_correlation(exp_data, sim_data):
    """評估模擬數據與實驗數據的相關性"""
    if len(exp_data) != len(sim_data):
        min_len = min(len(exp_data), len(sim_data))
        exp_data = exp_data[:min_len]
        sim_data = sim_data[:min_len]
    
    correlation, p_value = pearsonr(exp_data, sim_data)
    return correlation, p_value

def main():
    """主要執行函數"""
    print("=== 約瑟夫森結模擬參數優化器 ===\n")
    
    # 分析實驗數據
    exp_analysis = analyze_experimental_data()
    
    # 生成優化參數
    print("\n生成優化參數...")
    optimized_params = generate_optimized_parameters(exp_analysis)
    
    # 保存參數配置
    params_file = Path("/Users/albert-mac/Code/GitHub/Josephson/data/simulated/optimized_parameters.json")
    with open(params_file, 'w', encoding='utf-8') as f:
        json.dump(optimized_params, f, indent=2, ensure_ascii=False)
    print(f"優化參數已保存到: {params_file}")
    
    # 生成改進的模擬數據並評估
    exp_data_dir = Path("/Users/albert-mac/Code/GitHub/Josephson/data/experimental")
    sim_data_dir = Path("/Users/albert-mac/Code/GitHub/Josephson/data/simulated")
    sim_data_dir.mkdir(exist_ok=True)
    
    correlations = {}
    
    print("\n生成改進的模擬數據...")
    for filename, params in optimized_params.items():
        try:
            # 載入對應的實驗數據
            exp_file = exp_data_dir / filename
            exp_data = pd.read_csv(exp_file)
            
            # 生成改進的模擬數據
            sim_data = generate_improved_simulated_data(filename, params, exp_analysis[filename])
            
            # 保存模擬數據
            sim_file = sim_data_dir / f"sim_{filename}"
            sim_data.to_csv(sim_file, index=False)
            
            # 評估相關性
            correlation, p_value = evaluate_correlation(exp_data['Ic'].values, sim_data['Ic'].values)
            correlations[filename] = {
                'correlation': correlation,
                'p_value': p_value,
                'exp_mean': np.mean(exp_data['Ic']),
                'sim_mean': np.mean(sim_data['Ic'])
            }
            
            print(f"  {filename}: 相關係數 = {correlation:.4f} (p={p_value:.4f})")
            
        except Exception as e:
            print(f"  處理 {filename} 時出錯: {e}")
    
    # 輸出總結
    print(f"\n=== 優化結果總結 ===")
    correlations_values = [c['correlation'] for c in correlations.values()]
    if correlations_values:
        print(f"平均相關係數: {np.mean(correlations_values):.4f}")
        print(f"最高相關係數: {np.max(correlations_values):.4f}")
        print(f"最低相關係數: {np.min(correlations_values):.4f}")
        print(f"相關係數標準差: {np.std(correlations_values):.4f}")
        
        # 找出表現最好的檔案
        best_file = max(correlations.items(), key=lambda x: x[1]['correlation'])
        print(f"表現最佳檔案: {best_file[0]} (相關係數: {best_file[1]['correlation']:.4f})")
    
    # 保存相關性結果
    corr_file = Path("/Users/albert-mac/Code/GitHub/Josephson/results/optimization_correlations.json")
    with open(corr_file, 'w', encoding='utf-8') as f:
        json.dump(correlations, f, indent=2, ensure_ascii=False)
    print(f"\n相關性結果已保存到: {corr_file}")

if __name__ == "__main__":
    main()
