#!/usr/bin/env python3
"""
改進的模擬數據生成器，解決常數輸入問題
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import json
import warnings

def josephson_pattern_robust(y_field, Ic_base, phase_offset, asymmetry, field_to_flux_factor):
    """
    穩健的約瑟夫森結模式，確保產生變化的數據
    """
    flux_quantum = 2.067e-15
    
    # 將 y_field 轉換為磁通
    phi_ext = y_field * field_to_flux_factor
    normalized_flux = np.pi * phi_ext / flux_quantum
    
    # 確保有足夠的變化 - 調整磁通範圍
    flux_range = np.max(normalized_flux) - np.min(normalized_flux)
    if flux_range < 2 * np.pi:  # 至少一個完整周期
        # 擴展磁通範圍
        center_flux = np.mean(normalized_flux)
        normalized_flux = center_flux + (normalized_flux - center_flux) * (2 * np.pi / flux_range)
    
    # 基本 sinc 模式
    with np.errstate(divide='ignore', invalid='ignore'):
        sinc_term = np.sin(normalized_flux + phase_offset) / (normalized_flux + 1e-10)
        sinc_term = np.where(np.abs(normalized_flux) < 1e-10, 1.0, sinc_term)
    
    # 添加不對稱性和額外變化
    modulation = 1 + asymmetry * np.cos(2 * normalized_flux) + 0.1 * np.sin(4 * normalized_flux)
    pattern = np.abs(sinc_term) * modulation
    
    # 確保有足夠的動態範圍
    pattern = pattern - np.min(pattern) + 0.1  # 避免零值
    pattern = pattern / np.max(pattern)  # 正規化到 [0.1, 1]
    
    Ic = Ic_base * pattern
    
    return Ic

def generate_robust_simulated_data(filename, exp_data_info):
    """
    生成穩健的模擬數據，確保有合理的變化
    """
    y_field_range = exp_data_info['y_field_range']
    n_points = exp_data_info['n_points']
    target_mean = exp_data_info['mean_Ic']
    target_std = exp_data_info['std_Ic']
    
    # 生成磁場值
    y_field = np.linspace(y_field_range[0], y_field_range[1], n_points)
    
    # 基於文件名調整參數
    np.random.seed(hash(filename) % 2**32)  # 確保可重現性
    
    # 調整參數範圍
    Ic_base = target_mean * np.random.uniform(0.8, 1.2)
    phase_offset = np.random.uniform(-np.pi/2, np.pi/2)
    asymmetry = np.random.uniform(-0.3, 0.3)
    
    # 調整磁場轉換係數以確保在磁場範圍內有足夠的變化
    field_span = y_field_range[1] - y_field_range[0]
    periods_in_range = np.random.uniform(2, 6)  # 2-6個周期
    field_to_flux_factor = periods_in_range * 2.067e-15 / field_span
    
    # 生成基本模式
    Ic_pattern = josephson_pattern_robust(y_field, Ic_base, phase_offset, asymmetry, field_to_flux_factor)
    
    # 添加適當的噪聲
    noise_level = target_std * np.random.uniform(0.3, 0.7)
    noise = np.random.normal(0, noise_level, n_points)
    Ic_measured = Ic_pattern + noise
    
    # 確保電流值為正
    Ic_measured = np.maximum(Ic_measured, 0.01 * target_mean)
    
    # 調整到目標統計特性
    current_mean = np.mean(Ic_measured)
    current_std = np.std(Ic_measured)
    
    # 調整平均值
    Ic_measured = Ic_measured * (target_mean / current_mean)
    
    # 調整標準差
    Ic_centered = Ic_measured - np.mean(Ic_measured)
    Ic_measured = np.mean(Ic_measured) + Ic_centered * (target_std / np.std(Ic_centered))
    
    # 再次確保正值
    Ic_measured = np.maximum(Ic_measured, 0.01 * target_mean)
    
    return pd.DataFrame({
        'y_field': y_field,
        'Ic': Ic_measured
    })

def load_experimental_data_info():
    """載入實驗數據信息"""
    exp_data_dir = Path("/Users/albert-mac/Code/GitHub/Josephson/data/experimental")
    exp_info = {}
    
    for csv_file in exp_data_dir.glob("*.csv"):
        try:
            data = pd.read_csv(csv_file)
            if 'y_field' in data.columns and 'Ic' in data.columns:
                exp_info[csv_file.name] = {
                    'y_field_range': [float(data['y_field'].min()), float(data['y_field'].max())],
                    'n_points': len(data),
                    'mean_Ic': float(data['Ic'].mean()),
                    'std_Ic': float(data['Ic'].std()),
                    'max_Ic': float(data['Ic'].max()),
                    'min_Ic': float(data['Ic'].min())
                }
        except Exception as e:
            print(f"警告：無法載入 {csv_file.name}: {e}")
    
    return exp_info

def evaluate_simulation_quality(exp_data, sim_data, filename):
    """評估模擬數據質量"""
    results = {}
    
    # 基本統計比較
    results['exp_mean'] = float(np.mean(exp_data))
    results['sim_mean'] = float(np.mean(sim_data))
    results['exp_std'] = float(np.std(exp_data))
    results['sim_std'] = float(np.std(sim_data))
    results['mean_ratio'] = float(results['sim_mean'] / results['exp_mean'])
    results['std_ratio'] = float(results['sim_std'] / results['exp_std'])
    
    # 檢查數據變化性
    exp_variation = np.std(exp_data) / np.mean(exp_data)
    sim_variation = np.std(sim_data) / np.mean(sim_data)
    results['exp_cv'] = float(exp_variation)
    results['sim_cv'] = float(sim_variation)
    
    # 相關性分析（如果兩個數據都有變化）
    if exp_variation > 1e-10 and sim_variation > 1e-10:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                correlation, p_value = pearsonr(exp_data, sim_data)
                if not np.isnan(correlation):
                    results['correlation'] = float(correlation)
                    results['p_value'] = float(p_value)
                else:
                    results['correlation'] = 0.0
                    results['p_value'] = 1.0
            except:
                results['correlation'] = 0.0
                results['p_value'] = 1.0
    else:
        results['correlation'] = 0.0
        results['p_value'] = 1.0
        if exp_variation <= 1e-10:
            print(f"  警告：{filename} 實驗數據變化太小")
        if sim_variation <= 1e-10:
            print(f"  警告：{filename} 模擬數據變化太小")
    
    return results

def main():
    """主要執行函數"""
    print("=== 改進的約瑟夫森結模擬數據生成器 ===\n")
    
    # 載入實驗數據信息
    exp_info = load_experimental_data_info()
    print(f"載入了 {len(exp_info)} 個實驗數據檔案")
    
    # 設置輸出目錄
    exp_data_dir = Path("/Users/albert-mac/Code/GitHub/Josephson/data/experimental")
    sim_data_dir = Path("/Users/albert-mac/Code/GitHub/Josephson/data/simulated")
    sim_data_dir.mkdir(exist_ok=True)
    
    results = {}
    valid_correlations = []
    
    print("\n生成改進的模擬數據：")
    
    for filename, info in exp_info.items():
        try:
            # 載入實驗數據
            exp_data = pd.read_csv(exp_data_dir / filename)
            
            # 生成改進的模擬數據
            sim_data = generate_robust_simulated_data(filename, info)
            
            # 保存模擬數據
            sim_file = sim_data_dir / f"improved_sim_{filename}"
            sim_data.to_csv(sim_file, index=False)
            
            # 評估質量
            quality = evaluate_simulation_quality(exp_data['Ic'].values, sim_data['Ic'].values, filename)
            results[filename] = quality
            
            if not np.isnan(quality['correlation']):
                valid_correlations.append(quality['correlation'])
            
            print(f"  {filename}:")
            print(f"    相關係數: {quality['correlation']:.4f} (p={quality['p_value']:.4f})")
            print(f"    平均值比率: {quality['mean_ratio']:.3f}, 標準差比率: {quality['std_ratio']:.3f}")
            print(f"    變異係數 - 實驗: {quality['exp_cv']:.4f}, 模擬: {quality['sim_cv']:.4f}")
            
        except Exception as e:
            print(f"  處理 {filename} 時出錯: {e}")
    
    # 總結結果
    print(f"\n=== 改進結果總結 ===")
    if valid_correlations:
        print(f"有效相關係數數量: {len(valid_correlations)}")
        print(f"平均相關係數: {np.mean(valid_correlations):.4f}")
        print(f"最高相關係數: {np.max(valid_correlations):.4f}")
        print(f"最低相關係數: {np.min(valid_correlations):.4f}")
        print(f"相關係數標準差: {np.std(valid_correlations):.4f}")
        
        # 找出表現最好的檔案
        valid_results = {k: v for k, v in results.items() if not np.isnan(v['correlation'])}
        if valid_results:
            best_file = max(valid_results.items(), key=lambda x: x[1]['correlation'])
            print(f"表現最佳檔案: {best_file[0]} (相關係數: {best_file[1]['correlation']:.4f})")
    else:
        print("沒有有效的相關係數計算結果")
    
    # 保存結果
    results_file = Path("/Users/albert-mac/Code/GitHub/Josephson/results/improved_simulation_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n改進結果已保存到: {results_file}")

if __name__ == "__main__":
    main()
