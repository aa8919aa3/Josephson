"""
生成約瑟夫森結模擬數據

這個腳本生成與實驗數據格式相同的模擬數據，用於測試和比較分析。
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path


def generate_josephson_data(y_field_range, n_points=153, Ic_base=1e-6, 
                           flux_quantum=2.067833831e-15, noise_level=1e-7,
                           phase_offset=0, asymmetry=0.0):
    """
    生成約瑟夫森結的模擬臨界電流數據
    
    Parameters:
    -----------
    y_field_range : tuple
        磁場範圍 (y_min, y_max)
    n_points : int
        數據點數
    Ic_base : float
        基準臨界電流
    flux_quantum : float
        磁通量子
    noise_level : float
        雜訊水準
    phase_offset : float
        相位偏移
    asymmetry : float
        不對稱性參數
        
    Returns:
    --------
    pd.DataFrame
        包含 y_field 和 Ic 列的數據框
    """
    # 生成磁場值
    y_field = np.linspace(y_field_range[0], y_field_range[1], n_points)
    
    # 將 y_field 轉換為磁通（假設某個轉換係數）
    # 這裡假設 y_field 是正規化的磁場，需要轉換為實際磁通
    field_to_flux_factor = 1e-14  # 轉換係數
    phi_ext = y_field * field_to_flux_factor
    
    # 計算週期性磁通調制的臨界電流
    # 使用標準的 |sin(πΦ/Φ₀)/(πΦ/Φ₀)| 模式
    normalized_flux = np.pi * phi_ext / flux_quantum
    
    # 避免除零錯誤
    with np.errstate(divide='ignore', invalid='ignore'):
        sinc_term = np.sin(normalized_flux + phase_offset) / normalized_flux
        sinc_term = np.where(np.abs(normalized_flux) < 1e-10, 1.0, sinc_term)
    
    # 計算臨界電流（包含不對稱性）
    Ic_ideal = Ic_base * np.abs(sinc_term) * (1 + asymmetry * np.cos(2 * normalized_flux))
    
    # 添加雜訊
    noise = np.random.normal(0, noise_level, n_points)
    Ic_measured = Ic_ideal + noise
    
    # 確保電流值為正
    Ic_measured = np.abs(Ic_measured)
    
    # 創建數據框
    data = pd.DataFrame({
        'y_field': y_field,
        'Ic': Ic_measured
    })
    
    return data


def load_experimental_data():
    """載入實驗數據以獲取實際的磁場範圍和數據點數"""
    exp_data_dir = Path("/Users/albert-mac/Code/GitHub/Josephson/data/experimental")
    exp_data_info = {}
    
    for exp_file in exp_data_dir.glob("*.csv"):
        try:
            data = pd.read_csv(exp_file)
            if len(data.columns) >= 2:
                data.columns = ['y_field', 'Ic']
                exp_data_info[exp_file.name] = {
                    'y_field_range': (data['y_field'].min(), data['y_field'].max()),
                    'n_points': len(data),
                    'Ic_range': (data['Ic'].min(), data['Ic'].max())
                }
                print(f"載入實驗數據: {exp_file.name} ({len(data)} 點)")
        except Exception as e:
            print(f"載入 {exp_file.name} 失敗: {e}")
    
    return exp_data_info


def generate_all_simulated_data():
    """生成所有模擬數據檔案，基於實際實驗數據的磁場範圍和數據點數"""
    
    # 設定輸出目錄
    output_dir = Path("/Users/albert-mac/Code/GitHub/Josephson/data/simulated")
    output_dir.mkdir(exist_ok=True)
    
    # 載入實驗數據信息
    exp_data_info = load_experimental_data()
    
    # 定義不同樣本的物理參數
    samples_config = {
        '317Ic.csv': {
            'Ic_base': 1.5e-5,
            'phase_offset': 0.1,
            'asymmetry': 0.02,
            'noise_level': 1.0e-6
        },
        '335Ic.csv': {
            'Ic_base': 1.2e-5,
            'phase_offset': -0.1,
            'asymmetry': -0.01,
            'noise_level': 8.0e-7
        },
        '336Ic.csv': {
            'Ic_base': 1.3e-5,
            'phase_offset': 0.15,
            'asymmetry': 0.03,
            'noise_level': 9.0e-7
        },
        '337Ic.csv': {
            'Ic_base': 1.1e-5,
            'phase_offset': -0.05,
            'asymmetry': -0.02,
            'noise_level': 7.0e-7
        },
        '338Ic.csv': {
            'Ic_base': 1.4e-5,
            'phase_offset': 0.2,
            'asymmetry': 0.04,
            'noise_level': 1.1e-6
        },
        '341Ic.csv': {
            'Ic_base': 1.6e-5,
            'phase_offset': -0.15,
            'asymmetry': -0.03,
            'noise_level': 1.2e-6
        },
        '346Ic.csv': {
            'Ic_base': 1.0e-5,
            'phase_offset': 0.05,
            'asymmetry': 0.01,
            'noise_level': 6.0e-7
        },
        '352Ic.csv': {
            'Ic_base': 1.7e-5,
            'phase_offset': -0.2,
            'asymmetry': -0.04,
            'noise_level': 1.3e-6
        },
        '435Ic.csv': {
            'Ic_base': 1.8e-5,
            'phase_offset': 0.25,
            'asymmetry': 0.05,
            'noise_level': 1.4e-6
        },
        '439Ic.csv': {
            'Ic_base': 1.9e-5,
            'phase_offset': -0.25,
            'asymmetry': -0.05,
            'noise_level': 1.5e-6
        },
        'kay164Ic-.csv': {
            'Ic_base': 2.0e-6,
            'phase_offset': -0.3,
            'asymmetry': -0.06,
            'noise_level': 2.0e-7
        },
        'kay164Ic+.csv': {
            'Ic_base': 2.2e-6,
            'phase_offset': 0.3,
            'asymmetry': 0.06,
            'noise_level': 2.2e-7
        }
    }
    
    print("正在生成模擬數據...")
    
    for filename, config in samples_config.items():
        if filename in exp_data_info:
            print(f"生成 {filename}...")
            
            # 獲取對應實驗數據的信息
            exp_info = exp_data_info[filename]
            
            # 為每個檔案設定不同的隨機種子，確保可重現性
            np.random.seed(hash(filename) % 2**32)
            
            # 生成數據，使用實際的磁場範圍和數據點數
            data = generate_josephson_data(
                y_field_range=exp_info['y_field_range'],
                n_points=exp_info['n_points'],
                **config
            )
            
            # 保存到檔案
            output_path = output_dir / filename
            data.to_csv(output_path, index=False)
            
            print(f"已保存: {output_path} ({exp_info['n_points']} 點)")
        else:
            print(f"⚠️ 找不到對應的實驗數據: {filename}")
    
    print("\n所有模擬數據生成完成！")
    print(f"檔案保存位置: {output_dir}")


if __name__ == "__main__":
    generate_all_simulated_data()
