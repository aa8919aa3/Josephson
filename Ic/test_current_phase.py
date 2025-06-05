#!/usr/bin/env python3
"""
Current-Phase Relation 功能測試腳本
測試新增的電流相位關係分析功能
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK TC']  # 使用思源黑體
plt.rcParams['axes.unicode_minus'] = False
# 導入主模擬器
from Is_Sim import JosephsonCurrentSimulator

def test_current_phase_relation():
    """測試電流相位關係功能"""
    print("=== Current-Phase Relation 功能測試 ===")
    
    # 創建模擬器
    simulator = JosephsonCurrentSimulator()
    
    # 設定測試參數
    transparency_values = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]  # 測試用較少的透明度值
    temperature = 0.1  # K
    
    print(f"測試透明度值: {transparency_values}")
    print(f"測試溫度: {temperature} K")
    
    # 1. 測試 Current-Phase Relation 計算
    print("\n1. 測試 Current-Phase Relation 計算...")
    try:
        phase_results = simulator.calculate_current_phase_relation(
            transparency_values=transparency_values,
            temperature=temperature
        )
        print("✓ Current-Phase Relation 計算成功")
        
        # 檢查結果結構
        print(f"  - 包含模型數量: {len(phase_results['models'])}")
        print(f"  - 相位點數: {len(phase_results['phase'])}")
        
        # 檢查每個模型的數據
        for model_name in phase_results['models']:
            model_data = phase_results['models'][model_name]
            print(f"  - {model_name} 模型: {len(model_data)} 透明度值")
            
    except Exception as e:
        print(f"✗ Current-Phase Relation 計算失敗: {e}")
        return False
    
    # 2. 測試諧波分析
    print("\n2. 測試諧波分析...")
    try:
        harmonic_results = simulator.analyze_harmonics(phase_results, max_harmonics=3)
        print("✓ 諧波分析成功")
        
        # 檢查諧波結果
        for model_name in harmonic_results['models']:
            model_data = harmonic_results['models'][model_name]
            if model_data:  # 檢查是否有數據
                sample_key = list(model_data.keys())[0]
                sample_data = model_data[sample_key]
                thd = sample_data['total_harmonic_distortion']
                print(f"  - {model_name} 模型 THD: {thd:.4f}")
                
    except Exception as e:
        print(f"✗ 諧波分析失敗: {e}")
        return False
    
    # 3. 測試圖表生成
    print("\n3. 測試圖表生成...")
    try:
        # 測試 Current-Phase Relation 圖
        simulator.plot_current_phase_relation(
            phase_results, 
            'test_current_phase_relation.png'
        )
        print("✓ Current-Phase Relation 圖生成成功")
        
        # 測試諧波分析圖
        simulator.plot_harmonic_analysis(
            harmonic_results, 
            'test_harmonic_analysis.png'
        )
        print("✓ 諧波分析圖生成成功")
        
    except Exception as e:
        print(f"✗ 圖表生成失敗: {e}")
        return False
    
    # 4. 數據驗證
    print("\n4. 數據驗證...")
    try:
        # 檢查 Current-Phase Relation 的基本性質
        sample_model = 'interpolation'
        if sample_model in phase_results['models']:
            model_data = phase_results['models'][sample_model]
            first_T_key = list(model_data.keys())[0]
            sample_data = model_data[first_T_key]
            
            phase = sample_data['phase']
            current = sample_data['current_phase']
            I_c_0 = sample_data['I_c_0']
            
            # 檢查相位範圍
            phase_range = phase[-1] - phase[0]
            expected_range = 2 * np.pi
            if abs(phase_range - expected_range) < 0.1:
                print(f"✓ 相位範圍正確: {phase_range:.3f} ≈ 2π")
            else:
                print(f"✗ 相位範圍錯誤: {phase_range:.3f} ≠ 2π")
                
            # 檢查電流最大值
            max_current = np.max(np.abs(current))
            if abs(max_current - I_c_0) < 0.1 * I_c_0:
                print(f"✓ 電流最大值正確: {max_current:.4f} ≈ {I_c_0:.4f}")
            else:
                print(f"⚠ 電流最大值可能異常: {max_current:.4f} vs {I_c_0:.4f}")
                
        print("✓ 數據驗證完成")
        
    except Exception as e:
        print(f"✗ 數據驗證失敗: {e}")
        return False
    
    print("\n=== 測試完成 ===")
    print("所有 Current-Phase Relation 功能測試通過！")
    print("\n生成的測試文件:")
    print("- test_current_phase_relation.png")
    print("- test_harmonic_analysis.png")
    
    return True

def display_sample_results():
    """顯示一些示例結果"""
    print("\n=== 示例結果展示 ===")
    
    # 創建一個簡單的示例
    simulator = JosephsonCurrentSimulator()
    
    # 計算單一透明度的電流相位關係
    T = 0.5
    temperature = 0.1
    
    # 獲取零磁場臨界電流
    I_c_0_interp = simulator.critical_current_interpolation(T, temperature)
    
    # 生成相位數組
    phase = np.linspace(0, 2*np.pi, 100)
    
    # 基本正弦關係
    current = I_c_0_interp * np.sin(phase)
    
    # 顯示一些關鍵點
    print(f"透明度 T = {T}")
    print(f"臨界電流 I_c(0) = {I_c_0_interp:.4f}")
    print(f"φ = 0: I = {current[0]:.4f}")
    print(f"φ = π/2: I = {current[25]:.4f}")
    print(f"φ = π: I = {current[50]:.4f}")
    print(f"φ = 3π/2: I = {current[75]:.4f}")

if __name__ == "__main__":
    # 運行測試
    success = test_current_phase_relation()
    
    if success:
        display_sample_results()
        sys.exit(0)
    else:
        print("測試失敗！請檢查程式碼。")
        sys.exit(1)
