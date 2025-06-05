#!/usr/bin/env python3
"""
Current-Phase Relation 功能演示腳本
展示如何使用新實現的電流相位關係分析功能

使用方法：
python demo_current_phase.py
"""

import numpy as np
import matplotlib.pyplot as plt
from Is_Sim import JosephsonCurrentSimulator

def demo_current_phase_relation():
    """演示 Current-Phase Relation 功能"""
    print("🔬 Current-Phase Relation 功能演示")
    print("=" * 50)
    
    # 創建模擬器
    simulator = JosephsonCurrentSimulator()
    
    # 設定演示參數
    transparencies = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]  # 選擇3個代表性透明度
    temperature = 0.1  # K
    
    print(f"📊 演示參數:")
    print(f"   透明度: {transparencies}")
    print(f"   溫度: {temperature} K")
    print(f"   理論模型: AB, KO, 內插, 散射")
    
    # 1. 計算電流相位關係
    print("\n🧮 步驟 1: 計算電流相位關係...")
    phase_results = simulator.calculate_current_phase_relation(
        transparency_values=transparencies,
        temperature=temperature
    )
    
    # 顯示關鍵結果
    print(f"✅ 計算完成!")
    print(f"   相位點數: {len(phase_results['phase'])}")
    print(f"   模型數量: {len(phase_results['models'])}")
    
    # 2. 諧波分析
    print("\n🎵 步驟 2: 諧波分析...")
    harmonic_results = simulator.analyze_harmonics(phase_results, max_harmonics=3)
    
    # 顯示諧波分析結果
    print("✅ 諧波分析完成!")
    for model_name in ['AB', 'KO', 'interpolation', 'scattering']:
        if model_name in harmonic_results['models']:
            model_data = harmonic_results['models'][model_name]
            if model_data:
                sample_key = list(model_data.keys())[0]
                thd = model_data[sample_key]['total_harmonic_distortion']
                print(f"   {model_name:12} THD: {thd:.4f}")
    
    # 3. 生成可視化
    print("\n📈 步驟 3: 生成可視化圖表...")
    
    # Current-Phase Relation 圖
    simulator.plot_current_phase_relation(
        phase_results, 
        'demo_current_phase_relation.png'
    )
    
    # 諧波分析圖
    simulator.plot_harmonic_analysis(
        harmonic_results, 
        'demo_harmonic_analysis.png'
    )
    
    print("✅ 圖表生成完成!")
    print("   📁 demo_current_phase_relation.png")
    print("   📁 demo_harmonic_analysis.png")
    
    # 4. 數據分析展示
    print("\n📋 步驟 4: 關鍵數據分析...")
    
    # 選擇內插模型進行詳細分析
    interp_data = phase_results['models']['interpolation']
    phase = phase_results['phase']
    
    print("\n🔍 內插模型詳細分析:")
    print("   透明度    臨界電流    最大電流    相位@最大電流")
    print("   " + "-" * 50)
    
    for T_key, data in interp_data.items():
        T = data['transparency']
        I_c_0 = data['I_c_0']
        current = data['current_phase']
        max_current = np.max(np.abs(current))
        max_phase_idx = np.argmax(np.abs(current))
        max_phase = phase[max_phase_idx]
        
        print(f"   {T:7.3f}   {I_c_0:8.4f}   {max_current:8.4f}   {max_phase:12.4f}")
    
    # 5. 物理洞察
    print("\n🧠 物理洞察:")
    print("   💡 電流相位關係反映約瑟夫結的量子相干性")
    print("   💡 THD值量化非線性程度，高透明度下更顯著")
    print("   💡 不同理論模型在高透明度時出現差異")
    print("   💡 諧波分析揭示量子散射和多體效應")
    
    return phase_results, harmonic_results

def quick_comparison():
    """快速比較不同透明度的電流相位特性"""
    print("\n🚀 快速比較演示")
    print("=" * 30)
    
    simulator = JosephsonCurrentSimulator()
    
    # 計算不同透明度的關鍵參數
    transparencies = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    temperature = 0.1
    
    print("透明度對電流相位關係的影響:")
    print("透明度   I_c(AB)   I_c(KO)   I_c(內插)   I_c(散射)")
    print("-" * 55)
    
    for T in transparencies:
        I_AB = simulator.critical_current_AB(T, temperature)
        I_KO = simulator.critical_current_KO(T, temperature)
        I_interp = simulator.critical_current_interpolation(T, temperature)
        I_scatter = simulator.critical_current_scattering(T, temperature)
        
        print(f"{T:6.1f}   {I_AB:7.4f}   {I_KO:7.4f}   {I_interp:8.4f}   {I_scatter:8.4f}")

if __name__ == "__main__":
    try:
        # 主演示
        phase_results, harmonic_results = demo_current_phase_relation()
        
        # 快速比較
        quick_comparison()
        
        print("\n🎉 演示完成!")
        print("Current-Phase Relation 功能已成功演示")
        print("\n📁 生成的文件:")
        print("   - demo_current_phase_relation.png")
        print("   - demo_harmonic_analysis.png")
        
    except Exception as e:
        print(f"❌ 演示過程中發生錯誤: {e}")
        print("請檢查程式碼或依賴項")
