#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Current-Phase Relation Heatmap 演示腳本
專門展示Josephson結的電流-相位關係熱力圖

Author: AI Assistant
Date: 2025年6月4日
"""

import numpy as np
import matplotlib.pyplot as plt
from Is_Sim import JosephsonCurrentSimulator
import warnings
warnings.filterwarnings('ignore')

def main():
    """主演示函數"""
    print("=" * 60)
    print("Current-Phase Relation Heatmap 演示")
    print("=" * 60)
    
    # 創建模擬器
    simulator = JosephsonCurrentSimulator()
    
    # 設置參數 - 更密集的採樣以獲得更好的熱力圖效果
    transparency_values = np.logspace(-2, -0.01, 20)  # 從0.01到0.98，20個點
    phase_range = (-2*np.pi, 2*np.pi)  # 4π的範圍
    num_points = 1001  # 更多的相位點
    temperature = 0.05  # 更低的溫度以獲得更清晰的特徵
    
    print(f"透明度範圍: {transparency_values[0]:.3f} - {transparency_values[-1]:.3f}")
    print(f"透明度點數: {len(transparency_values)}")
    print(f"相位範圍: {phase_range[0]/np.pi:.1f}π to {phase_range[1]/np.pi:.1f}π")
    print(f"相位點數: {num_points}")
    print(f"溫度: {temperature} K")
    
    # 1. 計算Current-Phase Relation
    print("\n1. 計算Current-Phase Relation...")
    phase_results = simulator.calculate_current_phase_relation(
        transparency_values=transparency_values,
        phase_range=phase_range,
        num_points=num_points,
        temperature=temperature,
        models=['AB', 'KO', 'interpolation', 'scattering']
    )
    
    # 2. 生成標準熱力圖
    print("2. 生成Current-Phase Heatmap...")
    simulator.plot_current_phase_heatmap(
        phase_results, 
        'demo_current_phase_heatmap.png'
    )
    
    # 3. 生成進階分析圖
    print("3. 生成進階相位分析圖...")
    simulator.plot_advanced_phase_analysis(
        phase_results, 
        'demo_advanced_phase_analysis.png'
    )
    
    # 4. 創建自定義高解析度熱力圖
    print("4. 生成高解析度單模型熱力圖...")
    create_high_resolution_heatmap(phase_results, 'interpolation')
    
    # 5. 創建互動式分析
    print("5. 生成互動式分析圖...")
    create_interactive_analysis(phase_results)
    
    print("\n=" * 60)
    print("演示完成！生成的文件:")
    print("  - demo_current_phase_heatmap.png (四模型熱力圖)")
    print("  - demo_advanced_phase_analysis.png (進階3D分析)")
    print("  - demo_high_res_interpolation_heatmap.png (高解析度熱力圖)")
    print("  - demo_interactive_analysis.png (互動式分析)")
    print("=" * 60)

def create_high_resolution_heatmap(phase_results, model_name='interpolation'):
    """創建高解析度單模型熱力圖"""
    
    if model_name not in phase_results['models']:
        print(f"模型 {model_name} 不存在")
        return
    
    phase = phase_results['phase']
    transparency_values = phase_results['parameters']['transparency_values']
    model_data = phase_results['models'][model_name]
    
    # 準備數據
    phase_mesh, T_mesh = np.meshgrid(phase, transparency_values)
    current_matrix = np.zeros((len(transparency_values), len(phase)))
    
    for i, T in enumerate(transparency_values):
        T_key = f'T_{T:.3f}'
        if T_key in model_data:
            current_matrix[i, :] = model_data[T_key]['current_phase']
    
    # 創建大型高解析度圖
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 左圖：熱力圖
    im1 = ax1.imshow(current_matrix, 
                     extent=[phase[0], phase[-1], 
                            transparency_values[0], transparency_values[-1]],
                     aspect='auto', 
                     origin='lower',
                     cmap='RdYlBu_r',
                     interpolation='bilinear')
    
    # 添加詳細等高線
    contours = ax1.contour(phase_mesh, T_mesh, current_matrix, 
                          levels=25, colors='black', alpha=0.6, linewidths=0.8)
    ax1.clabel(contours, inline=True, fontsize=10, fmt='%.3f')
    
    ax1.set_xlabel('相位 φ (弧度)', fontsize=14)
    ax1.set_ylabel('透明度 T', fontsize=14)
    ax1.set_title(f'{model_name.upper()} 模型 - 高解析度Current-Phase Heatmap', 
                 fontsize=16, fontweight='bold')
    
    # 設置x軸刻度
    phase_ticks = np.arange(-2*np.pi, 2.5*np.pi, np.pi/2)
    phase_labels = []
    for tick in phase_ticks:
        if tick == 0:
            phase_labels.append('0')
        elif abs(tick - np.pi) < 1e-10:
            phase_labels.append('π')
        elif abs(tick + np.pi) < 1e-10:
            phase_labels.append('-π')
        elif abs(tick - 2*np.pi) < 1e-10:
            phase_labels.append('2π')
        elif abs(tick + 2*np.pi) < 1e-10:
            phase_labels.append('-2π')
        else:
            phase_labels.append(f'{tick/np.pi:.1f}π')
    
    ax1.set_xticks(phase_ticks)
    ax1.set_xticklabels(phase_labels)
    ax1.grid(True, alpha=0.3)
    
    # 顏色條
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('歸一化電流 I/I_c0', fontsize=12)
    
    # 右圖：3D表面圖
    ax2 = fig.add_subplot(122, projection='3d')
    surf = ax2.plot_surface(phase_mesh, T_mesh, current_matrix, 
                           cmap='viridis', alpha=0.9, 
                           linewidth=0, antialiased=True)
    
    ax2.set_xlabel('相位 φ (弧度)', fontsize=12)
    ax2.set_ylabel('透明度 T', fontsize=12)
    ax2.set_zlabel('電流 I/I_c0', fontsize=12)
    ax2.set_title(f'{model_name.upper()} - 3D視圖', fontsize=14)
    
    # 添加顏色條
    fig.colorbar(surf, ax=ax2, shrink=0.6)
    
    plt.tight_layout()
    plt.savefig('demo_high_res_interpolation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"高解析度{model_name}模型熱力圖已保存")

def create_interactive_analysis(phase_results):
    """創建互動式分析圖"""
    
    phase = phase_results['phase']
    transparency_values = phase_results['parameters']['transparency_values']
    
    # 創建多面板分析圖
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    models = ['AB', 'KO', 'interpolation', 'scattering']
    selected_phases = [0, np.pi/2, np.pi, 3*np.pi/2]
    phase_labels = ['φ=0', 'φ=π/2', 'φ=π', 'φ=3π/2']
    
    # 上排：不同相位下的透明度依賴性
    for i, (selected_phase, phase_label) in enumerate(zip(selected_phases[:3], phase_labels[:3])):
        ax = axes[0, i]
        
        # 找到最接近選定相位的索引
        phase_idx = np.argmin(np.abs(phase - selected_phase))
        
        for j, model_name in enumerate(models):
            if model_name not in phase_results['models']:
                continue
            
            model_data = phase_results['models'][model_name]
            current_at_phase = []
            
            for T in transparency_values:
                T_key = f'T_{T:.3f}'
                if T_key in model_data:
                    current_at_phase.append(model_data[T_key]['current_phase'][phase_idx])
                else:
                    current_at_phase.append(0)
            
            ax.plot(transparency_values, current_at_phase, 'o-', 
                   linewidth=2, markersize=4, label=model_name, alpha=0.8)
        
        ax.set_xlabel('透明度 T')
        ax.set_ylabel('歸一化電流')
        ax.set_title(f'電流-透明度關係 @ {phase_label}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 下排：特定透明度下的相位依賴性
    selected_transparencies = [0.1, 0.5, 0.9]
    T_labels = ['T=0.1', 'T=0.5', 'T=0.9']
    
    for i, (selected_T, T_label) in enumerate(zip(selected_transparencies, T_labels)):
        ax = axes[1, i]
        
        # 找到最接近的透明度
        T_idx = np.argmin(np.abs(np.array(transparency_values) - selected_T))
        actual_T = transparency_values[T_idx]
        
        for model_name in models:
            if model_name not in phase_results['models']:
                continue
            
            model_data = phase_results['models'][model_name]
            T_key = f'T_{actual_T:.3f}'
            
            if T_key in model_data:
                current = model_data[T_key]['current_phase']
                ax.plot(phase, current, linewidth=2, label=model_name, alpha=0.8)
        
        ax.set_xlabel('相位 φ (弧度)')
        ax.set_ylabel('歸一化電流')
        ax.set_title(f'電流-相位關係 @ T={actual_T:.2f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 設置x軸刻度
        phase_ticks = np.arange(-2*np.pi, 2.5*np.pi, np.pi)
        phase_labels_x = [f'{int(tick/np.pi)}π' if tick != 0 else '0' for tick in phase_ticks]
        ax.set_xticks(phase_ticks)
        ax.set_xticklabels(phase_labels_x)
    
    plt.tight_layout()
    plt.savefig('demo_interactive_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("互動式分析圖已保存")

if __name__ == "__main__":
    main()
