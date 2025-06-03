#!/usr/bin/env python3
"""
溫度效應分析演示

展示新的溫度效應建模功能，包括：
- 溫度相關的臨界電流
- 超導能隙的溫度依賴性
- 熱噪聲效應
- 溫度掃描分析
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import logging

# 添加項目路徑
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from josephson_analysis.models.temperature_effects import (
    TemperatureEffectsModel, 
    create_nb_junction_model,
    create_al_junction_model
)

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 設置 Plotly 為非互動模式
try:
    import plotly.io as pio
    pio.renderers.default = "plotly_mimetype+notebook"
except ImportError:
    pass

def demonstrate_temperature_effects():
    """演示溫度效應功能"""
    print("🌡️  約瑟夫森結溫度效應分析演示")
    print("=" * 50)
    
    # 創建鈮結模型
    nb_model = create_nb_junction_model(Tc=9.2)
    
    # 設置分析參數
    phi_ext = np.linspace(-2, 2, 200)  # 磁通範圍
    temperatures = [1.4, 4.2, 7.0, 8.5]  # 測試溫度點
    
    # 創建圖表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('約瑟夫森結溫度效應分析', fontsize=16, fontweight='bold')
    
    # 1. 不同溫度下的電流響應
    ax1 = axes[0, 0]
    colors = ['blue', 'green', 'orange', 'red']
    
    for i, T in enumerate(temperatures):
        current, info = nb_model.temperature_dependent_response(
            phi_ext, T, include_noise=True
        )
        
        ax1.plot(phi_ext, current*1e6, color=colors[i], 
                label=f'T = {T}K (Ic = {info["critical_current"]*1e6:.1f}μA)',
                linewidth=2, alpha=0.8)
    
    ax1.set_xlabel('外部磁通 (Φ₀)')
    ax1.set_ylabel('電流 (μA)')
    ax1.set_title('溫度相關電流響應')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 臨界電流和能隙的溫度依賴性
    ax2 = axes[0, 1]
    T_sweep = np.linspace(0.1, 9.1, 100)
    Ic_T = nb_model.critical_current_temperature(T_sweep)
    Delta_T = nb_model.energy_gap_temperature(T_sweep)
    
    # 雙軸圖
    ax2_twin = ax2.twinx()
    
    line1 = ax2.plot(T_sweep, Ic_T*1e6, 'b-', linewidth=2, label='Ic(T)')
    line2 = ax2_twin.plot(T_sweep, Delta_T*1000, 'r-', linewidth=2, label='Δ(T)')
    
    ax2.set_xlabel('溫度 (K)')
    ax2.set_ylabel('臨界電流 (μA)', color='blue')
    ax2_twin.set_ylabel('超導能隙 (meV)', color='red')
    ax2.set_title('臨界電流和能隙的溫度依賴性')
    
    # 合併圖例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # 3. 熱噪聲分析
    ax3 = axes[1, 0]
    bandwidths = [1e3, 1e4, 1e5, 1e6]  # 不同頻寬
    
    for bw in bandwidths:
        I_noise = nb_model.thermal_noise_current(T_sweep, bw)
        ax3.semilogy(T_sweep, I_noise*1e9, 
                    label=f'BW = {bw:.0e} Hz', linewidth=2)
    
    ax3.set_xlabel('溫度 (K)')
    ax3.set_ylabel('熱噪聲電流 (nA)')
    ax3.set_title('熱噪聲的溫度和頻寬依賴性')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 相位擴散分析
    ax4 = axes[1, 1]
    I_bias_values = [0.1, 0.3, 0.5, 0.7]  # 歸一化偏置電流
    
    for i_norm in I_bias_values:
        gamma_rates = []
        valid_temps = []
        
        for T in T_sweep:
            if T < nb_model.Tc:
                Ic_T = nb_model.critical_current_temperature(T)
                I_bias = i_norm * Ic_T
                gamma = nb_model.phase_diffusion_rate(T, I_bias)
                if gamma > 0 and np.isfinite(gamma):
                    gamma_rates.append(gamma)
                    valid_temps.append(T)
        
        if gamma_rates:
            ax4.semilogy(valid_temps, gamma_rates, 
                        label=f'I/Ic = {i_norm}', linewidth=2)
    
    ax4.set_xlabel('溫度 (K)')
    ax4.set_ylabel('相位擴散率 (1/s)')
    ax4.set_title('相位擴散率的溫度依賴性')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存圖表
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    output_file = results_dir / "temperature_effects_analysis.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"📊 溫度效應分析圖表已保存：{output_file}")
    
    plt.show()

def compare_materials():
    """比較不同超導材料的溫度效應"""
    print("\\n🔬 不同超導材料溫度效應比較")
    print("=" * 40)
    
    # 創建不同材料的模型
    nb_model = create_nb_junction_model(Tc=9.2)  # 鈮
    al_model = create_al_junction_model(Tc=1.2)  # 鋁
    
    # 溫度範圍
    phi_ext = np.linspace(-1, 1, 100)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('不同超導材料的溫度效應比較', fontsize=16, fontweight='bold')
    
    # 測試溫度（每種材料使用其Tc的不同比例）
    nb_temps = [1.4, 4.2, 7.0]  # 鈮的測試溫度
    al_temps = [0.1, 0.5, 1.0]  # 鋁的測試溫度
    
    # 1. 鈮結響應
    ax1 = axes[0]
    for T in nb_temps:
        current, info = nb_model.temperature_dependent_response(phi_ext, T)
        ax1.plot(phi_ext, current*1e6, 
                label=f'T = {T}K (T/Tc = {T/9.2:.2f})',
                linewidth=2)
    
    ax1.set_xlabel('外部磁通 (Φ₀)')
    ax1.set_ylabel('電流 (μA)')
    ax1.set_title('鈮約瑟夫森結 (Tc = 9.2K)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 鋁結響應
    ax2 = axes[1] 
    for T in al_temps:
        current, info = al_model.temperature_dependent_response(phi_ext, T)
        ax2.plot(phi_ext, current*1e7, 
                label=f'T = {T}K (T/Tc = {T/1.2:.2f})',
                linewidth=2)
    
    ax2.set_xlabel('外部磁通 (Φ₀)')
    ax2.set_ylabel('電流 (100 nA)')
    ax2.set_title('鋁約瑟夫森結 (Tc = 1.2K)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 歸一化比較
    ax3 = axes[2]
    T_nb = np.linspace(0.1, 9.1, 100)
    T_al = np.linspace(0.1, 1.1, 100)
    
    # 歸一化溫度
    t_nb = T_nb / 9.2
    t_al = T_al / 1.2
    
    # 歸一化臨界電流
    Ic_nb_norm = nb_model.critical_current_temperature(T_nb) / nb_model.Ic_0
    Ic_al_norm = al_model.critical_current_temperature(T_al) / al_model.Ic_0
    
    ax3.plot(t_nb, Ic_nb_norm, 'b-', linewidth=2, label='鈮 (Tc = 9.2K)')
    ax3.plot(t_al, Ic_al_norm, 'r-', linewidth=2, label='鋁 (Tc = 1.2K)')
    
    ax3.set_xlabel('歸一化溫度 (T/Tc)')
    ax3.set_ylabel('歸一化臨界電流 (Ic/Ic₀)')
    ax3.set_title('歸一化溫度依賴性比較')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存圖表
    results_dir = project_root / "results"
    output_file = results_dir / "materials_temperature_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"📊 材料比較圖表已保存：{output_file}")
    
    plt.show()

def temperature_sweep_demonstration():
    """演示溫度掃描功能"""
    print("\\n🌡️  溫度掃描分析演示")
    print("=" * 30)
    
    # 創建模型
    model = create_nb_junction_model()
    
    # 設置磁通範圍
    phi_ext = np.linspace(-1.5, 1.5, 150)
    
    # 進行溫度掃描
    print("正在進行溫度掃描分析...")
    sweep_results = model.temperature_sweep_analysis(
        phi_ext, 
        T_range=(1.4, 8.5),
        num_points=20
    )
    
    # 創建3D視覺化
    fig = plt.figure(figsize=(16, 12))
    
    # 3D 電流響應圖
    ax1 = fig.add_subplot(221, projection='3d')
    
    PHI, TEMP = np.meshgrid(phi_ext, sweep_results['temperatures'])
    CURRENT = np.array(sweep_results['current_responses']) * 1e6
    
    surf = ax1.plot_surface(PHI, TEMP, CURRENT, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('外部磁通 (Φ₀)')
    ax1.set_ylabel('溫度 (K)')
    ax1.set_zlabel('電流 (μA)')
    ax1.set_title('溫度-磁通-電流 3D 關係')
    
    # 溫度相關參數演化
    ax2 = fig.add_subplot(222)
    ax2.plot(sweep_results['temperatures'], 
             sweep_results['critical_currents']*1e6, 
             'b-', linewidth=2, label='臨界電流')
    ax2.set_xlabel('溫度 (K)')
    ax2.set_ylabel('臨界電流 (μA)', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    
    ax2_twin = ax2.twinx()
    ax2_twin.plot(sweep_results['temperatures'], 
                  sweep_results['energy_gaps']*1000, 
                  'r-', linewidth=2, label='能隙')
    ax2_twin.set_ylabel('超導能隙 (meV)', color='red')
    ax2_twin.tick_params(axis='y', labelcolor='red')
    ax2.set_title('溫度相關參數演化')
    ax2.grid(True, alpha=0.3)
    
    # 熱圖顯示
    ax3 = fig.add_subplot(223)
    im = ax3.imshow(CURRENT, aspect='auto', cmap='RdYlBu_r', 
                   extent=[phi_ext[0], phi_ext[-1], 
                          sweep_results['temperatures'][0], 
                          sweep_results['temperatures'][-1]])
    ax3.set_xlabel('外部磁通 (Φ₀)')
    ax3.set_ylabel('溫度 (K)')
    ax3.set_title('電流響應熱圖')
    plt.colorbar(im, ax=ax3, label='電流 (μA)')
    
    # 特定磁通下的溫度依賴性
    ax4 = fig.add_subplot(224)
    flux_indices = [75, 90, 105, 120]  # 選擇幾個磁通點
    flux_values = phi_ext[flux_indices]
    
    for i, idx in enumerate(flux_indices):
        current_at_flux = CURRENT[:, idx]
        ax4.plot(sweep_results['temperatures'], current_at_flux,
                'o-', label=f'Φ = {flux_values[i]:.2f}Φ₀', linewidth=2)
    
    ax4.set_xlabel('溫度 (K)')
    ax4.set_ylabel('電流 (μA)')
    ax4.set_title('特定磁通點的溫度依賴性')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存圖表
    results_dir = project_root / "results"
    output_file = results_dir / "temperature_sweep_analysis.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"📊 溫度掃描分析圖表已保存：{output_file}")
    
    plt.show()
    
    # 輸出分析摘要
    print("\\n📋 溫度掃描分析摘要:")
    print(f"  溫度範圍: {sweep_results['temperatures'][0]:.1f}K - {sweep_results['temperatures'][-1]:.1f}K")
    print(f"  臨界電流變化: {np.min(sweep_results['critical_currents'])*1e6:.1f} - {np.max(sweep_results['critical_currents'])*1e6:.1f} μA")
    print(f"  能隙變化: {np.min(sweep_results['energy_gaps'])*1000:.2f} - {np.max(sweep_results['energy_gaps'])*1000:.2f} meV")
    print(f"  熱噪聲範圍: {np.min(sweep_results['thermal_noise'])*1e9:.2f} - {np.max(sweep_results['thermal_noise'])*1e9:.2f} nA")

def main():
    """主執行函數"""
    print("🚀 約瑟夫森結溫度效應分析系統")
    print("=" * 50)
    print("這個演示展示了新增的溫度效應建模功能，包括：")
    print("• BCS 理論的超導能隙溫度依賴性")
    print("• Ambegaokar-Baratoff 關係的臨界電流")
    print("• Nyquist 熱噪聲建模")
    print("• 相位擴散效應")
    print("• 多材料比較分析")
    print("• 溫度掃描功能")
    print()
    
    try:
        # 基礎溫度效應演示
        demonstrate_temperature_effects()
        
        # 材料比較
        compare_materials()
        
        # 溫度掃描演示
        temperature_sweep_demonstration()
        
        print("\\n✅ 溫度效應分析演示完成！")
        print("\\n🎯 下一步建議:")
        print("  1. 將溫度模型集成到主要分析流程")
        print("  2. 添加溫度掃描到實時優化系統")
        print("  3. 開發溫度相關的參數估計算法")
        print("  4. 創建溫度效應的實驗驗證工具")
        
    except Exception as e:
        logger.error(f"演示過程中發生錯誤: {e}")
        raise

if __name__ == "__main__":
    main()
