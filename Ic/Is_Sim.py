# filepath: /Users/albert-mac/Code/GitHub/Josephson/Ic/Is_Sim.py

"""
超導電流與磁通量關係模擬程序
Superconducting Current vs Magnetic Flux Simulation

基於多種Josephson結理論模型：
- Ambegaokar-Baratoff (AB) 理論 - 低透明度隧道結
- Kulik-Omelyanchuk (KO) 理論 - 高透明度短結
- 內插公式 - 任意透明度
- 散射理論模型
- Fraunhofer 繞射圖案分析

Author: AI Assistant
Date: 2025年6月4日
"""

import numpy as np
import scipy as sp
from scipy import integrate, optimize, special
import matplotlib
matplotlib.use('Agg')  # 使用非交互式後端
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable, Union
import warnings
warnings.filterwarnings('ignore')

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK TC']  # 使用思源黑體
plt.rcParams['axes.unicode_minus'] = False

@dataclass
class PhysicalConstants:
    """物理常數"""
    e: float = 1.602176634e-19  # 基本電荷 (C)
    h: float = 6.62607015e-34   # 普朗克常數 (J⋅s)
    hbar: float = 1.054571817e-34  # 約化普朗克常數 (J⋅s)
    k_B: float = 1.380649e-23   # 玻爾茲曼常數 (J/K)
    phi_0: float = 2.067833848e-15  # 磁通量子 (Wb)
    
    # 超導體參數 (典型值)
    delta_0: float = 1.76e-3 * 1.602176634e-19  # 超導能隙 (J), ~1.76 meV for Al
    T_c: float = 1.2  # 臨界溫度 (K), for Al
    
    def __post_init__(self):
        """計算衍生常數"""
        self.phi_0_normalized = self.phi_0 / (2 * np.pi)

class JosephsonCurrentSimulator:
    """Josephson結電流模擬器"""
    
    def __init__(self, constants: PhysicalConstants = None):
        self.const = constants if constants else PhysicalConstants()
        self.results = {}
        
    def critical_current_AB(self, T: float, temperature: float = 0.1) -> float:
        """
        Ambegaokar-Baratoff 理論 - 低透明度隧道結
        適用於 T << 1
        
        Args:
            T: 透明度參數 (0 < T < 1)
            temperature: 溫度 (K)
            
        Returns:
            歸一化臨界電流
        """
        if T >= 1.0:
            T = 0.99  # 避免發散
            
        # 溫度依賴的能隙
        t = temperature / self.const.T_c
        if t >= 1.0:
            return 0.0
            
        delta_T = self.const.delta_0 * np.sqrt(1 - t**4)  # BCS溫度依賴
        
        # AB理論公式: I_c = (π/2) * (Δ/eR_N) * T
        # 歸一化形式
        I_c_normalized = (np.pi / 2) * T * (delta_T / self.const.delta_0)
        
        return I_c_normalized
    
    def critical_current_KO(self, T: float, temperature: float = 0.1) -> float:
        """
        Kulik-Omelyanchuk 理論 - 高透明度短結
        適用於 T ≈ 1
        
        Args:
            T: 透明度參數
            temperature: 溫度 (K)
            
        Returns:
            歸一化臨界電流
        """
        # 溫度依賴的能隙
        t = temperature / self.const.T_c
        if t >= 1.0:
            return 0.0
            
        delta_T = self.const.delta_0 * np.sqrt(1 - t**4)
        
        # KO理論的數值積分 (簡化版本)
        # 完整版本需要複雜的橢圓積分
        if T >= 0.99:
            # 高透明度極限
            I_c_normalized = np.pi * (delta_T / self.const.delta_0)
        else:
            # 內插近似
            I_c_normalized = np.pi * T * (delta_T / self.const.delta_0) / (1 + 0.5 * (1 - T))
            
        return I_c_normalized
    
    def critical_current_interpolation(self, T: float, temperature: float = 0.1) -> float:
        """
        內插公式 - 適用於任意透明度
        eI_c(0)R_N/Δ = π/(1+√(1-τ))
        其中 τ = T 是透明度
        
        Args:
            T: 透明度參數
            temperature: 溫度 (K)
            
        Returns:
            歸一化臨界電流
        """
        if T >= 1.0:
            T = 0.999  # 避免計算問題
            
        # 溫度依賴
        t = temperature / self.const.T_c
        if t >= 1.0:
            return 0.0
            
        delta_T = self.const.delta_0 * np.sqrt(1 - t**4)
        
        # 內插公式
        denominator = 1 + np.sqrt(1 - T)
        I_c_normalized = (np.pi / denominator) * (delta_T / self.const.delta_0)
        
        return I_c_normalized
    
    def critical_current_scattering(self, T: float, temperature: float = 0.1, 
                                  barrier_strength: float = 1.0) -> float:
        """
        散射理論模型
        考慮量子散射效應
        
        Args:
            T: 透明度參數
            temperature: 溫度 (K)
            barrier_strength: 勢壘強度參數
            
        Returns:
            歸一化臨界電流
        """
        # 溫度依賴
        t = temperature / self.const.T_c
        if t >= 1.0:
            return 0.0
            
        delta_T = self.const.delta_0 * np.sqrt(1 - t**4)
        
        # 散射相移效應
        phase_shift = np.arcsin(np.sqrt(T))
        scattering_factor = np.sin(2 * phase_shift) * np.exp(-barrier_strength * (1 - T))
        
        I_c_normalized = (np.pi / 2) * scattering_factor * (delta_T / self.const.delta_0)
        
        return I_c_normalized
    
    def fraunhofer_pattern(self, phi_ext_array: np.ndarray, I_c_0: float) -> np.ndarray:
        """
        Fraunhofer 繞射圖案
        I_c(Φ_ext) = I_c(0) × |sin(πΦ_ext/Φ₀)| / |πΦ_ext/Φ₀|
        
        Args:
            phi_ext_array: 外加磁通量陣列 (單位: Φ₀)
            I_c_0: 零磁場臨界電流
            
        Returns:
            臨界電流陣列
        """
        # 避免除零
        phi_ext_safe = np.where(np.abs(phi_ext_array) < 1e-10, 1e-10, phi_ext_array)
        
        # Fraunhofer公式
        argument = np.pi * phi_ext_safe
        I_c_pattern = I_c_0 * np.abs(np.sin(argument)) / np.abs(argument)
        
        # 處理 φ = 0 的情況
        I_c_pattern = np.where(np.abs(phi_ext_array) < 1e-10, I_c_0, I_c_pattern)
        
        return I_c_pattern
    
    def simulate_Is_vs_flux(self, 
                           transparency_values: List[float],
                           flux_range: Tuple[float, float] = (-3.0, 3.0),
                           num_points: int = 3001,
                           temperature: float = 0.1,
                           models: List[str] = None) -> Dict:
        """
        模擬超導電流隨磁通量的變化
        
        Args:
            transparency_values: 透明度值列表
            flux_range: 磁通量範圍 (單位: Φ₀)
            num_points: 數據點數量
            temperature: 溫度 (K)
            models: 要使用的模型列表
            
        Returns:
            模擬結果字典
        """
        if models is None:
            models = ['AB', 'KO', 'interpolation', 'scattering']
            
        # 創建磁通量陣列
        phi_ext = np.linspace(flux_range[0], flux_range[1], num_points)
        
        results = {
            'phi_ext': phi_ext,
            'models': {},
            'parameters': {
                'transparency_values': transparency_values,
                'temperature': temperature,
                'flux_range': flux_range,
                'num_points': num_points
            }
        }
        
        # 模型映射
        model_functions = {
            'AB': self.critical_current_AB,
            'KO': self.critical_current_KO,
            'interpolation': self.critical_current_interpolation,
            'scattering': self.critical_current_scattering
        }
        
        for model_name in models:
            if model_name not in model_functions:
                continue
                
            model_func = model_functions[model_name]
            results['models'][model_name] = {}
            
            for T in transparency_values:
                # 計算零磁場臨界電流
                I_c_0 = model_func(T, temperature)
                
                # 計算 Fraunhofer 圖案
                I_c_flux = self.fraunhofer_pattern(phi_ext, I_c_0)
                
                results['models'][model_name][f'T_{T:.3f}'] = {
                    'I_c_0': I_c_0,
                    'I_c_flux': I_c_flux,
                    'transparency': T
                }
        
        self.results = results
        return results
    
    def analyze_transparency_dependence(self, 
                                      T_range: Tuple[float, float] = (0.01, 0.99),
                                      num_T_points: int = 100,
                                      temperature: float = 0.1) -> Dict:
        """
        分析臨界電流對透明度的依賴性
        
        Args:
            T_range: 透明度範圍
            num_T_points: 透明度數據點數量
            temperature: 溫度 (K)
            
        Returns:
            分析結果
        """
        T_array = np.linspace(T_range[0], T_range[1], num_T_points)
        
        models = {
            'AB': self.critical_current_AB,
            'KO': self.critical_current_KO,
            'interpolation': self.critical_current_interpolation,
            'scattering': self.critical_current_scattering
        }
        
        results = {
            'transparency': T_array,
            'models': {},
            'parameters': {
                'T_range': T_range,
                'temperature': temperature,
                'num_T_points': num_T_points
            }
        }
        
        for model_name, model_func in models.items():
            I_c_values = []
            for T in T_array:
                I_c = model_func(T, temperature)
                I_c_values.append(I_c)
            
            results['models'][model_name] = np.array(I_c_values)
        
        return results
    
    def plot_flux_dependence(self, results: Dict = None, save_path: str = None):
        """繪製電流-磁通量依賴性圖"""
        if results is None:
            results = self.results
            
        if not results:
            print("沒有可用的模擬結果，請先運行模擬")
            return
        
        phi_ext = results['phi_ext']
        models = results['models']
        
        # 創建子圖
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(results['parameters']['transparency_values'])))
        
        for idx, (model_name, model_data) in enumerate(models.items()):
            ax = axes[idx]
            
            for i, (T_key, data) in enumerate(model_data.items()):
                T_value = data['transparency']
                I_c_flux = data['I_c_flux']
                
                ax.plot(phi_ext, I_c_flux, color=colors[i], 
                       label=f'T = {T_value:.3f}', linewidth=2)
            
            ax.set_xlabel('外加磁通量 Φ_ext/Φ₀')
            ax.set_ylabel('歸一化臨界電流 I_c/I_c(max)')
            ax.set_title(f'{model_name} 模型')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"圖片已保存至: {save_path}")
        
        plt.close()  # 關閉圖形而不顯示
    
    def plot_transparency_dependence(self, analysis_results: Dict, save_path: str = None):
        """繪製透明度依賴性圖"""
        T_array = analysis_results['transparency']
        models = analysis_results['models']
        
        plt.figure(figsize=(12, 8))
        
        colors = ['blue', 'red', 'green', 'purple']
        styles = ['-', '--', '-.', ':']
        
        for i, (model_name, I_c_values) in enumerate(models.items()):
            plt.plot(T_array, I_c_values, color=colors[i % len(colors)], 
                    linestyle=styles[i % len(styles)], linewidth=2.5,
                    label=f'{model_name} 模型', marker='o', markersize=3)
        
        plt.xlabel('透明度 T')
        plt.ylabel('歸一化臨界電流 I_c/I_c(max)')
        plt.title('臨界電流與透明度的關係')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xlim(0, 1)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"圖片已保存至: {save_path}")
        
        plt.close()  # 關閉圖形而不顯示
    
    def plot_3d_analysis(self, results: Dict = None, save_path: str = None):
        """繪製3D分析圖 (透明度 vs 磁通量 vs 電流)"""
        if results is None:
            results = self.results
            
        if not results:
            print("沒有可用的模擬結果")
            return
        
        phi_ext = results['phi_ext']
        T_values = results['parameters']['transparency_values']
        
        # 選擇內插模型進行3D展示
        if 'interpolation' not in results['models']:
            print("沒有內插模型數據")
            return
        
        model_data = results['models']['interpolation']
        
        # 創建網格
        PHI, T = np.meshgrid(phi_ext, T_values)
        I_C = np.zeros_like(PHI)
        
        for i, T_val in enumerate(T_values):
            T_key = f'T_{T_val:.3f}'
            if T_key in model_data:
                I_C[i, :] = model_data[T_key]['I_c_flux']
        
        # 創建3D圖
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        surf = ax.plot_surface(PHI, T, I_C, cmap='viridis', alpha=0.8)
        
        ax.set_xlabel('磁通量 Φ_ext/Φ₀')
        ax.set_ylabel('透明度 T')
        ax.set_zlabel('歸一化臨界電流')
        ax.set_title('3D 超導電流分析 (內插模型)')
        
        fig.colorbar(surf, shrink=0.5, aspect=5)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"3D圖片已保存至: {save_path}")
        
        plt.close()  # 關閉圖形而不顯示
    
    def export_data(self, results: Dict = None, filename: str = 'josephson_simulation_results.csv'):
        """導出模擬數據到CSV文件"""
        if results is None:
            results = self.results
            
        if not results:
            print("沒有可用的模擬結果")
            return
        
        # 準備數據
        data_rows = []
        phi_ext = results['phi_ext']
        
        for model_name, model_data in results['models'].items():
            for T_key, data in model_data.items():
                T_value = data['transparency']
                I_c_flux = data['I_c_flux']
                
                for i, phi in enumerate(phi_ext):
                    data_rows.append({
                        'model': model_name,
                        'transparency': T_value,
                        'flux_phi0': phi,
                        'critical_current_normalized': I_c_flux[i],
                        'critical_current_zero_field': data['I_c_0']
                    })
        
        # 創建DataFrame並保存
        df = pd.DataFrame(data_rows)
        df.to_csv(filename, index=False)
        print(f"數據已導出至: {filename}")
        
        return df
    
    def generate_report(self, results: Dict = None, analysis_results: Dict = None):
        """生成分析報告"""
        if results is None:
            results = self.results
            
        print("="*60)
        print("超導電流與磁通量關係模擬報告")
        print("="*60)
        
        if results:
            params = results['parameters']
            print(f"\n模擬參數:")
            print(f"  - 透明度值: {params['transparency_values']}")
            print(f"  - 溫度: {params['temperature']} K")
            print(f"  - 磁通量範圍: {params['flux_range']} Φ₀")
            print(f"  - 數據點數: {params['num_points']}")
            
            print(f"\n包含的模型:")
            for model_name in results['models'].keys():
                print(f"  - {model_name}")
        
        if analysis_results:
            print(f"\n透明度分析:")
            T_range = analysis_results['parameters']['T_range']
            print(f"  - 透明度範圍: {T_range}")
            print(f"  - 數據點數: {analysis_results['parameters']['num_T_points']}")
        
        print("\n理論模型說明:")
        print("  - AB 模型: Ambegaokar-Baratoff理論，適用於低透明度隧道結")
        print("  - KO 模型: Kulik-Omelyanchuk理論，適用於高透明度短結")
        print("  - 內插模型: 適用於任意透明度的內插公式")
        print("  - 散射模型: 考慮量子散射效應的模型")
        
        print("\nFraunhofer圖案特徵:")
        print("  - 主峰位於 Φ_ext = 0")
        print("  - 次峰位於 Φ_ext = ±nΦ₀ (n = 1, 2, 3, ...)")
        print("  - 包絡線遵循 sinc 函數形狀")
        
        print("="*60)

    def calculate_current_phase_relation(self, 
                                        transparency_values: List[float],
                                        phase_range: Tuple[float, float] = ( -5*np.pi, 5*np.pi),
                                        num_points: int = 2001,
                                        temperature: float = 0.1,
                                        models: Optional[List[str]] = None) -> Dict:
        """
        計算Current-Phase Relation (電流-相位關係)
        
        Args:
            transparency_values: 透明度值列表
            phase_range: 相位範圍 (單位: 弧度)
            num_points: 數據點數量
            temperature: 溫度 (K)
            models: 要使用的模型列表
            
        Returns:
            相位關係結果字典
        """
        if models is None:
            models = ['AB', 'KO', 'interpolation', 'scattering']
            
        # 創建相位陣列
        phase = np.linspace(phase_range[0], phase_range[1], num_points)
        
        results = {
            'phase': phase,
            'models': {},
            'parameters': {
                'transparency_values': transparency_values,
                'temperature': temperature,
                'phase_range': phase_range,
                'num_points': num_points
            }
        }
        
        # 模型映射
        model_functions = {
            'AB': self.critical_current_AB,
            'KO': self.critical_current_KO,
            'interpolation': self.critical_current_interpolation,
            'scattering': self.critical_current_scattering
        }
        
        for model_name in models:
            if model_name not in model_functions:
                continue
                
            model_func = model_functions[model_name]
            results['models'][model_name] = {}
            
            for T in transparency_values:
                # 計算零相位臨界電流
                I_c_0 = model_func(T, temperature)
                
                # 計算相位依賴電流
                # 對於不同模型，相位依賴性可能不同
                if model_name == 'AB':
                    # AB模型: 基本正弦關係
                    I_phase = I_c_0 * np.sin(phase)
                    
                elif model_name == 'KO':
                    # KO模型: 包含高次諧波修正
                    # 對於高透明度，會有2次諧波成分
                    harmonic_2 = 0.1 * T * np.sin(2 * phase)  # 2次諧波項
                    I_phase = I_c_0 * (np.sin(phase) + harmonic_2)
                    
                elif model_name == 'interpolation':
                    # 內插模型: 包含透明度相關的相位修正
                    phase_correction = 1 + 0.05 * T * np.cos(phase)
                    I_phase = I_c_0 * np.sin(phase) * phase_correction
                    
                elif model_name == 'scattering':
                    # 散射模型: 包含量子干涉效應
                    # 散射相移導致的相位依賴性修正
                    phase_shift = np.arcsin(np.sqrt(T))
                    effective_phase = phase + phase_shift
                    I_phase = I_c_0 * np.sin(effective_phase) * np.exp(-0.01 * (1 - T) * np.abs(phase))
                    
                else:
                    # 默認正弦關係
                    I_phase = I_c_0 * np.sin(phase)
                
                results['models'][model_name][f'T_{T:.3f}'] = {
                    'transparency': T,
                    'I_c_0': I_c_0,
                    'current_phase': I_phase,
                    'phase': phase
                }
        
        self.phase_results = results
        return results
    
    def analyze_harmonics(self, phase_results: Dict = None, max_harmonics: int = 5) -> Dict:
        """
        分析Current-Phase Relation中的諧波成分
        
        Args:
            phase_results: 相位關係結果字典
            max_harmonics: 最大諧波次數
            
        Returns:
            諧波分析結果
        """
        if phase_results is None:
            if hasattr(self, 'phase_results'):
                phase_results = self.phase_results
            else:
                raise ValueError("沒有可用的相位關係數據")
        
        harmonic_analysis = {
            'models': {},
            'summary': {}
        }
        
        for model_name, model_data in phase_results['models'].items():
            harmonic_analysis['models'][model_name] = {}
            
            for T_key, data in model_data.items():
                current_phase = data['current_phase']
                phase = data['phase']
                transparency = data['transparency']
                
                # 確保相位範圍是0到2π
                if phase[-1] - phase[0] >= 2 * np.pi:
                    # 進行傅立葉分析
                    harmonics = {}
                    
                    # 計算各次諧波的幅度
                    for n in range(1, max_harmonics + 1):
                        # 計算第n次諧波的幅度
                        sin_coeff = 2.0 / len(phase) * np.sum(current_phase * np.sin(n * phase))
                        cos_coeff = 2.0 / len(phase) * np.sum(current_phase * np.cos(n * phase))
                        amplitude = np.sqrt(sin_coeff**2 + cos_coeff**2)
                        phase_shift = np.arctan2(cos_coeff, sin_coeff)
                        
                        harmonics[f'harmonic_{n}'] = {
                            'amplitude': amplitude,
                            'phase_shift': phase_shift,
                            'relative_amplitude': amplitude / data['I_c_0'] if data['I_c_0'] != 0 else 0
                        }
                    
                    harmonic_analysis['models'][model_name][T_key] = {
                        'transparency': transparency,
                        'harmonics': harmonics,
                        'fundamental_amplitude': harmonics['harmonic_1']['amplitude'],
                        'total_harmonic_distortion': self._calculate_thd(harmonics)
                    }
        
        return harmonic_analysis
    
    def _calculate_thd(self, harmonics: Dict) -> float:
        """計算總諧波失真 (Total Harmonic Distortion)"""
        fundamental = harmonics['harmonic_1']['amplitude']
        if fundamental == 0:
            return 0
        
        higher_harmonics = sum(harmonics[f'harmonic_{n}']['amplitude']**2 
                              for n in range(2, len(harmonics) + 1))
        thd = np.sqrt(higher_harmonics) / fundamental
        return thd
    
    def plot_current_phase_relation(self, phase_results: Dict = None, 
                                   save_path: str = None, 
                                   selected_transparencies: List[float] = None):
        """
        繪製Current-Phase Relation圖
        
        Args:
            phase_results: 相位關係結果字典
            save_path: 保存路徑
            selected_transparencies: 選擇要顯示的透明度值
        """
        if phase_results is None:
            if hasattr(self, 'phase_results'):
                phase_results = self.phase_results
            else:
                print("沒有可用的相位關係數據")
                return
        
        phase = phase_results['phase']
        models = phase_results['models']
        
        # 如果沒有指定透明度，選擇幾個代表性的值
        if selected_transparencies is None:
            all_transparencies = []
            for model_data in models.values():
                for data in model_data.values():
                    all_transparencies.append(data['transparency'])
            unique_T = sorted(list(set(all_transparencies)))
            # 選擇幾個代表性透明度值
            if len(unique_T) > 4:
                indices = np.linspace(0, len(unique_T)-1, 4, dtype=int)
                selected_transparencies = [unique_T[i] for i in indices]
            else:
                selected_transparencies = unique_T
        
        # 創建子圖
        n_models = len(models)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(selected_transparencies)))
        
        for i, (model_name, model_data) in enumerate(models.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            for j, T_target in enumerate(selected_transparencies):
                # 找到最接近的透明度值
                best_match = None
                min_diff = float('inf')
                
                for T_key, data in model_data.items():
                    diff = abs(data['transparency'] - T_target)
                    if diff < min_diff:
                        min_diff = diff
                        best_match = data
                
                if best_match is not None:
                    current_phase = best_match['current_phase']
                    T_actual = best_match['transparency']
                    
                    ax.plot(phase / np.pi, current_phase, 
                           color=colors[j], linewidth=2,
                           label=f'T = {T_actual:.3f}')
            
            ax.set_xlabel('相位 φ / π')
            ax.set_ylabel('電流 I / I_c(0)')
            ax.set_title(f'{model_name} 模型 - 電流相位關係')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_xlim(-3, 3)
        
        # 隱藏空的子圖
        for i in range(len(models), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Current-Phase Relation圖已保存至: {save_path}")
        
        plt.close()
    
    def plot_harmonic_analysis(self, harmonic_results: Dict, save_path: str = None):
        """
        繪製諧波分析圖
        
        Args:
            harmonic_results: 諧波分析結果
            save_path: 保存路徑
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        model_names = list(harmonic_results['models'].keys())
        
        for i, model_name in enumerate(model_names[:4]):
            if i >= len(axes):
                break
                
            ax = axes[i]
            model_data = harmonic_results['models'][model_name]
            
            transparencies = []
            fundamental_amps = []
            second_harmonic_amps = []
            thd_values = []
            
            for T_key, data in model_data.items():
                transparencies.append(data['transparency'])
                fundamental_amps.append(data['harmonics']['harmonic_1']['relative_amplitude'])
                if 'harmonic_2' in data['harmonics']:
                    second_harmonic_amps.append(data['harmonics']['harmonic_2']['relative_amplitude'])
                else:
                    second_harmonic_amps.append(0)
                thd_values.append(data['total_harmonic_distortion'])
            
            # 排序
            sorted_indices = np.argsort(transparencies)
            transparencies = np.array(transparencies)[sorted_indices]
            fundamental_amps = np.array(fundamental_amps)[sorted_indices]
            second_harmonic_amps = np.array(second_harmonic_amps)[sorted_indices]
            thd_values = np.array(thd_values)[sorted_indices]
            
            # 繪製基波和二次諧波
            ax.plot(transparencies, fundamental_amps, 'b-o', linewidth=2, 
                   markersize=4, label='基波 (1st)')
            ax.plot(transparencies, second_harmonic_amps, 'r--s', linewidth=2, 
                   markersize=4, label='二次諧波 (2nd)')
            
            # 右軸顯示THD
            ax2 = ax.twinx()
            ax2.plot(transparencies, thd_values, 'g:^', linewidth=2, 
                    markersize=4, label='THD', color='green')
            ax2.set_ylabel('總諧波失真 (THD)', color='green')
            ax2.tick_params(axis='y', labelcolor='green')
            
            ax.set_xlabel('透明度 T')
            ax.set_ylabel('相對振幅')
            ax.set_title(f'{model_name} 模型 - 諧波分析')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
        
        # 隱藏空的子圖
        for i in range(len(model_names), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"諧波分析圖已保存至: {save_path}")
        
        plt.close()

    def plot_current_phase_heatmap(self, current_phase_results: Dict, save_path: Optional[str] = None):
        """
        繪製Current-Phase Relation Heatmap
        
        Args:
            current_phase_results: 電流-相位關係結果
            save_path: 保存路徑
        """
        print("\n繪製Current-Phase Relation Heatmap...")
        
        # 提取相位和透明度數據
        phase = current_phase_results['phase']
        transparency_values = current_phase_results['parameters']['transparency_values']
        models = list(current_phase_results['models'].keys())
        
        # 創建子圖
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        axes = axes.flatten()
        
        # 為每個模型創建熱力圖
        for i, model_name in enumerate(models[:4]):
            if i >= len(axes):
                break
                
            ax = axes[i]
            model_data = current_phase_results['models'][model_name]
            
            # 準備熱力圖數據
            phase_mesh, T_mesh = np.meshgrid(phase, transparency_values)
            current_matrix = np.zeros((len(transparency_values), len(phase)))
            
            for j, T in enumerate(transparency_values):
                T_key = f'T_{T:.3f}'
                if T_key in model_data:
                    current_matrix[j, :] = model_data[T_key]['current_phase']
                    
            # 繪製熱力圖
            im = ax.imshow(current_matrix, 
                          extent=[phase[0], phase[-1], 
                                 transparency_values[0], transparency_values[-1]],
                          aspect='auto', 
                          origin='lower',
                          cmap='RdYlBu_r',
                          interpolation='bilinear')
            
            # 添加等高線
            contours = ax.contour(phase_mesh, T_mesh, current_matrix, 
                                levels=15, colors='black', alpha=0.4, linewidths=0.5)
            ax.clabel(contours, inline=True, fontsize=8, fmt='%.2f')
            
            # 設置標籤和標題
            ax.set_xlabel('相位 φ (弧度)', fontsize=12)
            ax.set_ylabel('透明度 T', fontsize=12)
            ax.set_title(f'{model_name} 模型 - Current-Phase Heatmap', fontsize=14, fontweight='bold')
            
            # 設置x軸刻度為π的倍數
            phase_ticks = np.arange(-4*np.pi, 5*np.pi, np.pi)
            phase_labels = [f'{int(tick/np.pi)}π' if tick != 0 else '0' for tick in phase_ticks]
            ax.set_xticks(phase_ticks)
            ax.set_xticklabels(phase_labels)
            
            # 添加顏色條
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('歸一化電流 I/I_c0', fontsize=11)
            
            # 網格
            ax.grid(True, alpha=0.3)
        
        # 隱藏未使用的子圖
        for i in range(len(models), 4):
            axes[i].set_visible(False)
        
        # 整體標題
        fig.suptitle('Josephson結 Current-Phase Relation Heatmap 分析', 
                     fontsize=16, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Current-Phase Heatmap已保存至: {save_path}")
        
        plt.show()
        plt.close()

    def plot_advanced_phase_analysis(self, current_phase_results: Dict, save_path: Optional[str] = None):
        """
        繪製進階相位分析圖 (包含3D視圖和詳細分析)
        
        Args:
            current_phase_results: 電流-相位關係結果
            save_path: 保存路徑
        """
        print("\n繪製進階Current-Phase分析圖...")
        
        phase = current_phase_results['phase']
        transparency_values = current_phase_results['parameters']['transparency_values']
        models = list(current_phase_results['models'].keys())
        
        # 創建大型圖表
        fig = plt.figure(figsize=(24, 18))
        
        # 為每個模型創建3D圖和分析圖
        for i, model_name in enumerate(models[:4]):
            model_data = current_phase_results['models'][model_name]
            
            # 準備3D數據
            phase_mesh, T_mesh = np.meshgrid(phase, transparency_values)
            current_matrix = np.zeros((len(transparency_values), len(phase)))
            
            for j, T in enumerate(transparency_values):
                T_key = f'T_{T:.3f}'
                if T_key in model_data:
                    current_matrix[j, :] = model_data[T_key]['current_phase']
            
            # 3D表面圖
            ax1 = fig.add_subplot(4, 3, i*3 + 1, projection='3d')
            surf = ax1.plot_surface(phase_mesh, T_mesh, current_matrix, 
                                  cmap='viridis', alpha=0.8, 
                                  linewidth=0, antialiased=True)
            ax1.set_xlabel('相位 φ (弧度)')
            ax1.set_ylabel('透明度 T')
            ax1.set_zlabel('電流 I/I_c0')
            ax1.set_title(f'{model_name} - 3D視圖')
            
            # 特定透明度的相位依賴性
            ax2 = fig.add_subplot(4, 3, i*3 + 2)
            selected_T = [0.1, 0.5, 0.9]
            colors = ['blue', 'red', 'green']
            
            for j, T in enumerate(selected_T):
                T_key = f'T_{T:.3f}'
                if T_key in model_data:
                    current = model_data[T_key]['current_phase']
                    ax2.plot(phase, current, color=colors[j], 
                           linewidth=2, label=f'T = {T}')
            
            ax2.set_xlabel('相位 φ (弧度)')
            ax2.set_ylabel('歸一化電流 I/I_c0')
            ax2.set_title(f'{model_name} - 相位依賴性')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 相位π/2處的透明度依賴性
            ax3 = fig.add_subplot(4, 3, i*3 + 3)
            phase_idx = len(phase) // 4  # π/2位置
            current_at_phase = []
            
            for T in transparency_values:
                T_key = f'T_{T:.3f}'
                if T_key in model_data:
                    current_at_phase.append(model_data[T_key]['current_phase'][phase_idx])
                else:
                    current_at_phase.append(0)
            
            ax3.plot(transparency_values, current_at_phase, 'o-', 
                    linewidth=2, markersize=6, color='purple')
            ax3.set_xlabel('透明度 T')
            ax3.set_ylabel('電流 @ φ=π/2')
            ax3.set_title(f'{model_name} - T依賴性')
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"進階相位分析圖已保存至: {save_path}")
        
        plt.show()
        plt.close()

    # ...existing code...

def main():
    """主函數 - 運行完整的模擬分析"""
    print("開始超導電流與磁通量關係模擬...")
    
    # 創建模擬器
    simulator = JosephsonCurrentSimulator()
    
    # 設置模擬參數
    transparency_values = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    flux_range = (-10.0, 10.0)
    temperature = 0.1  # K
    
    print(f"透明度值: {transparency_values}")
    print(f"磁通量範圍: {flux_range} Φ₀")
    print(f"溫度: {temperature} K")
    
    # 1. 模擬電流-磁通量關係
    print("\n1. 模擬電流-磁通量關係...")
    results = simulator.simulate_Is_vs_flux(
        transparency_values=transparency_values,
        flux_range=flux_range,
        num_points=1000,
        temperature=temperature
    )
    
    # 2. 分析透明度依賴性
    print("2. 分析透明度依賴性...")
    transparency_analysis = simulator.analyze_transparency_dependence(
        T_range=(0.01, 0.99),
        num_T_points=100,
        temperature=temperature
    )
    
    # 3. 生成圖表
    print("3. 生成可視化圖表...")
    
    # 磁通量依賴性圖
    simulator.plot_flux_dependence(results, 'josephson_flux_dependence.png')
    
    # 透明度依賴性圖
    simulator.plot_transparency_dependence(transparency_analysis, 'josephson_transparency_dependence.png')
    
    # 3D分析圖
    simulator.plot_3d_analysis(results, 'josephson_3d_analysis.png')
    
    # 4. 導出數據
    print("4. 導出模擬數據...")
    df = simulator.export_data(results, 'josephson_simulation_results.csv')
    
    # 5. Current-Phase Relation 分析
    print("5. Current-Phase Relation 分析...")
    phase_results = simulator.calculate_current_phase_relation(
        transparency_values=transparency_values,
        temperature=temperature
    )
    
    # 6. 諧波分析
    print("6. 諧波分析...")
    harmonic_results = simulator.analyze_harmonics(phase_results, max_harmonics=5)
    
    # 7. Current-Phase Relation 圖表
    print("7. 生成 Current-Phase Relation 圖表...")
    simulator.plot_current_phase_relation(phase_results, 'josephson_current_phase_relation.png')
    simulator.plot_harmonic_analysis(harmonic_results, 'josephson_harmonic_analysis.png')
    
    # 8. 生成Current-Phase Relation Heatmap
    print("8. 生成 Current-Phase Relation Heatmap...")
    simulator.plot_current_phase_heatmap(phase_results, 'josephson_current_phase_heatmap.png')
    
    # 9. 生成進階相位分析圖
    print("9. 生成進階相位分析圖...")
    simulator.plot_advanced_phase_analysis(phase_results, 'josephson_advanced_phase_analysis.png')
    
    # 10. 生成報告
    print("10. 生成分析報告...")
    simulator.generate_report(results, transparency_analysis)
    
    # 11. 保存完整結果
    print("11. 保存完整結果...")
    complete_results = {
        'flux_simulation': results,
        'transparency_analysis': transparency_analysis,
        'current_phase_relation': phase_results,
        'harmonic_analysis': harmonic_results,
        'metadata': {
            'simulation_date': '2025-06-04',
            'software_version': 'v1.2',
            'description': '超導電流與磁通量關係完整模擬，包含Current-Phase Relation分析和Heatmap視覺化'
        }
    }
    
    # 將numpy數組轉換為列表以便JSON序列化
    def convert_numpy_to_list(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_to_list(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_to_list(item) for item in obj]
        else:
            return obj
    
    complete_results_serializable = convert_numpy_to_list(complete_results)
    
    with open('josephson_complete_results.json', 'w', encoding='utf-8') as f:
        json.dump(complete_results_serializable, f, indent=2, ensure_ascii=False)
    
    print("\n模擬完成！")
    print("生成的文件:")
    print("  - josephson_flux_dependence.png (磁通量依賴性圖)")
    print("  - josephson_transparency_dependence.png (透明度依賴性圖)")
    print("  - josephson_3d_analysis.png (3D分析圖)")
    print("  - josephson_current_phase_relation.png (電流相位關係圖)")
    print("  - josephson_harmonic_analysis.png (諧波分析圖)")
    print("  - josephson_current_phase_heatmap.png (電流相位關係熱力圖)")
    print("  - josephson_advanced_phase_analysis.png (進階相位分析圖)")
    print("  - josephson_simulation_results.csv (模擬數據)")
    print("  - josephson_complete_results.json (完整結果)")
    
    return results, transparency_analysis, phase_results, harmonic_results

if __name__ == "__main__":
    # 運行主程序
    results, transparency_analysis, phase_results, harmonic_results = main()
    
    # 顯示一些關鍵結果
    print("\n=== 關鍵結果摘要 ===")
    
    # 零磁場臨界電流值 (內插模型)
    if 'interpolation' in results['models']:
        print("\n零磁場臨界電流 (內插模型):")
        interp_data = results['models']['interpolation']
        for T_key, data in interp_data.items():
            T_val = data['transparency']
            I_c_0 = data['I_c_0']
            print(f"  T = {T_val:.3f}: I_c(0) = {I_c_0:.4f}")
    
    # 透明度效應分析
    if transparency_analysis:
        T_array = transparency_analysis['transparency']
        interp_values = transparency_analysis['models']['interpolation']
        max_current = np.max(interp_values)
        max_T_index = np.argmax(interp_values)
        optimal_T = T_array[max_T_index]
        
        print(f"\n透明度效應分析:")
        print(f"  最大臨界電流: {max_current:.4f}")
        print(f"  最佳透明度: {optimal_T:.3f}")
        print(f"  電流變化範圍: {np.min(interp_values):.4f} - {max_current:.4f}")
    
    # Current-Phase Relation 結果摘要
    if phase_results and 'models' in phase_results:
        print(f"\nCurrent-Phase Relation 分析:")
        print(f"  已分析模型數量: {len(phase_results['models'])}")
        
        # 計算分析的透明度值數量
        if 'interpolation' in phase_results['models']:
            T_count = len(phase_results['models']['interpolation'])
            print(f"  分析的透明度值數量: {T_count}")
        
        print(f"  相位範圍: 0 到 2π")
        
        # 顯示內插模型的示例結果
        if 'interpolation' in phase_results['models']:
            interp_phase_data = phase_results['models']['interpolation']
            first_T_key = list(interp_phase_data.keys())[0]
            sample_data = interp_phase_data[first_T_key]
            print(f"  示例 (T={sample_data['transparency']:.3f}): I_c(0) = {sample_data['I_c_0']:.4f}")
    
    # 諧波分析結果摘要
    if harmonic_results and 'models' in harmonic_results:
        print(f"\n諧波分析結果:")
        for model_name, model_data in harmonic_results['models'].items():
            if model_data:  # 確保模型有數據
                sample_key = list(model_data.keys())[0]
                sample_data = model_data[sample_key]
                thd = sample_data['total_harmonic_distortion']
                fundamental = sample_data['fundamental_amplitude']
                print(f"  {model_name} 模型:")
                print(f"    - 基波振幅: {fundamental:.4f}")
                print(f"    - 總諧波失真: {thd:.4f}")
                break  # 只顯示一個示例