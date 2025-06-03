"""
溫度效應建模模組

實現約瑟夫森結的溫度相關物理效應，包括：
- BCS 超導能隙溫度依賴性
- 臨界電流溫度依賴性  
- 熱噪聲效應
- 相位擴散建模
"""

import numpy as np
from typing import Union, Tuple, Optional
from scipy.special import ellipk, ellipe
import logging

logger = logging.getLogger(__name__)

class TemperatureEffectsModel:
    """
    約瑟夫森結溫度效應建模類
    
    基於 BCS 理論和實驗觀測實現溫度相關的物理效應
    """
    
    def __init__(self, 
                 Tc: float = 9.0,  # 臨界溫度 (K)
                 Delta_0: float = 1.76e-3,  # T=0時的能隙 (eV) 
                 Ic_0: float = 1e-6,  # T=0時的臨界電流 (A)
                 R_n: float = 1e3):  # 正常態電阻 (Ω)
        """
        初始化溫度效應模型
        
        Parameters:
        -----------
        Tc : float
            超導臨界溫度 (K)
        Delta_0 : float  
            T=0K時的超導能隙 (eV)
        Ic_0 : float
            T=0K時的臨界電流 (A)
        R_n : float
            正常態電阻 (Ω)
        """
        self.Tc = Tc
        self.Delta_0 = Delta_0
        self.Ic_0 = Ic_0
        self.R_n = R_n
        
        # 物理常數
        self.k_B = 8.617e-5  # 玻爾茲曼常數 (eV/K)
        self.h = 4.136e-15   # 普朗克常數 (eV·s)
        self.e = 1.602e-19   # 電子電荷 (C)
        
        logger.info(f"溫度效應模型初始化: Tc={Tc}K, Δ₀={Delta_0*1000:.2f}meV")
    
    def energy_gap_temperature(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        計算超導能隙的溫度依賴性
        
        使用 BCS 弱耦合近似理論
        
        Parameters:
        -----------
        T : float or array
            溫度 (K)
            
        Returns:
        --------
        Delta_T : float or array
            溫度T時的超導能隙 (eV)
        """
        T = np.asarray(T)
        
        # 避免 T >= Tc 的情況
        T_safe = np.minimum(T, self.Tc * 0.999)
        
        # BCS 弱耦合理論能隙方程的近似解
        t = T_safe / self.Tc  # 歸一化溫度
        
        # 使用精確的 BCS 能隙方程近似
        Delta_T = self.Delta_0 * np.tanh(1.74 * np.sqrt(self.Tc / T_safe - 1))
        
        # 處理 T >= Tc 的情況
        Delta_T = np.where(T >= self.Tc, 0.0, Delta_T)
        
        return Delta_T
    
    def critical_current_temperature(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        計算臨界電流的溫度依賴性
        
        基於 Ambegaokar-Baratoff 關係
        
        Parameters:
        -----------
        T : float or array
            溫度 (K)
            
        Returns:
        --------
        Ic_T : float or array
            溫度T時的臨界電流 (A)
        """
        Delta_T = self.energy_gap_temperature(T)
        
        # Ambegaokar-Baratoff 關係: Ic ∝ Δ tanh(Δ/2kT)
        # 添加量子修正項
        arg = Delta_T / (2 * self.k_B * np.maximum(T, 0.1))
        Ic_T = self.Ic_0 * (Delta_T / self.Delta_0) * np.tanh(arg)
        
        return Ic_T
    
    def thermal_noise_current(self, T: Union[float, np.ndarray], 
                            bandwidth: float = 1e6) -> Union[float, np.ndarray]:
        """
        計算熱噪聲電流
        
        基於 Nyquist 定理
        
        Parameters:
        -----------
        T : float or array
            溫度 (K)
        bandwidth : float
            測量頻寬 (Hz)
            
        Returns:
        --------
        I_noise : float or array
            熱噪聲電流 RMS 值 (A)
        """
        # Nyquist 熱噪聲: <I²> = 4kT·B/R
        I_noise_squared = 4 * self.k_B * T * bandwidth * self.e / self.R_n
        I_noise = np.sqrt(I_noise_squared)
        
        return I_noise
    
    def phase_diffusion_rate(self, T: Union[float, np.ndarray], 
                           I_bias: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        計算相位擴散率
        
        考慮熱激發導致的相位擴散效應
        
        Parameters:
        -----------
        T : float or array
            溫度 (K)
        I_bias : float or array  
            偏置電流 (A)
            
        Returns:
        --------
        gamma : float or array
            相位擴散率 (1/s)
        """
        Ic_T = self.critical_current_temperature(T)
        
        # 避免除零
        I_bias = np.asarray(I_bias)
        Ic_T = np.asarray(Ic_T)
        
        # 相位擴散率的熱激活形式
        # γ = ω_J exp(-ΔU/kT)，其中 ΔU 是勢壘高度
        omega_J = 2 * self.e * Ic_T / (self.h)  # 約瑟夫森頻率
        
        # 勢壘高度計算
        i_norm = np.minimum(np.abs(I_bias) / Ic_T, 0.99)  # 歸一化電流
        Delta_U = 2 * Ic_T * self.h / (2 * self.e) * (1 - i_norm**2)**(3/2)
        
        # 相位擴散率
        gamma = omega_J * np.exp(-Delta_U / (self.k_B * T * self.e))
        
        return gamma
    
    def temperature_dependent_response(self, 
                                     phi_ext: np.ndarray,
                                     T: float,
                                     I_bias: Optional[float] = None,
                                     include_noise: bool = True,
                                     noise_bandwidth: float = 1e6) -> Tuple[np.ndarray, dict]:
        """
        計算完整的溫度相關約瑟夫森響應
        
        Parameters:
        -----------
        phi_ext : array
            外部磁通 (以磁通量子為單位)
        T : float
            溫度 (K)
        I_bias : float, optional
            偏置電流 (A)
        include_noise : bool
            是否包含熱噪聲
        noise_bandwidth : float
            噪聲頻寬 (Hz)
            
        Returns:
        --------
        current : array
            溫度修正的電流響應 (A)
        info : dict
            溫度效應的詳細信息
        """
        phi_ext = np.asarray(phi_ext)
        
        # 計算溫度相關參數
        Ic_T = self.critical_current_temperature(T)
        Delta_T = self.energy_gap_temperature(T)
        
        # 基礎約瑟夫森電流
        current = Ic_T * np.sin(2 * np.pi * phi_ext)
        
        # 添加熱噪聲
        if include_noise:
            I_noise = self.thermal_noise_current(T, noise_bandwidth)
            noise = np.random.normal(0, I_noise, len(phi_ext))
            current += noise
        
        # 計算相位擴散效應（如果提供偏置電流）
        gamma = None
        if I_bias is not None:
            gamma = self.phase_diffusion_rate(T, I_bias)
            # 相位擴散導致的電流展寬（簡化模型）
            phase_broadening = np.sqrt(gamma * self.h / (2 * self.e * Ic_T))
            current *= np.exp(-phase_broadening * np.abs(phi_ext))
        
        # 收集溫度效應信息
        info = {
            'temperature': T,
            'energy_gap': Delta_T,
            'critical_current': Ic_T, 
            'thermal_noise_rms': self.thermal_noise_current(T, noise_bandwidth) if include_noise else 0,
            'phase_diffusion_rate': gamma,
            'gap_reduction_factor': Delta_T / self.Delta_0,
            'current_reduction_factor': Ic_T / self.Ic_0
        }
        
        return current, info
    
    def temperature_sweep_analysis(self,
                                 phi_ext: np.ndarray, 
                                 T_range: Tuple[float, float] = (1.4, 10.0),
                                 num_points: int = 50) -> dict:
        """
        進行溫度掃描分析
        
        Parameters:
        -----------
        phi_ext : array
            外部磁通範圍
        T_range : tuple
            溫度範圍 (K)
        num_points : int
            溫度點數
            
        Returns:
        --------
        results : dict
            溫度掃描結果
        """
        T_min, T_max = T_range
        temperatures = np.linspace(T_min, min(T_max, self.Tc * 0.99), num_points)
        
        results = {
            'temperatures': temperatures,
            'critical_currents': [],
            'energy_gaps': [],
            'current_responses': [],
            'thermal_noise': []
        }
        
        logger.info(f"開始溫度掃描分析: {T_min}K - {T_max}K, {num_points} 個溫度點")
        
        for T in temperatures:
            # 計算溫度相關響應
            current, info = self.temperature_dependent_response(phi_ext, T)
            
            results['critical_currents'].append(info['critical_current'])
            results['energy_gaps'].append(info['energy_gap'])
            results['current_responses'].append(current)
            results['thermal_noise'].append(info['thermal_noise_rms'])
        
        # 轉換為 numpy 數組
        for key in ['critical_currents', 'energy_gaps', 'thermal_noise']:
            results[key] = np.array(results[key])
        
        logger.info("溫度掃描分析完成")
        return results

# 工廠函數
def create_nb_junction_model(Tc: float = 9.2) -> TemperatureEffectsModel:
    """創建鈮約瑟夫森結溫度模型"""
    return TemperatureEffectsModel(
        Tc=Tc,
        Delta_0=1.52e-3,  # 鈮的能隙
        Ic_0=1e-6,
        R_n=1e3
    )

def create_al_junction_model(Tc: float = 1.2) -> TemperatureEffectsModel:
    """創建鋁約瑟夫森結溫度模型"""  
    return TemperatureEffectsModel(
        Tc=Tc,
        Delta_0=1.8e-4,  # 鋁的能隙
        Ic_0=1e-7,
        R_n=5e3
    )
