"""
實時參數優化系統 - 約瑟夫森結分析框架的動態優化模組
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from scipy.optimize import minimize, differential_evolution
from scipy.stats import pearsonr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import time
import logging
from pathlib import Path

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealTimeOptimizer:
    """
    實時參數優化器：自動調整約瑟夫森結模型參數以最大化擬合品質
    """
    
    def __init__(self, 
                 target_correlation: float = 0.5,
                 max_iterations: int = 100,
                 convergence_threshold: float = 1e-4,
                 optimization_method: str = 'bayesian'):
        """
        初始化實時優化器
        
        Args:
            target_correlation: 目標相關係數
            max_iterations: 最大迭代次數
            convergence_threshold: 收斂閾值
            optimization_method: 優化方法 ('bayesian', 'differential', 'gradient')
        """
        self.target_correlation = target_correlation
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.optimization_method = optimization_method
        
        # 優化歷史記錄
        self.optimization_history = []
        self.best_parameters = None
        self.best_score = -np.inf
        
        # 參數邊界定義
        self.parameter_bounds = {
            'Ic_base': (1e-8, 1e-4),
            'field_scale': (1e-8, 1e-4),
            'phase_offset': (0, 2*np.pi),
            'asymmetry': (-0.5, 0.5),
            'noise_level': (1e-9, 1e-5)
        }
        
        # 初始化貝葉斯優化器
        if optimization_method == 'bayesian':
            self.gp_optimizer = GaussianProcessRegressor(
                kernel=Matern(length_scale=1.0, nu=2.5),
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=10
            )
            self.bayesian_samples = []
            self.bayesian_scores = []
    
    def objective_function(self, parameters: np.ndarray, 
                          experimental_data: np.ndarray, 
                          field_data: np.ndarray) -> float:
        """
        目標函數：計算給定參數下的模型品質分數
        
        Args:
            parameters: 模型參數向量
            experimental_data: 實驗數據
            field_data: 磁場數據
            
        Returns:
            負相關係數（用於最小化）
        """
        try:
            # 解包參數
            Ic_base, field_scale, phase_offset, asymmetry, noise_level = parameters
            
            # 生成約瑟夫森結響應
            simulated_data = self.generate_josephson_response(
                field_data, Ic_base, field_scale, phase_offset, asymmetry, noise_level
            )
            
            # 計算相關係數
            if len(simulated_data) != len(experimental_data):
                min_len = min(len(simulated_data), len(experimental_data))
                simulated_data = simulated_data[:min_len]
                experimental_data = experimental_data[:min_len]
            
            correlation, _ = pearsonr(experimental_data, simulated_data)
            
            # 處理 NaN 值
            if np.isnan(correlation):
                return 1.0  # 返回大的懲罰值
            
            # 添加物理約束懲罰
            penalty = self.physics_constraint_penalty(parameters)
            
            # 返回負相關係數（用於最小化）
            return -(correlation - penalty)
            
        except Exception as e:
            logger.warning(f"目標函數計算錯誤: {e}")
            return 1.0  # 返回大的懲罰值
    
    def generate_josephson_response(self, field_data: np.ndarray, 
                                  Ic_base: float, field_scale: float, 
                                  phase_offset: float, asymmetry: float, 
                                  noise_level: float) -> np.ndarray:
        """
        生成約瑟夫森結的磁場響應
        
        Args:
            field_data: 磁場數據
            Ic_base: 基礎臨界電流
            field_scale: 磁場轉換因子
            phase_offset: 相位偏移
            asymmetry: 非對稱性參數
            noise_level: 雜訊水平
            
        Returns:
            模擬的電流響應
        """
        # 將磁場轉換為相位
        phase = field_data * field_scale + phase_offset
        
        # 基本約瑟夫森結響應
        Ic_response = Ic_base * np.abs(np.cos(phase / 2))
        
        # 添加非對稱性
        Ic_response *= (1 + asymmetry * np.sin(phase))
        
        # 添加雜訊
        noise = np.random.normal(0, noise_level, len(field_data))
        Ic_response += noise
        
        return Ic_response
    
    def physics_constraint_penalty(self, parameters: np.ndarray) -> float:
        """
        物理約束懲罰函數
        
        Args:
            parameters: 模型參數
            
        Returns:
            懲罰值
        """
        penalty = 0.0
        
        Ic_base, field_scale, phase_offset, asymmetry, noise_level = parameters
        
        # 檢查參數是否在合理範圍內
        if Ic_base <= 0 or field_scale <= 0 or noise_level <= 0:
            penalty += 10.0
        
        # 檢查非對稱性參數
        if abs(asymmetry) > 0.5:
            penalty += 5.0
        
        # 檢查相位偏移
        if phase_offset < 0 or phase_offset > 2*np.pi:
            penalty += 2.0
        
        return penalty
    
    def bayesian_optimization_step(self, experimental_data: np.ndarray, 
                                 field_data: np.ndarray) -> np.ndarray:
        """
        執行一步貝葉斯優化
        
        Args:
            experimental_data: 實驗數據
            field_data: 磁場數據
            
        Returns:
            建議的下一組參數
        """
        if len(self.bayesian_samples) < 5:
            # 初始隨機採樣
            parameters = self.random_sample_parameters()
        else:
            # 使用高斯過程預測
            X = np.array(self.bayesian_samples)
            y = np.array(self.bayesian_scores)
            
            self.gp_optimizer.fit(X, y)
            
            # 尋找最有希望的下一個點
            parameters = self.acquisition_function_optimization(experimental_data, field_data)
        
        return parameters
    
    def random_sample_parameters(self) -> np.ndarray:
        """
        隨機採樣參數
        
        Returns:
            隨機參數向量
        """
        parameters = []
        param_names = ['Ic_base', 'field_scale', 'phase_offset', 'asymmetry', 'noise_level']
        
        for param_name in param_names:
            low, high = self.parameter_bounds[param_name]
            if param_name in ['Ic_base', 'field_scale', 'noise_level']:
                # 對數尺度採樣
                parameters.append(np.exp(np.random.uniform(np.log(low), np.log(high))))
            else:
                # 線性尺度採樣
                parameters.append(np.random.uniform(low, high))
        
        return np.array(parameters)
    
    def acquisition_function_optimization(self, experimental_data: np.ndarray, 
                                        field_data: np.ndarray) -> np.ndarray:
        """
        優化採集函數以找到下一個採樣點
        
        Args:
            experimental_data: 實驗數據
            field_data: 磁場數據
            
        Returns:
            最優參數向量
        """
        def acquisition_function(parameters):
            parameters = parameters.reshape(1, -1)
            
            # 預測均值和標準差
            mean, std = self.gp_optimizer.predict(parameters, return_std=True)
            
            # Upper Confidence Bound (UCB)
            beta = 2.0  # 探索係數
            return -(mean + beta * std)  # 負號因為我們要最大化
        
        # 設置邊界
        bounds = [(self.parameter_bounds[param][0], self.parameter_bounds[param][1]) 
                 for param in ['Ic_base', 'field_scale', 'phase_offset', 'asymmetry', 'noise_level']]
        
        # 優化採集函數
        result = differential_evolution(acquisition_function, bounds, seed=42)
        
        return result.x
    
    def optimize_parameters(self, experimental_data: np.ndarray, 
                          field_data: np.ndarray,
                          initial_guess: Optional[np.ndarray] = None) -> Dict:
        """
        主要優化函數
        
        Args:
            experimental_data: 實驗數據
            field_data: 磁場數據
            initial_guess: 初始參數猜測
            
        Returns:
            優化結果字典
        """
        start_time = time.time()
        logger.info(f"開始參數優化，方法: {self.optimization_method}")
        
        best_correlation = -np.inf
        iteration_without_improvement = 0
        
        for iteration in range(self.max_iterations):
            if self.optimization_method == 'bayesian':
                parameters = self.bayesian_optimization_step(experimental_data, field_data)
            elif self.optimization_method == 'differential':
                parameters = self.differential_evolution_step(experimental_data, field_data)
            else:
                parameters = self.gradient_optimization_step(experimental_data, field_data, initial_guess)
            
            # 計算目標函數值
            score = self.objective_function(parameters, experimental_data, field_data)
            correlation = -score  # 轉換回正相關係數
            
            # 記錄結果
            if self.optimization_method == 'bayesian':
                self.bayesian_samples.append(parameters.copy())
                self.bayesian_scores.append(score)
            
            self.optimization_history.append({
                'iteration': iteration,
                'parameters': parameters.copy(),
                'correlation': correlation,
                'score': score,
                'timestamp': time.time()
            })
            
            # 檢查是否有改善
            if correlation > best_correlation + self.convergence_threshold:
                best_correlation = correlation
                self.best_parameters = parameters.copy()
                self.best_score = correlation
                iteration_without_improvement = 0
                
                logger.info(f"迭代 {iteration}: 新的最佳相關係數 = {correlation:.6f}")
                
                # 檢查是否達到目標
                if correlation >= self.target_correlation:
                    logger.info(f"達到目標相關係數 {self.target_correlation}")
                    break
            else:
                iteration_without_improvement += 1
            
            # 早期停止條件
            if iteration_without_improvement > 20:
                logger.info("20次迭代無改善，停止優化")
                break
        
        optimization_time = time.time() - start_time
        
        # 準備結果
        result = {
            'best_parameters': {
                'Ic_base': self.best_parameters[0],
                'field_scale': self.best_parameters[1],
                'phase_offset': self.best_parameters[2],
                'asymmetry': self.best_parameters[3],
                'noise_level': self.best_parameters[4]
            },
            'best_correlation': self.best_score,
            'total_iterations': len(self.optimization_history),
            'optimization_time': optimization_time,
            'convergence_achieved': self.best_score >= self.target_correlation,
            'optimization_history': self.optimization_history
        }
        
        logger.info(f"優化完成，最佳相關係數: {self.best_score:.6f}, 耗時: {optimization_time:.2f}秒")
        
        return result
    
    def differential_evolution_step(self, experimental_data: np.ndarray, 
                                  field_data: np.ndarray) -> np.ndarray:
        """
        差分進化優化步驟
        """
        bounds = [(self.parameter_bounds[param][0], self.parameter_bounds[param][1]) 
                 for param in ['Ic_base', 'field_scale', 'phase_offset', 'asymmetry', 'noise_level']]
        
        result = differential_evolution(
            lambda x: self.objective_function(x, experimental_data, field_data),
            bounds,
            maxiter=1,
            seed=42
        )
        
        return result.x
    
    def gradient_optimization_step(self, experimental_data: np.ndarray, 
                                 field_data: np.ndarray,
                                 initial_guess: Optional[np.ndarray] = None) -> np.ndarray:
        """
        梯度優化步驟
        """
        if initial_guess is None:
            initial_guess = self.random_sample_parameters()
        
        bounds = [(self.parameter_bounds[param][0], self.parameter_bounds[param][1]) 
                 for param in ['Ic_base', 'field_scale', 'phase_offset', 'asymmetry', 'noise_level']]
        
        result = minimize(
            lambda x: self.objective_function(x, experimental_data, field_data),
            initial_guess,
            method='L-BFGS-B',
            bounds=bounds
        )
        
        return result.x if result.success else initial_guess

def demo_real_time_optimization():
    """
    演示實時優化系統的使用
    """
    # 生成模擬實驗數據
    np.random.seed(42)
    field_data = np.linspace(-1, 1, 100)
    true_params = [1e-5, 2e-6, 1.5, 0.1, 1e-7]
    
    # 生成"實驗"數據
    experimental_data = RealTimeOptimizer().generate_josephson_response(
        field_data, *true_params
    )
    
    # 創建優化器
    optimizer = RealTimeOptimizer(
        target_correlation=0.8,
        max_iterations=50,
        optimization_method='bayesian'
    )
    
    # 執行優化
    result = optimizer.optimize_parameters(experimental_data, field_data)
    
    print("=== 實時優化結果 ===")
    print(f"最佳相關係數: {result['best_correlation']:.6f}")
    print(f"總迭代次數: {result['total_iterations']}")
    print(f"優化時間: {result['optimization_time']:.2f} 秒")
    print("最佳參數:")
    for param, value in result['best_parameters'].items():
        print(f"  {param}: {value:.6e}")
    
    return result

if __name__ == "__main__":
    demo_real_time_optimization()
