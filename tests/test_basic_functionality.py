"""
基本功能測試

測試 Josephson 結分析工具的核心功能。
"""

import unittest
import numpy as np
import sys
import os

# 添加項目根目錄到路徑
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from josephson_analysis.models.josephson_physics import (
    JosephsonPeriodicAnalyzer,
    full_josephson_model,
    simplified_josephson_model
)

class TestJosephsonModels(unittest.TestCase):
    """測試 Josephson 模型"""
    
    def setUp(self):
        """設置測試參數"""
        self.phi_ext = np.linspace(-20e-5, 0, 100)
        self.params = {
            'Ic': 1.0e-6,
            'phi_0': np.pi/4,
            'f': 5e4,
            'T': 0.8,
            'k': -0.01,
            'r': 5e-3,
            'C': 10.0e-6,
            'd': -10.0e-3
        }
    
    def test_full_model(self):
        """測試完整模型"""
        current = full_josephson_model(self.phi_ext, **self.params)
        
        # 檢查輸出
        self.assertEqual(len(current), len(self.phi_ext))
        self.assertTrue(np.all(np.isfinite(current)))
        
        # 檢查週期性
        self.assertGreater(np.std(current), 1e-8)  # 應該有變化
    
    def test_simplified_model(self):
        """測試簡化模型"""
        simple_params = {k: v for k, v in self.params.items() if k != 'T'}
        current = simplified_josephson_model(self.phi_ext, **simple_params)
        
        # 檢查輸出
        self.assertEqual(len(current), len(self.phi_ext))
        self.assertTrue(np.all(np.isfinite(current)))

class TestJosephsonAnalyzer(unittest.TestCase):
    """測試分析器"""
    
    def