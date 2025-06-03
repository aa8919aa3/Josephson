"""
統計評估工具

提供完整的模型統計評估功能，包括 R²、RMSE、模型比較等。
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from scipy import stats

# 設定 Plotly 為非互動模式，避免終端輸出問題
pio.renderers.default = "json"

class ModelStatistics:
    """
    模型統計評估類別
    
    提供完整的回歸分析統計指標計算和診斷。
    """
    
    def __init__(self, y_true, y_pred, n_params=None, model_name="Model"):
        """
        初始化統計評估
        
        Parameters:
        -----------
        y_true : array-like
            真實觀測值
        y_pred : array-like
            模型預測值
        n_params : int, optional
            模型參數數量
        model_name : str
            模型名稱
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.model_name = model_name
        
        # 移除 NaN 值
        self.valid_mask = ~(np.isnan(self.y_true) | np.isnan(self.y_pred))
        self.y_true_clean = self.y_true[self.valid_mask]
        self.y_pred_clean = self.y_pred[self.valid_mask]
        
        self.n = len(self.y_true_clean)
        self.n_params = n_params if n_params is not None else 1
        
        # 計算所有統計指標
        self._calculate_all_statistics()
    
    def _calculate_all_statistics(self):
        """計算所有統計指標"""
        if self.n == 0:
            self._set_invalid_stats()
            return
        
        # 基本統計量
        self.residuals = self.y_true_clean - self.y_pred_clean
        self.y_mean = np.mean(self.y_true_clean)
        
        # 平方和計算
        self.sse = np.sum(self.residuals**2)  # Sum of Squared Errors
        self.sst = np.sum((self.y_true_clean - self.y_mean)**2)  # Total Sum of Squares
        self.ssr = np.sum((self.y_pred_clean - self.y_mean)**2)  # Sum of Squares Regression
        
        # R-squared 相關指標
        self.r_squared = self._calculate_r_squared()
        self.adjusted_r_squared = self._calculate_adjusted_r_squared()
        
        # 誤差指標
        self.rmse = np.sqrt(self.sse / self.n)  # Root Mean Squared Error
        self.mae = np.mean(np.abs(self.residuals))  # Mean Absolute Error
        self.mse = self.sse / self.n  # Mean Squared Error
        self.mape = self._calculate_mape()  # Mean Absolute Percentage Error
        
        # 信息準則
        self.aic = self._calculate_aic()
        self.bic = self._calculate_bic()
        
        # 其他指標
        self.correlation = self._calculate_correlation()
        self.explained_variance = self._calculate_explained_variance()
    
    def _calculate_r_squared(self):
        """計算決定係數 (R-squared)"""
        if self.sst == 0:
            return 1.0 if self.sse < 1e-10 else 0.0
        return 1 - (self.sse / self.sst)
    
    def _calculate_adjusted_r_squared(self):
        """計算調整後R平方 (Adjusted R-squared)"""
        if self.n <= self.n_params:
            return np.nan
        
        if self.sst == 0:
            return 1.0 if self.sse < 1e-10 else 0.0
        
        return 1 - ((self.sse / (self.n - self.n_params - 1)) / (self.sst / (self.n - 1)))
    
    def _calculate_mape(self):
        """計算平均絕對百分比誤差"""
        non_zero_mask = np.abs(self.y_true_clean) > 1e-10
        if not np.any(non_zero_mask):
            return np.nan
        
        mape_values = np.abs(self.residuals[non_zero_mask] / self.y_true_clean[non_zero_mask]) * 100
        return np.mean(mape_values)
    
    def _calculate_aic(self):
        """計算 Akaike Information Criterion"""
        if self.n <= 0 or self.sse <= 0:
            return np.nan
        return self.n * np.log(self.sse / self.n) + 2 * self.n_params
    
    def _calculate_bic(self):
        """計算 Bayesian Information Criterion"""
        if self.n <= 0 or self.sse <= 0:
            return np.nan
        return self.n * np.log(self.sse / self.n) + self.n_params * np.log(self.n)
    
    def _calculate_correlation(self):
        """計算相關係數"""
        if len(self.y_true_clean) < 2:
            return np.nan
        return np.corrcoef(self.y_true_clean, self.y_pred_clean)[0, 1]
    
    def _calculate_explained_variance(self):
        """計算解釋變異數"""
        var_y = np.var(self.y_true_clean)
        if var_y == 0:
            return 1.0 if np.var(self.residuals) < 1e-10 else 0.0
        return 1 - np.var(self.residuals) / var_y
    
    def get_summary_dict(self):
        """返回統計摘要字典"""
        return {
            'model_name': self.model_name,
            'n_observations': self.n,
            'n_parameters': self.n_params,
            'r_squared': self.r_squared,
            'adjusted_r_squared': self.adjusted_r_squared,
            'rmse': self.rmse,
            'mae': self.mae,
            'mape': self.mape,
            'mse': self.mse,
            'sse': self.sse,
            'sst': self.sst,
            'ssr': self.ssr,
            'aic': self.aic,
            'bic': self.bic,
            'correlation': self.correlation,
            'explained_variance': self.explained_variance
        }
    
    def print_summary(self):
        """列印詳細統計摘要"""
        print(f"\n{'='*60}")
        print(f"📊 {self.model_name} 統計評估摘要")
        print(f"{'='*60}")
        
        print(f"🔢 基本信息:")
        print(f"   觀測值數量: {self.n:,}")
        print(f"   模型參數數量: {self.n_params}")
        
        print(f"\n📈 擬合品質指標:")
        print(f"   R-squared (決定係數): {self.r_squared:.6f}")
        print(f"   Adjusted R² (調整後R²): {self.adjusted_r_squared:.6f}")
        print(f"   相關係數: {self.correlation:.6f}")
        
        print(f"\n📏 誤差指標:")
        print(f"   RMSE (均方根誤差): {self.rmse:.6e}")
        print(f"   MAE (平均絕對誤差): {self.mae:.6e}")
        print(f"   MAPE (平均絕對百分比誤差): {self.mape:.2f}%")
        
        print(f"\n📊 平方和分解:")
        print(f"   SSE (誤差平方和): {self.sse:.6e}")
        print(f"   SSR (回歸平方和): {self.ssr:.6e}")
        print(f"   SST (總平方和): {self.sst:.6e}")
        
        print(f"\n🔍 模型選擇準則:")
        print(f"   AIC: {self.aic:.2f}")
        print(f"   BIC: {self.bic:.2f}")

def compare_multiple_models(*stats_objects, plot_comparison=True):
    """
    比較多個模型的統計指標
    
    Parameters:
    -----------
    *stats_objects : ModelStatistics objects
        要比較的統計物件
    plot_comparison : bool
        是否繪製比較圖表
    
    Returns:
    --------
    pd.DataFrame : 比較結果表格
    """
    if len(stats_objects) == 0:
        print("❌ 沒有提供任何模型進行比較")
        return None
    
    # 收集所有統計指標
    comparison_data = []
    for stats_obj in stats_objects:
        summary = stats_obj.get_summary_dict()
        comparison_data.append(summary)
    
    # 建立比較表格
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.set_index('model_name')
    
    # 列印比較結果
    print(f"\n{'='*80}")
    print(f"🏆 模型比較結果")
    print(f"{'='*80}")
    
    print("\n📊 主要擬合指標:")
    print(comparison_df[['r_squared', 'adjusted_r_squared', 'correlation']].round(6))
    
    print("\n📏 誤差指標:")
    print(comparison_df[['rmse', 'mae', 'mape']].round(6))
    
    print("\n🔍 模型選擇準則 (越小越好):")
    print(comparison_df[['aic', 'bic']].round(2))
    
    # 找出最佳模型
    best_models = {}
    metrics = ['r_squared', 'adjusted_r_squared', 'correlation']
    for metric in metrics:
        if metric in comparison_df.columns:
            best_idx = comparison_df[metric].idxmax()
            best_models[metric] = (best_idx, comparison_df.loc[best_idx, metric])
    
    error_metrics = ['rmse', 'mae', 'aic', 'bic']
    for metric in error_metrics:
        if metric in comparison_df.columns:
            best_idx = comparison_df[metric].idxmin()
            best_models[metric] = (best_idx, comparison_df.loc[best_idx, metric])
    
    print(f"\n🥇 各指標最佳模型:")
    for metric, (model_name, value) in best_models.items():
        print(f"   {metric.upper()}: {model_name} ({value:.6f})")
    
    return comparison_df

def plot_comprehensive_model_diagnostics(stats_obj, phi_ext=None, title_prefix=""):
    """
    繪製完整的模型診斷圖
    
    Parameters:
    -----------
    stats_obj : ModelStatistics
        統計物件
    phi_ext : array-like, optional
        外部磁通數據（用於 x 軸）
    title_prefix : str
        圖表標題前綴
    """
    
    # 建立子圖
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            '真實值 vs 預測值', '殘差 vs 預測值',
            '殘差直方圖', 'Q-Q 正態檢驗圖'
        )
    )
    
    # 1. 真實值 vs 預測值
    fig.add_trace(
        go.Scatter(x=stats_obj.y_true_clean, y=stats_obj.y_pred_clean,
                  mode='markers', name='數據點',
                  marker=dict(size=5, opacity=0.6, color='blue')),
        row=1, col=1
    )
    
    # 添加 y=x 參考線
    min_val = min(stats_obj.y_true_clean.min(), stats_obj.y_pred_clean.min())
    max_val = max(stats_obj.y_true_clean.max(), stats_obj.y_pred_clean.max())
    fig.add_trace(
        go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                  mode='lines', name='完美擬合線',
                  line=dict(color='red', dash='dash')),
        row=1, col=1
    )
    
    # 2. 殘差 vs 預測值
    fig.add_trace(
        go.Scatter(x=stats_obj.y_pred_clean, y=stats_obj.residuals,
                  mode='markers', name='殘差',
                  marker=dict(size=5, opacity=0.6, color='green')),
        row=1, col=2
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)
    
    # 3. 殘差直方圖
    fig.add_trace(
        go.Histogram(x=stats_obj.residuals, name='殘差分布',
                    nbinsx=30, opacity=0.7),
        row=2, col=1
    )
    
    # 4. Q-Q 圖
    if len(stats_obj.residuals) > 2:
        sorted_residuals = np.sort(stats_obj.residuals)
        n = len(sorted_residuals)
        theoretical_quantiles = stats.norm.ppf(np.arange(1, n+1) / (n+1))
        
        fig.add_trace(
            go.Scatter(x=theoretical_quantiles, y=sorted_residuals,
                      mode='markers', name='Q-Q 點',
                      marker=dict(size=4, opacity=0.6, color='purple')),
            row=2, col=2
        )
        
        # Q-Q 參考線
        qq_slope, qq_intercept = np.polyfit(theoretical_quantiles, sorted_residuals, 1)
        qq_line_y = qq_slope * theoretical_quantiles + qq_intercept
        fig.add_trace(
            go.Scatter(x=theoretical_quantiles, y=qq_line_y,
                      mode='lines', name='Q-Q 參考線',
                      line=dict(color='red', dash='dash')),
            row=2, col=2
        )
    
    # 更新佈局
    fig.update_layout(
        title_text=f"{title_prefix} 模型診斷分析 - {stats_obj.model_name}",
        height=600,
        showlegend=True
    )
    
    # 更新軸標題
    fig.update_xaxes(title_text="真實值", row=1, col=1)
    fig.update_yaxes(title_text="預測值", row=1, col=1)
    fig.update_xaxes(title_text="預測值", row=1, col=2)
    fig.update_yaxes(title_text="殘差", row=1, col=2)
    fig.update_xaxes(title_text="殘差", row=2, col=1)
    fig.update_yaxes(title_text="頻率", row=2, col=1)
    fig.update_xaxes(title_text="理論分位數", row=2, col=2)
    fig.update_yaxes(title_text="樣本分位數", row=2, col=2)
    
    fig.show()