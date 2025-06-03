"""
磁通響應視覺化工具

提供 Josephson 結磁通響應分析的專用視覺化功能。
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# 設定 Plotly 為非互動模式，避免終端輸出問題
pio.renderers.default = "json"

def plot_flux_response(analyzer, model_type='both', show_fit=True):
    """
    繪製磁通響應曲線
    
    Parameters:
    -----------
    analyzer : JosephsonPeriodicAnalyzer
        分析器對象
    model_type : str
        模型類型
    show_fit : bool
        是否顯示擬合曲線
    """
    
    if not analyzer.simulation_results:
        print("❌ 沒有可用的模擬數據")
        return
    
    phi_ext = analyzer.simulation_results['phi_ext']
    
    # 創建圖表
    if model_type == 'both' and 'full_model' in analyzer.simulation_results and 'simplified_model' in analyzer.simulation_results:
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('完整非線性模型', '簡化正弦模型'),
            vertical_spacing=0.1
        )
        models = ['full_model', 'simplified_model']
        rows = [1, 2]
    else:
        fig = go.Figure()
        if model_type == 'both':
            models = [key for key in analyzer.simulation_results.keys() if key.endswith('_model')]
        else:
            models = [f'{model_type}_model']
        rows = [None] * len(models)
    
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, model_key in enumerate(models):
        if model_key not in analyzer.simulation_results:
            continue
            
        data = analyzer.simulation_results[model_key]
        row = rows[i] if rows[i] is not None else None
        
        # 理論曲線
        fig.add_trace(
            go.Scatter(
                x=phi_ext * 1e5,  # 轉換為 10^-5 單位
                y=data['I_theory'] * 1e6,  # 轉換為 μA
                mode='lines',
                name=f'{data["name"]} (理論)',
                line=dict(color=colors[i % len(colors)], width=2)
            ),
            row=row, col=1
        )
        
        # 含雜訊數據
        fig.add_trace(
            go.Scatter(
                x=phi_ext * 1e5,
                y=data['I_noisy'] * 1e6,
                mode='markers',
                name=f'{data["name"]} (測量)',
                marker=dict(size=3, opacity=0.6, color=colors[i % len(colors)]),
                error_y=dict(type='data', array=data['errors'] * 1e6, visible=True)
            ),
            row=row, col=1
        )
        
        # 擬合曲線（如果有）
        if show_fit and hasattr(analyzer, 'analysis_results') and model_key in analyzer.analysis_results:
            analysis = analyzer.analysis_results[model_key]
            if 'lomb_scargle' in analysis:
                ls_result = analysis['lomb_scargle']
                fig.add_trace(
                    go.Scatter(
                        x=phi_ext * 1e5,
                        y=ls_result['ls_model'] * 1e6,
                        mode='lines',
                        name=f'{data["name"]} (LS擬合)',
                        line=dict(color=colors[i % len(colors)], dash='dash', width=2)
                    ),
                    row=row, col=1
                )
    
    # 更新佈局
    fig.update_layout(
        title_text="Josephson 結磁通響應分析",
        height=600 if len(models) == 2 else 400,
        showlegend=True
    )
    
    # 更新軸標題
    if len(models) == 2:
        fig.update_xaxes(title_text="外部磁通 Φ_ext (×10⁻⁵)", row=2, col=1)
        fig.update_yaxes(title_text="電流 I_s (μA)", row=1, col=1)
        fig.update_yaxes(title_text="電流 I_s (μA)", row=2, col=1)
    else:
        fig.update_xaxes(title_text="外部磁通 Φ_ext (×10⁻⁵)")
        fig.update_yaxes(title_text="電流 I_s (μA)")
    
    fig.show()

def plot_period_analysis(analyzer, model_type='both'):
    """
    繪製週期分析結果
    
    Parameters:
    -----------
    analyzer : JosephsonPeriodicAnalyzer
        分析器對象
    model_type : str
        模型類型
    """
    
    if not hasattr(analyzer, 'analysis_results') or not analyzer.analysis_results:
        print("❌ 請先執行週期性分析")
        return
    
    models_to_plot = []
    if model_type == 'both':
        models_to_plot = [key for key in analyzer.analysis_results.keys() if key.endswith('_model')]
    else:
        model_key = f'{model_type}_model'
        if model_key in analyzer.analysis_results:
            models_to_plot = [model_key]
    
    if not models_to_plot:
        print("❌ 沒有可用的分析結果")
        return
    
    # 創建子圖
    fig = make_subplots(
        rows=len(models_to_plot), cols=2,
        subplot_titles=[f'{analyzer.analysis_results[model]["model_name"]} - Lomb-Scargle' for model in models_to_plot] +
                      [f'{analyzer.analysis_results[model]["model_name"]} - FFT' for model in models_to_plot],
        horizontal_spacing=0.1,
        vertical_spacing=0.15
    )
    
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, model_key in enumerate(models_to_plot):
        analysis = analyzer.analysis_results[model_key]
        row = i + 1
        
        # Lomb-Scargle 週期圖
        if 'lomb_scargle' in analysis:
            ls_result = analysis['lomb_scargle']
            
            fig.add_trace(
                go.Scatter(
                    x=ls_result['frequency'],
                    y=ls_result['power'],
                    mode='lines',
                    name=f'LS 功率譜',
                    line=dict(color=colors[i % len(colors)])
                ),
                row=row, col=1
            )
            
            # 標記最佳頻率
            fig.add_trace(
                go.Scatter(
                    x=[ls_result['best_frequency']],
                    y=[ls_result['best_power']],
                    mode='markers',
                    name=f'最佳頻率 ({ls_result["best_frequency"]:.2e})',
                    marker=dict(size=10, color='red', symbol='star')
                ),
                row=row, col=1
            )
        
        # FFT 分析
        if 'fft' in analysis:
            fft_result = analysis['fft']
            
            fig.add_trace(
                go.Scatter(
                    x=fft_result['frequencies'],
                    y=fft_result['power_spectrum'],
                    mode='lines',
                    name=f'FFT 功率譜',
                    line=dict(color=colors[i % len(colors)])
                ),
                row=row, col=2
            )
            
            # 標記主要頻率
            fig.add_trace(
                go.Scatter(
                    x=[fft_result['dominant_frequency']],
                    y=[fft_result['dominant_power']],
                    mode='markers',
                    name=f'主要頻率 ({fft_result["dominant_frequency"]:.2e})',
                    marker=dict(size=10, color='red', symbol='star')
                ),
                row=row, col=2
            )
    
    # 更新佈局
    fig.update_layout(
        title_text="週期性分析結果",
        height=300 * len(models_to_plot),
        showlegend=True
    )
    
    # 更新軸標題
    for i in range(len(models_to_plot)):
        row = i + 1
        fig.update_xaxes(title_text="頻率", row=row, col=1)
        fig.update_yaxes(title_text="LS 功率", row=row, col=1)
        fig.update_xaxes(title_text="頻率", row=row, col=2)
        fig.update_yaxes(title_text="FFT 功率", row=row, col=2)
    
    fig.show()

def plot_comprehensive_analysis(analyzer):
    """
    繪製完整的綜合分析結果
    
    Parameters:
    -----------
    analyzer : JosephsonPeriodicAnalyzer
        分析器對象
    """
    
    if not analyzer.simulation_results:
        print("❌ 沒有可用的數據")
        return
    
    print("🎨 生成綜合分析圖表")
    
    # 繪製磁通響應
    plot_flux_response(analyzer, model_type='both', show_fit=True)
    
    # 繪製週期分析
    if hasattr(analyzer, 'analysis_results') and analyzer.analysis_results:
        plot_period_analysis(analyzer, model_type='both')
    
    print("✅ 圖表生成完成")