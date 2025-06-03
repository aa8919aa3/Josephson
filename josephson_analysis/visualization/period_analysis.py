"""
Period Analysis Visualization Functions

This module provides plotting functions for periodicity analysis
including FFT, Lomb-Scargle, and autocorrelation analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional, Tuple, Dict, Any


def plot_period_spectrum(frequency: np.ndarray, power: np.ndarray,
                        method: str = "FFT",
                        peak_frequencies: Optional[np.ndarray] = None,
                        title: Optional[str] = None,
                        interactive: bool = True) -> Any:
    """
    Plot power spectrum from periodicity analysis
    
    Parameters:
    -----------
    frequency : np.ndarray
        Frequency values
    power : np.ndarray
        Power spectrum values
    method : str
        Analysis method ("FFT", "Lomb-Scargle", etc.)
    peak_frequencies : Optional[np.ndarray]
        Detected peak frequencies to highlight
    title : Optional[str]
        Plot title
    interactive : bool
        Whether to use plotly (True) or matplotlib (False)
        
    Returns:
    --------
    Figure object
    """
    if title is None:
        title = f"{method} Power Spectrum"
    
    if interactive:
        return _plot_spectrum_plotly(frequency, power, method, peak_frequencies, title)
    else:
        return _plot_spectrum_matplotlib(frequency, power, method, peak_frequencies, title)


def _plot_spectrum_plotly(frequency: np.ndarray, power: np.ndarray,
                         method: str, peak_frequencies: Optional[np.ndarray],
                         title: str) -> go.Figure:
    """Plotly implementation of spectrum plot"""
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=frequency, y=power,
                           mode='lines',
                           name=f'{method} Spectrum',
                           line=dict(color='blue', width=1.5)))
    
    if peak_frequencies is not None:
        # Highlight detected peaks
        peak_powers = np.interp(peak_frequencies, frequency, power)
        fig.add_trace(go.Scatter(x=peak_frequencies, y=peak_powers,
                               mode='markers',
                               name='Detected Peaks',
                               marker=dict(color='red', size=8, symbol='diamond')))
    
    fig.update_layout(
        title=title,
        xaxis_title="Frequency (Hz⁻¹)",
        yaxis_title="Power",
        hovermode='x unified',
        xaxis=dict(type='log' if method == "Lomb-Scargle" else 'linear')
    )
    
    return fig


def _plot_spectrum_matplotlib(frequency: np.ndarray, power: np.ndarray,
                             method: str, peak_frequencies: Optional[np.ndarray],
                             title: str) -> plt.Figure:
    """Matplotlib implementation of spectrum plot"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(frequency, power, 'b-', linewidth=1.5, label=f'{method} Spectrum')
    
    if peak_frequencies is not None:
        peak_powers = np.interp(peak_frequencies, frequency, power)
        ax.scatter(peak_frequencies, peak_powers, color='red', s=50, 
                  marker='D', label='Detected Peaks', zorder=5)
    
    if method == "Lomb-Scargle":
        ax.set_xscale('log')
    
    ax.set_title(title)
    ax.set_xlabel('Frequency (Hz⁻¹)')
    ax.set_ylabel('Power')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_lomb_scargle_periodogram(frequency: np.ndarray, power: np.ndarray,
                                 false_alarm_levels: Optional[Dict[str, float]] = None,
                                 title: str = "Lomb-Scargle Periodogram") -> go.Figure:
    """
    Plot Lomb-Scargle periodogram with false alarm probability levels
    
    Parameters:
    -----------
    frequency : np.ndarray
        Frequency values
    power : np.ndarray
        Lomb-Scargle power values
    false_alarm_levels : Optional[Dict[str, float]]
        False alarm probability levels {label: power_threshold}
    title : str
        Plot title
        
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=frequency, y=power,
                           mode='lines',
                           name='Lomb-Scargle Power',
                           line=dict(color='blue', width=1.5)))
    
    if false_alarm_levels is not None:
        colors = ['red', 'orange', 'green']
        for i, (label, threshold) in enumerate(false_alarm_levels.items()):
            color = colors[i % len(colors)]
            fig.add_hline(y=threshold, line_dash="dash", line_color=color,
                         annotation_text=f"FAP {label}")
    
    fig.update_layout(
        title=title,
        xaxis_title="Frequency (Hz⁻¹)",
        yaxis_title="Lomb-Scargle Power",
        xaxis=dict(type='log'),
        hovermode='x unified'
    )
    
    return fig


def plot_autocorrelation_analysis(lag: np.ndarray, autocorr: np.ndarray,
                                 period_estimate: Optional[float] = None,
                                 title: str = "Autocorrelation Analysis") -> go.Figure:
    """
    Plot autocorrelation function for period detection
    
    Parameters:
    -----------
    lag : np.ndarray
        Lag values
    autocorr : np.ndarray
        Autocorrelation values
    period_estimate : Optional[float]
        Estimated period to highlight
    title : str
        Plot title
        
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=lag, y=autocorr,
                           mode='lines',
                           name='Autocorrelation',
                           line=dict(color='purple', width=2)))
    
    if period_estimate is not None:
        # Add vertical line at estimated period
        fig.add_vline(x=period_estimate, line_dash="dash", line_color="red",
                     annotation_text=f"Period: {period_estimate:.3e}")
    
    fig.update_layout(
        title=title,
        xaxis_title="Lag",
        yaxis_title="Autocorrelation",
        hovermode='x unified'
    )
    
    return fig


def plot_parameter_fitting_results(phi_ext: np.ndarray, current: np.ndarray,
                                  fitted_current: np.ndarray,
                                  fit_params: Dict[str, float],
                                  title: str = "Parameter Fitting Results") -> go.Figure:
    """
    Plot fitting results with parameter information
    
    Parameters:
    -----------
    phi_ext : np.ndarray
        External flux values
    current : np.ndarray
        Experimental current values
    fitted_current : np.ndarray
        Fitted current values
    fit_params : Dict[str, float]
        Fitted parameters
    title : str
        Plot title
        
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    fig = make_subplots(rows=2, cols=1,
                       subplot_titles=('Data and Fit', 'Residuals'),
                       row_heights=[0.7, 0.3],
                       shared_xaxes=True)
    
    # Main plot
    fig.add_trace(go.Scatter(x=phi_ext, y=current,
                           mode='markers',
                           name='Experimental Data',
                           marker=dict(size=4, color='blue', opacity=0.7)),
                 row=1, col=1)
    
    fig.add_trace(go.Scatter(x=phi_ext, y=fitted_current,
                           mode='lines',
                           name='Fitted Model',
                           line=dict(color='red', width=2)),
                 row=1, col=1)
    
    # Residuals
    residuals = current - fitted_current
    fig.add_trace(go.Scatter(x=phi_ext, y=residuals,
                           mode='markers',
                           name='Residuals',
                           marker=dict(size=3, color='green')),
                 row=2, col=1)
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    
    # Add parameter information as annotation
    param_text = "<br>".join([f"{key}: {value:.3e}" for key, value in fit_params.items()])
    fig.add_annotation(
        text=param_text,
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="gray",
        borderwidth=1
    )
    
    fig.update_layout(
        title=title,
        height=600
    )
    
    fig.update_xaxes(title_text="External Flux Φ_ext", row=2, col=1)
    fig.update_yaxes(title_text="Current I_s (A)", row=1, col=1)
    fig.update_yaxes(title_text="Residuals (A)", row=2, col=1)
    
    return fig


def plot_parameter_sweep(param_values: np.ndarray, metric_values: np.ndarray,
                        param_name: str, metric_name: str,
                        optimal_value: Optional[float] = None,
                        title: Optional[str] = None) -> go.Figure:
    """
    Plot parameter sweep results
    
    Parameters:
    -----------
    param_values : np.ndarray
        Parameter values swept
    metric_values : np.ndarray
        Corresponding metric values (e.g., R², RMSE)
    param_name : str
        Name of parameter being swept
    metric_name : str
        Name of metric being plotted
    optimal_value : Optional[float]
        Optimal parameter value to highlight
    title : Optional[str]
        Plot title
        
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    if title is None:
        title = f"{param_name} Parameter Sweep"
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=param_values, y=metric_values,
                           mode='lines+markers',
                           name=f'{metric_name} vs {param_name}',
                           line=dict(color='blue', width=2),
                           marker=dict(size=5)))
    
    if optimal_value is not None:
        optimal_metric = np.interp(optimal_value, param_values, metric_values)
        fig.add_trace(go.Scatter(x=[optimal_value], y=[optimal_metric],
                               mode='markers',
                               name='Optimal Value',
                               marker=dict(color='red', size=10, symbol='star')))
    
    fig.update_layout(
        title=title,
        xaxis_title=param_name,
        yaxis_title=metric_name,
        hovermode='x unified'
    )
    
    return fig
