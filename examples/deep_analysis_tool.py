#!/usr/bin/env python3
"""
Deep Analysis Tool: Specialized analysis of best-performing experimental vs simulated data comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from josephson_analysis.utils.lmfit_tools import curve_fit_compatible
from scipy.stats import pearsonr
import json

# Configure matplotlib with English fonts only
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 100

def josephson_fit_function(y_field, Ic_base, field_scale, phase_offset, asymmetry, background):
    """
    Josephson junction function for fitting
    """
    flux_quantum = 2.067e-15
    phi_ext = y_field * field_scale
    normalized_flux = np.pi * phi_ext / flux_quantum
    
    with np.errstate(divide='ignore', invalid='ignore'):
        sinc_term = np.sin(normalized_flux + phase_offset) / (normalized_flux + 1e-10)
        sinc_term = np.where(np.abs(normalized_flux) < 1e-10, 1.0, sinc_term)
    
    pattern = np.abs(sinc_term) * (1 + asymmetry * np.cos(2 * normalized_flux))
    return Ic_base * pattern + background

def analyze_single_file_deeply(exp_file, sim_file, output_dir):
    """
    Deep analysis of single file experimental vs simulated data
    """
    # Load data
    exp_data = pd.read_csv(exp_file)
    sim_data = pd.read_csv(sim_file)
    
    filename = exp_file.name
    
    exp_y = exp_data['y_field'].values
    exp_Ic = exp_data['Ic'].values
    sim_y = sim_data['y_field'].values
    sim_Ic = sim_data['Ic'].values
    
    # Data alignment
    if len(exp_y) != len(sim_y):
        min_len = min(len(exp_y), len(sim_y))
        exp_y, exp_Ic = exp_y[:min_len], exp_Ic[:min_len]
        sim_y, sim_Ic = sim_y[:min_len], sim_Ic[:min_len]
    
    # Create analysis charts
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(f'Deep Analysis: {filename}', fontsize=16, fontweight='bold')
    
    # 1. Raw data comparison
    ax = axes[0, 0]
    ax.plot(exp_y, exp_Ic, 'b.-', label='Experimental Data', alpha=0.7, markersize=3)
    ax.plot(sim_y, sim_Ic, 'r.-', label='Simulated Data', alpha=0.7, markersize=3)
    ax.set_xlabel('y_field')
    ax.set_ylabel('Ic (A)')
    ax.set_title('Raw Data Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Normalized data comparison
    ax = axes[0, 1]
    exp_norm = (exp_Ic - np.mean(exp_Ic)) / np.std(exp_Ic)
    sim_norm = (sim_Ic - np.mean(sim_Ic)) / np.std(sim_Ic)
    ax.plot(exp_y, exp_norm, 'b.-', label='Experimental (Normalized)', alpha=0.7, markersize=3)
    ax.plot(sim_y, sim_norm, 'r.-', label='Simulated (Normalized)', alpha=0.7, markersize=3)
    ax.set_xlabel('y_field')
    ax.set_ylabel('Normalized Ic')
    ax.set_title('Normalized Data Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Scatter plot and correlation
    ax = axes[0, 2]
    correlation, p_value = pearsonr(exp_Ic, sim_Ic)
    ax.scatter(exp_Ic, sim_Ic, alpha=0.6, s=20)
    ax.plot([min(exp_Ic), max(exp_Ic)], [min(exp_Ic), max(exp_Ic)], 'r--', alpha=0.8)
    ax.set_xlabel('Experimental Ic (A)')
    ax.set_ylabel('Simulated Ic (A)')
    ax.set_title(f'Correlation Analysis\\nr = {correlation:.4f}, p = {p_value:.4f}')
    ax.grid(True, alpha=0.3)
    
    # 4. Residual analysis
    ax = axes[1, 0]
    residuals = sim_Ic - exp_Ic
    ax.plot(exp_y, residuals, 'g.-', alpha=0.7, markersize=3)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.8)
    ax.fill_between(exp_y, residuals, alpha=0.3, color='green')
    ax.set_xlabel('y_field')
    ax.set_ylabel('Residuals (Simulated - Experimental)')
    ax.set_title('Residual Analysis')
    ax.grid(True, alpha=0.3)
    
    # 5. Spectral analysis (FFT)
    ax = axes[1, 1]
    exp_fft = np.abs(np.fft.fft(exp_Ic - np.mean(exp_Ic)))
    sim_fft = np.abs(np.fft.fft(sim_Ic - np.mean(sim_Ic)))
    freqs = np.fft.fftfreq(len(exp_Ic))
    valid_idx = freqs > 0
    ax.semilogy(freqs[valid_idx], exp_fft[valid_idx], 'b-', label='Experimental', alpha=0.7)
    ax.semilogy(freqs[valid_idx], sim_fft[valid_idx], 'r-', label='Simulated', alpha=0.7)
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Amplitude')
    ax.set_title('Spectral Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Distribution histogram
    ax = axes[1, 2]
    ax.hist(exp_Ic, bins=20, alpha=0.6, label='Experimental', density=True, color='blue')
    ax.hist(sim_Ic, bins=20, alpha=0.6, label='Simulated', density=True, color='red')
    ax.set_xlabel('Ic (A)')
    ax.set_ylabel('Density')
    ax.set_title('Value Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 7. Local correlation analysis
    ax = axes[2, 0]
    window_size = max(10, len(exp_Ic) // 10)
    local_corr = []
    x_centers = []
    for i in range(window_size, len(exp_Ic) - window_size):
        start_idx = i - window_size // 2
        end_idx = i + window_size // 2
        local_r, _ = pearsonr(exp_Ic[start_idx:end_idx], sim_Ic[start_idx:end_idx])
        local_corr.append(local_r)
        x_centers.append(exp_y[i])
    
    ax.plot(x_centers, local_corr, 'purple', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('y_field')
    ax.set_ylabel('Local Correlation Coefficient')
    ax.set_title('Local Correlation Variation')
    ax.grid(True, alpha=0.3)
    
    # 8. Fitting analysis
    ax = axes[2, 1]
    try:
        # Fit experimental data using lmfit (L-BFGS-B)
        popt_exp, _ = curve_fit_compatible(josephson_fit_function, exp_y, exp_Ic, 
                                        p0=[np.mean(exp_Ic), 1e-10, 0, 0, 0],
                                        maxfev=2000)
        exp_fit = josephson_fit_function(exp_y, *popt_exp)
        
        # Fit simulated data using lmfit (L-BFGS-B)
        popt_sim, _ = curve_fit_compatible(josephson_fit_function, sim_y, sim_Ic,
                                        p0=[np.mean(sim_Ic), 1e-10, 0, 0, 0],
                                        maxfev=2000)
        sim_fit = josephson_fit_function(sim_y, *popt_sim)
        
        ax.plot(exp_y, exp_Ic, 'b.', label='Experimental Data', alpha=0.6, markersize=3)
        ax.plot(exp_y, exp_fit, 'b-', label='Experimental Fit', linewidth=2)
        ax.plot(sim_y, sim_Ic, 'r.', label='Simulated Data', alpha=0.6, markersize=3)
        ax.plot(sim_y, sim_fit, 'r-', label='Simulated Fit', linewidth=2)
        
        # Calculate R²
        exp_r2 = 1 - np.sum((exp_Ic - exp_fit)**2) / np.sum((exp_Ic - np.mean(exp_Ic))**2)
        sim_r2 = 1 - np.sum((sim_Ic - sim_fit)**2) / np.sum((sim_Ic - np.mean(sim_Ic))**2)
        
        ax.set_title(f'Josephson Fitting\\nExp R² = {exp_r2:.4f}, Sim R² = {sim_r2:.4f}')
        
    except Exception as e:
        ax.text(0.5, 0.5, f'Fitting Failed:\\n{str(e)}', transform=ax.transAxes, 
                ha='center', va='center')
        ax.set_title('Josephson Fitting (Failed)')
    
    ax.set_xlabel('y_field')
    ax.set_ylabel('Ic (A)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 9. Statistical summary
    ax = axes[2, 2]
    ax.axis('off')
    
    stats_text = f"""Statistical Summary:
    
            Experimental Data:
            Mean: {np.mean(exp_Ic):.2e} A
            Std Dev: {np.std(exp_Ic):.2e} A
            CV: {np.std(exp_Ic)/np.mean(exp_Ic):.4f}
            Range: [{np.min(exp_Ic):.2e}, {np.max(exp_Ic):.2e}]

            Simulated Data:
            Mean: {np.mean(sim_Ic):.2e} A
            Std Dev: {np.std(sim_Ic):.2e} A
            CV: {np.std(sim_Ic)/np.mean(sim_Ic):.4f}
            Range: [{np.min(sim_Ic):.2e}, {np.max(sim_Ic):.2e}]

            Comparison:
            Correlation: {correlation:.4f}
            p-value: {p_value:.4f}
            Mean ratio: {np.mean(sim_Ic)/np.mean(exp_Ic):.4f}
            Std ratio: {np.std(sim_Ic)/np.std(exp_Ic):.4f}
            """
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save chart
    output_file = output_dir / f"deep_analysis_{filename.replace('.csv', '.png')}"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Deep analysis chart saved to: {output_file}")
    
    plt.show()
    
    return {
        'correlation': correlation,
        'p_value': p_value,
        'exp_stats': {
            'mean': float(np.mean(exp_Ic)),
            'std': float(np.std(exp_Ic)),
            'cv': float(np.std(exp_Ic)/np.mean(exp_Ic))
        },
        'sim_stats': {
            'mean': float(np.mean(sim_Ic)),
            'std': float(np.std(sim_Ic)),
            'cv': float(np.std(sim_Ic)/np.mean(sim_Ic))
        }
    }

def main():
    """Main execution function"""
    print("=== Josephson Junction Deep Analysis Tool ===\\n")
    
    # Setup paths
    exp_data_dir = Path("/Users/albert-mac/Code/GitHub/Josephson/data/experimental")
    sim_data_dir = Path("/Users/albert-mac/Code/GitHub/Josephson/data/simulated")
    results_dir = Path("/Users/albert-mac/Code/GitHub/Josephson/results")
    results_dir.mkdir(exist_ok=True)
    
    # Load improvement results to find best file
    results_file = results_dir / "improved_simulation_results.json"
    if results_file.exists():
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # Find file with highest correlation
        valid_results = {k: v for k, v in results.items() if not np.isnan(v['correlation'])}
        if valid_results:
            best_file = max(valid_results.items(), key=lambda x: x[1]['correlation'])
            filename = best_file[0]
            print(f"Analyzing best performing file: {filename} (correlation: {best_file[1]['correlation']:.4f})")
        else:
            filename = "317Ic.csv"  # Fallback
            print(f"Using fallback file for analysis: {filename}")
    else:
        filename = "317Ic.csv"  # Default
        print(f"Using default file for analysis: {filename}")
    
    # Execute deep analysis
    exp_file = exp_data_dir / filename
    sim_file = sim_data_dir / f"improved_sim_{filename}"
    
    if exp_file.exists() and sim_file.exists():
        analysis_results = analyze_single_file_deeply(exp_file, sim_file, results_dir)
        
        # Save analysis results
        analysis_file = results_dir / f"deep_analysis_{filename.replace('.csv', '.json')}"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)
        print(f"\\nAnalysis results saved to: {analysis_file}")
        
    else:
        print(f"Error: Cannot find files {exp_file} or {sim_file}")

if __name__ == "__main__":
    main()
