def analyze_parameter_accuracy(analyzer):
    """
    Analyze the accuracy of parameter estimation
    """
    print("\n🎯 Parameter Estimation Accuracy Analysis")
    print("="*60)
    
    for model_type in ['full', 'simplified']:
        if model_type in analyzer.simulation_results and model_type in analyzer.analysis_results:
            true_params = analyzer.simulation_results[model_type]['parameters']
            analysis = analyzer.analysis_results[model_type]
            
            print(f"\n📊 {analyzer.simulation_results[model_type]['model_name']}")
            print("-" * 40)
            
            # Frequency estimation
            true_freq = true_params['f']
            est_freq = analysis['best_frequency']
            freq_error = abs(est_freq - true_freq) / true_freq * 100
            
            print(f"Frequency Estimation:")
            print(f"  真實值: {true_freq:.6f}")
            print(f"  估計值: {est_freq:.6f}")
            print(f"  相對誤差: {freq_error:.2f}%")
            
            # 振幅估計（對於簡化模型）
            if model_type == 'simplified':
                true_amp = true_params['Ic']
                est_amp = analysis['amplitude']
                amp_error = abs(est_amp - true_amp) / true_amp * 100
                
                print(f"振幅估計:")
                print(f"  真實值: {true_amp:.6f}")
                print(f"  估計值: {est_amp:.6f}")
                print(f"  相對誤差: {amp_error:.2f}%")
            
            # 統計品質
            stats = analysis['statistics']
            print(f"統計品質:")
            print(f"  R²: {stats.r_squared:.6f}")
            print(f"  RMSE: {stats.rmse:.6e}")
            print(f"  MAE: {stats.mae:.6e}")

# 執行參數準確性分析
analyze_parameter_accuracy(analyzer)