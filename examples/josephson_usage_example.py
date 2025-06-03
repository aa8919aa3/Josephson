# Create analyzer instance
analyzer = JosephsonAnalyzer(save_data=True)

# Generate data for two models
print("🚀 Generating Josephson junction simulation data")

# 完整模型
full_data = analyzer.generate_josephson_data(
    model_type="full",
    Ic=1.0,
    f=0.5,
    T=0.8,
    noise_level=0.05,
    n_points=500
)

# 簡化模型
simplified_data = analyzer.generate_josephson_data(
    model_type="simplified",
    Ic=1.0,
    f=0.5,
    noise_level=0.05,
    n_points=500
)

# 執行 Lomb-Scargle 分析
print("\n🔬 執行 Lomb-Scargle 分析")
full_analysis = analyzer.analyze_with_lomb_scargle("full")
simplified_analysis = analyzer.analyze_with_lomb_scargle("simplified")

# 繪製完整分析結果
print("\n📊 繪製分析結果")
analyzer.plot_comprehensive_analysis("full")
analyzer.plot_comprehensive_analysis("simplified")

# 執行自定義模型擬合
print("\n🔧 執行自定義模型擬合")
full_fit = analyzer.fit_custom_model("full", use_true_model=True)
simplified_fit = analyzer.fit_custom_model("simplified", use_true_model=True)

# 比較模型統計
if full_analysis and simplified_analysis:
    print("\n🏆 模型比較")
    comparison = compare_multiple_models(
        full_analysis['statistics'],
        simplified_analysis['statistics'],
        plot_comparison=True
    )

# 詳細統計報告
if full_analysis:
    print("\n📈 完整模型詳細統計")
    full_analysis['statistics'].print_summary()
    plot_comprehensive_model_diagnostics(
        full_analysis['statistics'], 
        times=full_data['Phi_ext'],
        title_prefix="完整 Josephson 模型"
    )

if simplified_analysis:
    print("\n📈 簡化模型詳細統計")
    simplified_analysis['statistics'].print_summary()
    plot_comprehensive_model_diagnostics(
        simplified_analysis['statistics'], 
        times=simplified_data['Phi_ext'],
        title_prefix="簡化 Josephson 模型"
    )