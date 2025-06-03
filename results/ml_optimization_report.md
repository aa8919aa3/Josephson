# 機器學習優化分析報告

生成時間: 2025年6月3日

## 模型性能

### Random_Forest
- 訓練 R²: 0.7464
- 測試 R²: -1.1993
- 訓練 MSE: 0.001536
- 測試 MSE: 0.007735

### Neural_Network
- 訓練 R²: -0.6639
- 測試 R²: -9.4835
- 訓練 MSE: 0.010080
- 測試 MSE: 0.036871

## 特徵重要性

### Random_Forest
- field_mean: 0.2878
- gradient_mean: 0.2352
- primary_period_strength: 0.1493
- dominant_power: 0.0748
- primary_period: 0.0600
- cv: 0.0453
- mean: 0.0277
- local_variation_mean: 0.0236
- kurtosis: 0.0236
- gradient_std: 0.0192

## 建議

1. **特徵工程**: 基於重要性分析，重點關注高權重特徵
2. **模型選擇**: 選擇測試集表現最佳的模型進行預測
3. **參數優化**: 對低相關係數樣品進行重點優化
4. **數據擴充**: 考慮收集更多實驗數據以改善模型性能
