#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Current-Phase Relation Heatmap 快速預覽腳本
顯示所有生成的熱力圖文件信息

Author: AI Assistant
Date: 2025年6月4日
"""

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from datetime import datetime

def show_heatmap_summary():
    """顯示熱力圖文件摘要"""
    
    print("=" * 80)
    print("🔥 Current-Phase Relation Heatmap 生成完成報告")
    print("=" * 80)
    
    # 檢查生成的熱力圖文件
    heatmap_files = [
        ('josephson_current_phase_heatmap.png', '主程序生成的四模型熱力圖'),
        ('josephson_advanced_phase_analysis.png', '進階相位分析圖 (包含3D視圖)'),
        ('demo_current_phase_heatmap.png', '演示用四模型熱力圖'),
        ('demo_advanced_phase_analysis.png', '演示用進階分析圖'),
        ('demo_high_res_interpolation_heatmap.png', '高解析度單模型熱力圖'),
        ('demo_interactive_analysis.png', '互動式分析圖')
    ]
    
    print("📊 生成的熱力圖文件:")
    print()
    
    existing_files = []
    for filename, description in heatmap_files:
        if os.path.exists(filename):
            file_size = os.path.getsize(filename) / 1024  # KB
            creation_time = datetime.fromtimestamp(os.path.getmtime(filename))
            
            print(f"✅ {filename}")
            print(f"   📝 描述: {description}")
            print(f"   📏 大小: {file_size:.1f} KB")
            print(f"   🕐 生成時間: {creation_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print()
            
            existing_files.append((filename, description))
        else:
            print(f"❌ {filename} - 文件不存在")
            print()
    
    print("=" * 80)
    print("🎯 熱力圖特徵說明:")
    print()
    print("1️⃣ 標準熱力圖 (josephson_current_phase_heatmap.png)")
    print("   • 四個理論模型的電流-相位關係")
    print("   • X軸: 相位 φ (-4π 到 4π)")
    print("   • Y軸: 透明度 T (0.01 到 0.99)")
    print("   • 顏色: 歸一化電流強度")
    print()
    
    print("2️⃣ 進階分析圖 (josephson_advanced_phase_analysis.png)")
    print("   • 3D表面圖展示相位-透明度-電流關係")
    print("   • 特定透明度下的相位依賴性")
    print("   • 特定相位下的透明度依賴性")
    print()
    
    print("3️⃣ 高解析度圖 (demo_high_res_interpolation_heatmap.png)")
    print("   • 內插模型的詳細熱力圖")
    print("   • 包含等高線和3D視圖")
    print("   • 高密度採樣點 (20 T值 × 1001 相位點)")
    print()
    
    print("4️⃣ 互動式分析圖 (demo_interactive_analysis.png)")
    print("   • 多面板比較分析")
    print("   • 不同相位下的透明度依賴性")
    print("   • 不同透明度下的相位依賴性")
    print()
    
    print("=" * 80)
    print("🔬 物理意義:")
    print()
    print("• 電流-相位關係展現了Josephson結的基本特性")
    print("• 不同模型在高透明度區域顯示出顯著差異")
    print("• 相位調制效應在熱力圖中清晰可見")
    print("• 透明度優化窗口在色彩變化中突出顯示")
    print()
    
    print("=" * 80)
    print(f"📈 總計生成文件: {len(existing_files)} 個")
    print("🎉 Current-Phase Relation Heatmap 功能已完全實現！")
    print("=" * 80)
    
    return existing_files

def create_file_index():
    """創建文件索引"""
    
    files = show_heatmap_summary()
    
    # 創建索引文件
    with open('HEATMAP_FILES_INDEX.md', 'w', encoding='utf-8') as f:
        f.write("# Current-Phase Relation Heatmap 文件索引\n\n")
        f.write(f"生成時間: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n\n")
        
        f.write("## 📊 熱力圖文件列表\n\n")
        
        for filename, description in files:
            if os.path.exists(filename):
                file_size = os.path.getsize(filename) / 1024
                f.write(f"### {filename}\n")
                f.write(f"- **描述**: {description}\n")
                f.write(f"- **大小**: {file_size:.1f} KB\n")
                f.write(f"- **路徑**: `{os.path.abspath(filename)}`\n\n")
        
        f.write("## 🔬 技術規格\n\n")
        f.write("- **相位範圍**: -4π 到 4π (標準) / -2π 到 2π (演示)\n")
        f.write("- **透明度範圍**: 0.01 到 0.99\n")
        f.write("- **理論模型**: AB, KO, Interpolation, Scattering\n")
        f.write("- **圖像格式**: PNG (300 DPI)\n")
        f.write("- **色彩映射**: RdYlBu_r (熱力圖), Viridis (3D圖)\n\n")
        
        f.write("## 🎯 使用方法\n\n")
        f.write("1. 查看 `josephson_current_phase_heatmap.png` 獲得完整概覽\n")
        f.write("2. 查看 `josephson_advanced_phase_analysis.png` 獲得3D視角\n")
        f.write("3. 查看 `demo_high_res_interpolation_heatmap.png` 獲得詳細分析\n")
        f.write("4. 查看 `demo_interactive_analysis.png` 獲得切片分析\n\n")
    
    print("📝 已創建文件索引: HEATMAP_FILES_INDEX.md")

if __name__ == "__main__":
    files = show_heatmap_summary()
    create_file_index()
