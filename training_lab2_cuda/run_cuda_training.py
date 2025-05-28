#!/usr/bin/env python3
"""
Cuda 優化的服裝風格分類訓練腳本
使用方法:
    python3 run_cuda_training.py --config balanced
    python3 run_cuda_training.py --config minimal
    python3 run_cuda_training.py --config performance
"""

import argparse
import sys
import os
from cuda_train_config import CudaTrainingConfig, QuickConfigs

def check_dataset(data_root):
    """檢查數據集是否存在"""
    if not os.path.exists(data_root):
        print(f"❌ 數據集目錄不存在: {data_root}")
        print("請確保數據集目錄存在並包含正確的子目錄結構")
        return False
    
    # 檢查是否有風格目錄
    style_categories = [
        'Artsy', 'Athleisure', 'BRITISH', 'CASUAL', 'GOTH', 'Japanese',
        'Kawaii', 'Korean', 'MINIMALIST', 'Preppy', 'STREET', 'Streetwear', 
        'Vintage', 'Y2K'
    ]
    
    gender_categories = ['MEN', 'WOMEN']
    
    found_dirs = 0
    for style in style_categories:
        for gender in gender_categories:
            folder_name = f"{style}_{gender}"
            folder_path = os.path.join(data_root, folder_name)
            if os.path.exists(folder_path):
                found_dirs += 1
    
    if found_dirs == 0:
        print(f"❌ 在 {data_root} 中沒有找到任何風格目錄")
        print("預期的目錄格式: Style_Gender (例如: CASUAL_MEN, Kawaii_WOMEN)")
        return False
    
    print(f"✅ 找到 {found_dirs} 個風格目錄")
    return True

def main():
    parser = argparse.ArgumentParser(description='Cuda 優化的服裝風格分類訓練')
    parser.add_argument('--config', type=str, default='balanced',
                       choices=['minimal', 'balanced', 'performance'],
                       help='選擇配置 (default: choosing suitable config based on hardware spec)')
    parser.add_argument('--data-root', type=str, default='./dataset',
                       help='數據集根目錄 (default: ./dataset)')
    parser.add_argument('--resume', type=str, default=None,
                       help='從檢查點恢復訓練的路徑')
    parser.add_argument('--dry-run', action='store_true',
                       help='只檢查配置，不實際訓練')
    
    args = parser.parse_args()
    
    print("🟩 Cuda 服裝風格分類訓練器")
    print("=" * 50)
    
    
    # 檢查數據集
    if not check_dataset(args.data_root):
        sys.exit(1)
    
    # 選擇配置
    if not hasattr(args, 'config'):
        # 根據設備自動調整
        config = CudaTrainingConfig.get_device_specific_config()
        
    elif args.config == 'minimal':
        config = QuickConfigs.minimal()
        print("📱 使用最小配置 (適合低記憶體設備)")
    elif args.config == 'performance':
        config = QuickConfigs.performance()
        print("🚀 使用性能配置 (適合高端設備)")
    elif args.config == 'balanced':
        config = QuickConfigs.balanced()
        print("⚖️ 使用平衡配置 (推薦)")
    
    config.DATA_ROOT = args.data_root
    config.print_config()
    
    if args.dry_run:
        print("🔍 乾運行模式 - 配置檢查完成")
        return
    
    # 導入訓練模組
    try:
        from test_cuda_optimized import main as main_train_function
    except ImportError as e:
        print(f"❌ 導入訓練模組失敗: {e}")
        sys.exit(1)
    
    main_train_function(config, args.resume)
    
if __name__ == "__main__":
    main() 