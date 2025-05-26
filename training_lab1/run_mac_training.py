#!/usr/bin/env python3
"""
Mac 優化的服裝風格分類訓練腳本
使用方法:
    python run_mac_training.py --config balanced
    python run_mac_training.py --config minimal
    python run_mac_training.py --config performance
"""

import argparse
import sys
import os
from mac_train_config import MacTrainingConfig, QuickConfigs



def check_dataset(data_root):
    """檢查數據集是否存在"""
    if not os.path.exists(data_root):
        print(f"❌ 數據集目錄不存在: {data_root}")
        print("請確保數據集目錄存在並包含正確的子目錄結構")
        return False
    
    # 檢查是否有風格目錄
    style_categories = [
        'Athleisure', 'BRITISH', 'CASUAL', 'GOTH', 'Kawaii', 
        'Korean', 'MINIMALIST', 'Preppy', 'STREET', 'Streetwear', 
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
    parser = argparse.ArgumentParser(description='Mac 優化的服裝風格分類訓練')
    parser.add_argument('--config', type=str, default='balanced',
                       choices=['minimal', 'balanced', 'performance'],
                       help='選擇配置預設 (default: balanced)')
    parser.add_argument('--data-root', type=str, default='./dataset',
                       help='數據集根目錄 (default: ./dataset)')
    parser.add_argument('--resume', type=str, default=None,
                       help='從檢查點恢復訓練的路徑')
    parser.add_argument('--dry-run', action='store_true',
                       help='只檢查配置，不實際訓練')
    
    args = parser.parse_args()
    
    print("🍎 Mac 服裝風格分類訓練器")
    print("=" * 50)
    
    
    # 檢查數據集
    if not check_dataset(args.data_root):
        sys.exit(1)
    
    # 選擇配置
    if args.config == 'minimal':
        config = QuickConfigs.minimal()
        print("📱 使用最小配置 (適合低記憶體設備)")
    elif args.config == 'performance':
        config = QuickConfigs.performance()
        print("🚀 使用性能配置 (適合高端設備)")
    else:
        config = QuickConfigs.balanced()
        print("⚖️  使用平衡配置 (推薦)")
    
    # 根據設備自動調整
    config = config.get_device_specific_config()
    config.DATA_ROOT = args.data_root
    config.print_config()
    
    if args.dry_run:
        print("🔍 乾運行模式 - 配置檢查完成")
        return
    
    # 導入訓練模組
    try:
        from test_mac_optimized import (
            setup_mac_optimization, OptimizedOutfitDataset, 
            LightweightStyleClassifier, MacOptimizedTrainer
        )
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        import torchvision.transforms as transforms
        import gc
    except ImportError as e:
        print(f"❌ 導入訓練模組失敗: {e}")
        sys.exit(1)
    
    # 開始訓練
    print("\n🚀 開始訓練...")
    
    # 設備設置
    device = setup_mac_optimization()
    
    # 數據增強
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=config.HORIZONTAL_FLIP_PROB),
        transforms.ColorJitter(
            brightness=config.COLOR_JITTER_BRIGHTNESS,
            contrast=config.COLOR_JITTER_CONTRAST
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 建立數據集
    train_dataset = OptimizedOutfitDataset(
        config.DATA_ROOT,
        transform=train_transform,
        max_samples_per_class=config.MAX_SAMPLES_PER_CLASS
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=config.DROP_LAST,
        persistent_workers=config.PERSISTENT_WORKERS
    )
    
    # 建立模型
    model = LightweightStyleClassifier(
        num_styles=config.NUM_STYLES,
        num_genders=config.NUM_GENDERS,
        feature_dim=config.FEATURE_DIM
    )
    model.to(device)
    
    # 計算參數數量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📈 模型參數: {total_params:,} (可訓練: {trainable_params:,})")
    
    # 訓練器
    trainer = MacOptimizedTrainer(model, device)
    
    # 優化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        betas=config.OPTIMIZER_BETAS
    )
    
    # 學習率調度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.NUM_EPOCHS,
        eta_min=config.LEARNING_RATE * config.SCHEDULER_ETA_MIN_RATIO
    )
    
    # 恢復訓練
    start_epoch = 0
    best_loss = float('inf')
    
    if args.resume and os.path.exists(args.resume):
        print(f"🔄 從檢查點恢復: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('loss', float('inf'))
    
    # 訓練循環
    try:
        for epoch in range(start_epoch, config.NUM_EPOCHS):
            print(f"\n{'='*50}")
            print(f"📅 Epoch {epoch+1}/{config.NUM_EPOCHS}")
            print(f"{'='*50}")
            
            # 訓練一個epoch
            avg_loss = trainer.train_epoch(
                train_loader, optimizer, epoch, config.MAX_MEMORY_MB
            )
            scheduler.step()
            
            current_lr = optimizer.param_groups[0]['lr']
            memory_usage = trainer.get_memory_usage()
            
            print(f"📊 Epoch {epoch+1} 結果:")
            print(f"  平均損失: {avg_loss:.4f}")
            print(f"  學習率: {current_lr:.6f}")
            print(f"  記憶體使用: {memory_usage:.1f}MB")
            
            # 保存最佳模型
            if avg_loss < best_loss and avg_loss > 0:
                best_loss = avg_loss
                torch.save(model.state_dict(), config.BEST_MODEL_PATH)
                print(f"💾 保存最佳模型，損失: {best_loss:.4f}")
            
            # 保存檢查點
            if (epoch + 1) % config.SAVE_CHECKPOINT_EVERY == 0:
                # 創建可序列化的配置字典
                config_dict = {}
                for key, value in config.__dict__.items():
                    if isinstance(value, (int, float, str, bool, list, tuple)):
                        config_dict[key] = value
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_loss,
                    'config': config_dict
                }
                checkpoint_path = f"{config.CHECKPOINT_PREFIX}{epoch+1}.pth"
                torch.save(checkpoint, checkpoint_path)
                print(f"💾 保存檢查點: {checkpoint_path}")
            
            # 記憶體清理
            gc.collect()
            if device.type == 'mps':
                torch.mps.empty_cache()
    
    except KeyboardInterrupt:
        print("\n⚠️  訓練被用戶中斷")
        # 保存當前狀態
        config_dict = {}
        for key, value in config.__dict__.items():
            if isinstance(value, (int, float, str, bool, list, tuple)):
                config_dict[key] = value
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss if 'avg_loss' in locals() else float('inf'),
            'config': config_dict
        }
        torch.save(checkpoint, 'interrupted_checkpoint.pth')
        print("💾 已保存中斷檢查點: interrupted_checkpoint.pth")
    
    except Exception as e:
        print(f"❌ 訓練過程中發生錯誤: {e}")
        sys.exit(1)
    
    # 保存最終模型
    torch.save(model.state_dict(), config.FINAL_MODEL_PATH)
    
    print("\n🎉 訓練完成！")
    print("📁 生成的文件:")
    print(f"  - {config.BEST_MODEL_PATH} (最佳模型)")
    print(f"  - {config.FINAL_MODEL_PATH} (最終模型)")
    print("  - checkpoint_mac_epoch_*.pth (檢查點)")

if __name__ == "__main__":
    main() 