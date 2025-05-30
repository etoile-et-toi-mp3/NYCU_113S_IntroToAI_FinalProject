#!/usr/bin/env python3
"""
時尚AI無監督學習訓練系統
支援多種backbone架構的SimCLR對比學習訓練
"""

# 設置matplotlib不彈出視窗
import matplotlib
matplotlib.use('Agg')  # 使用非互動式後端

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import timm
from transformers import ViTModel, ViTConfig
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import json
import time
import gc
import psutil
import multiprocessing as mp
import argparse
from datetime import datetime
from pathlib import Path
import warnings
import pillow_avif

# ==================== Mac優化設置 ====================

def setup_mac_optimization():
    """設置Mac特定的優化"""
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("✅ 檢測到 Metal Performance Shaders (MPS) 支持")
        device = torch.device("mps")
    else:
        print("⚠️  MPS 不可用，使用 CPU")
        device = torch.device("cpu")
    
    torch.set_num_threads(min(8, mp.cpu_count()))
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    return device

# ==================== Cuda加速設置 ====================

def setup_cuda_optimization():
    """設置 NVIDIA CUDA 特定的優化"""
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)

    if torch.cuda.is_available():
        print(f"✅ 檢測到 CUDA 支持: {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda")
        
        # CUDA 性能優化選項
        torch.backends.cudnn.benchmark = True  # 適用於輸入尺寸固定的模型
        torch.backends.cudnn.deterministic = False  # 提高速度但結果非完全可重現
    else:
        print("⚠️ CUDA 不可用，使用 CPU")
        device = torch.device("cpu")

    # 根據 CPU 核心數設置 PyTorch 線程數
    torch.set_num_threads(min(8, mp.cpu_count()))

    return device


# ==================== 輸出管理 ====================

def create_output_directory(base_dir="unsupervised_result"):
    """創建輸出目錄，格式為 unsupervised_result/[yyyymmddhhmmss]/"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_dir = os.path.join(base_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 輸出目錄: {output_dir}")
    return output_dir

def save_config_to_file(config, output_dir):
    """保存訓練配置到文件"""
    config_dict = {
        'config_type': config.config_type,
        'backbone_type': config.backbone_type,
        'feature_dim': config.feature_dim,
        'batch_size': config.batch_size,
        'num_epochs': config.num_epochs,
        'learning_rate': config.learning_rate,
        'max_samples_per_class': config.max_samples_per_class,
        'num_workers': config.num_workers,
        'max_memory_mb': config.max_memory_mb,
        'temperature': config.temperature,
        'timestamp': datetime.now().isoformat()
    }
    
    config_path = os.path.join(output_dir, 'training_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    print(f"💾 訓練配置已保存: {config_path}")

def save_training_history_plot(train_history, output_dir):
    """保存訓練歷史圖表"""
    if not train_history['epoch']:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Unsupervised Training History', fontsize=16)
    
    # 損失圖
    axes[0, 0].plot(train_history['epoch'], train_history['loss'], 'b-', label='Contrastive Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 學習率圖
    axes[0, 1].plot(train_history['epoch'], train_history['lr'], 'm-', label='Learning Rate')
    axes[0, 1].set_title('Learning Rate')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Learning Rate')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 記憶體使用圖
    axes[1, 0].plot(train_history['epoch'], train_history['memory_usage'], 'c-', label='Memory Usage')
    axes[1, 0].set_title('Memory Usage')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Memory (MB)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 溫度參數圖
    axes[1, 1].plot(train_history['epoch'], [train_history['temperature'][0]] * len(train_history['epoch']), 'r-', label=f'Temperature ({train_history["temperature"][0]})')
    axes[1, 1].set_title('Temperature Parameter')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Temperature')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 訓練歷史圖表已保存: {plot_path}")

# ==================== 配置管理 ====================

class TrainingConfig:
    """訓練配置類"""
    def __init__(self, config_type="balanced", backbone_type="resnet50", device="cpu"):
        self.config_type = config_type
        self.backbone_type = backbone_type
        
        # 基礎配置
        self.feature_dim = 128
        self.temperature = 0.2
        self.max_samples_per_class = None  # 無監督不需要類別限制
        
        # 根據配置類型設置參數
        if device == "cuda":
            if config_type == "minimal":
                self._cuda_set_minimal_config()
            elif config_type == "performance":
                self._cuda_set_performance_config()
            else:  # balanced
                self._cuda_set_balanced_config()
        else:
            if config_type == "minimal":
                self._mac_set_minimal_config()
            elif config_type == "performance":
                self._mac_set_performance_config()
            else:  # balanced
                self._mac_set_balanced_config()
    
    def _mac_set_minimal_config(self):
        """最小配置 - 適合記憶體有限的情況"""
        self.batch_size = 16
        self.num_epochs = 10
        self.learning_rate = 1e-4
        self.num_workers = 0
        self.max_memory_mb = 4000
        
    def _mac_set_balanced_config(self):
        """平衡配置 - 推薦設置"""
        self.batch_size = 32
        self.num_epochs = 20
        self.learning_rate = 3e-4
        self.num_workers = 2
        self.max_memory_mb = 8000
        
    def _mac_set_performance_config(self):
        """性能配置 - 適合高性能Mac"""
        self.batch_size = 64
        self.num_epochs = 30
        self.learning_rate = 5e-4
        self.num_workers = 4
        self.max_memory_mb = 16000
        
    def _cuda_set_minimal_config(self):
        """最小配置 - 適合記憶體有限的情況"""
        self.batch_size = 64
        self.num_epochs = 15
        self.learning_rate = 1e-4
        self.num_workers = 2
        self.max_memory_mb = 4096
        
    def _cuda_set_balanced_config(self):
        """平衡配置 - 推薦設置"""
        self.batch_size = 128
        self.num_epochs = 20
        self.learning_rate = 3e-4
        self.num_workers = 4
        self.max_memory_mb = 8192
        
    def _cuda_set_performance_config(self):
        """性能配置 - 適合高性能Mac"""
        self.batch_size = 256
        self.num_epochs = 30
        self.learning_rate = 5e-4
        self.num_workers = 8
        self.max_memory_mb = 16384

# ==================== Backbone定義 ====================

class FashionBackbone(nn.Module):
    """可配置的Fashion Backbone"""
    def __init__(self, backbone_type='resnet50', pretrained=True):
        super(FashionBackbone, self).__init__()
        self.backbone_type = backbone_type
        
        if backbone_type == 'mobilenet':
            self.backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None)
            self.backbone.classifier = nn.Identity()
            self.feature_dim = 960
            
        elif backbone_type == 'resnet18':
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            self.backbone.fc = nn.Identity()
            self.feature_dim = 512
            
        elif backbone_type == 'resnet50':
            self.backbone = timm.create_model("resnet50", pretrained=pretrained, num_classes=0)
            self.feature_dim = 2048
            
        elif backbone_type == 'efficientnet_b0':
            self.backbone = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=0)
            self.feature_dim = 1280
            
        elif backbone_type == 'efficientnet_b2':
            self.backbone = timm.create_model('efficientnet_b2', pretrained=pretrained, num_classes=0)
            self.feature_dim = 1408
            
        elif backbone_type == 'vit_tiny':
            if pretrained:
                self.backbone = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0)
            else:
                config = ViTConfig(hidden_size=192, num_hidden_layers=12, num_attention_heads=3)
                self.backbone = ViTModel(config)
            self.feature_dim = 192
            
        elif backbone_type == 'vit_small':
            if pretrained:
                self.backbone = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=0)
            else:
                config = ViTConfig(hidden_size=384, num_hidden_layers=12, num_attention_heads=6)
                self.backbone = ViTModel(config)
            self.feature_dim = 384
            
        elif backbone_type == 'fashion_resnet':
            self.backbone = self._create_fashion_resnet(pretrained)
            self.feature_dim = 512
            
        else:
            raise ValueError(f"不支持的backbone類型: {backbone_type}")
    
    def _create_fashion_resnet(self, pretrained):
        """創建針對時尚圖片優化的ResNet"""
        base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        base_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base_model.maxpool = nn.Identity()
        base_model.fc = nn.Identity()
        return base_model
    
    def forward(self, x):
        if 'vit' in self.backbone_type and hasattr(self.backbone, 'last_hidden_state'):
            outputs = self.backbone(x)
            if hasattr(outputs, 'last_hidden_state'):
                return outputs.last_hidden_state[:, 0, :]
            else:
                return outputs
        else:
            return self.backbone(x)
    
    def get_feature_dim(self):
        return self.feature_dim

# ==================== 數據集 ====================

class UnsupervisedFashionDataset(Dataset):
    def __init__(self, data_root, config, transform=None):
        self.data_root = Path(data_root)
        self.config = config
        self.transform = transform
        self.invalid_images = []
        
        # 掃描所有圖片
        image_paths = list(self.data_root.rglob("*.[jJ][pP][gG]")) + \
                      list(self.data_root.rglob("*.[pP][nN][gG]")) + \
                      list(self.data_root.rglob("*.[aA][vV][iI][fF]"))
        
        self.image_paths = []
        for p in image_paths:
            try:
                with Image.open(p) as img:
                    img.verify()
                self.image_paths.append(str(p))
            except Exception as e:
                self.invalid_images.append(str(p))
                warnings.warn(f"無法驗證圖片 {p}: {e}")
        
        print(f"📊 載入了 {len(self.image_paths)} 張有效圖片，{len(self.invalid_images)} 張無效圖片")
        
        if self.invalid_images:
            with open("invalid_images.txt", "w") as f:
                f.write("\n".join(self.invalid_images))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if image.size[0] < 10 or image.size[1] < 10:
                raise ValueError("圖片尺寸過小")
        except Exception as e:
            warnings.warn(f"無法載入圖片 {img_path}: {e}")
            image = Image.new('RGB', (224, 224), (128, 128, 128))
        
        if self.transform:
            image1 = self.transform(image)
            image2 = self.transform(image)
        else:
            image1 = image2 = transforms.ToTensor()(image)
        
        return image1, image2, img_path

# ==================== 模型定義 ====================

class UnsupervisedStyleModel(nn.Module):
    def __init__(self, config):
        super(UnsupervisedStyleModel, self).__init__()
        
        self.backbone = FashionBackbone(
            backbone_type=config.backbone_type, 
            pretrained=True
        )
        
        backbone_features = self.backbone.get_feature_dim()
        self.feature_dim = config.feature_dim
        
        # 投影頭 - 用於對比學習
        self.projection = nn.Sequential(
            nn.Linear(backbone_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, config.feature_dim),
            nn.BatchNorm1d(config.feature_dim)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        backbone_features = self.backbone(x)
        features = self.projection(backbone_features)
        return features

# ==================== 損失函數 ====================

def nt_xent_loss(features1, features2, temperature):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss for SimCLR
    修復版本：正確實現對比學習損失
    """
    batch_size = features1.shape[0]
    device = features1.device
    
    # 檢查特徵是否已歸一化
    features1 = F.normalize(features1, dim=1)
    features2 = F.normalize(features2, dim=1)
    
    # 合併兩個view的特徵: [2*batch_size, feature_dim]
    features = torch.cat([features1, features2], dim=0)
    
    # 計算相似度矩陣: [2*batch_size, 2*batch_size]
    similarity_matrix = torch.matmul(features, features.T) / temperature
    
    # 創建正樣本掩碼
    # 對於batch中第i個樣本，正樣本對是 (i, i+batch_size) 和 (i+batch_size, i)
    labels = torch.arange(batch_size, device=device)
    # 創建正樣本對的標籤
    positive_mask = torch.zeros(2 * batch_size, 2 * batch_size, device=device)
    for i in range(batch_size):
        positive_mask[i, i + batch_size] = 1  # view1 -> view2
        positive_mask[i + batch_size, i] = 1  # view2 -> view1
    
    # 創建負樣本掩碼（排除自己和正樣本）
    negative_mask = torch.ones(2 * batch_size, 2 * batch_size, device=device)
    negative_mask = negative_mask - torch.eye(2 * batch_size, device=device)  # 排除自己
    negative_mask = negative_mask - positive_mask  # 排除正樣本
    
    # 計算損失
    total_loss = 0.0
    num_positives = 0
    
    for i in range(2 * batch_size):
        # 找到正樣本
        positive_indices = torch.where(positive_mask[i] == 1)[0]
        if len(positive_indices) == 0:
            continue
            
        # 計算分子：正樣本的相似度
        positive_similarities = similarity_matrix[i, positive_indices]
        
        # 計算分母：所有負樣本的相似度
        negative_indices = torch.where(negative_mask[i] == 1)[0]
        if len(negative_indices) == 0:
            continue
            
        negative_similarities = similarity_matrix[i, negative_indices]
        
        # 計算InfoNCE損失
        for pos_sim in positive_similarities:
            numerator = torch.exp(pos_sim)
            denominator = numerator + torch.sum(torch.exp(negative_similarities))
            
            if denominator > 0:
                loss = -torch.log(numerator / denominator)
                total_loss += loss
                num_positives += 1
    
    if num_positives > 0:
        return total_loss / num_positives
    else:
        return torch.tensor(0.0, device=device, requires_grad=True)

# ==================== 訓練器 ====================

class OptimizedUnsupervisedTrainer:
    def __init__(self, model, config, device, output_dir):
        self.model = model
        self.config = config
        self.device = device
        self.output_dir = output_dir
        
        # 優化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.learning_rate,
            weight_decay=1e-4
        )
        
        # 學習率調度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.num_epochs
        )
        
        # 訓練記錄
        self.train_history = {
            'epoch': [],
            'loss': [],
            'lr': [],
            'memory_usage': [],
            'temperature': [config.temperature]
        }
        
        # 創建訓練日誌文件
        self.log_file = os.path.join(output_dir, 'training_log.txt')
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"無監督訓練開始時間: {datetime.now().isoformat()}\n")
            f.write(f"配置: {config.config_type}\n")
            f.write(f"Backbone: {config.backbone_type}\n")
            f.write("="*50 + "\n")
    
    def log_message(self, message):
        """記錄訓練日誌"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_msg + "\n")
    
    def get_memory_usage(self):
        """獲取記憶體使用情況"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB
    
    def train_epoch(self, dataloader, epoch):
        """訓練一個epoch"""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        
        for batch_idx, (images1, images2, _) in enumerate(dataloader):
            images1, images2 = images1.to(self.device), images2.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 前向傳播
            features1 = self.model(images1)
            features2 = self.model(images2)
            
            # 計算對比損失
            loss = nt_xent_loss(features1, features2, self.config.temperature)
            
            # 反向傳播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 統計
            total_loss += loss.item()
            total_samples += images1.size(0)
            
            # 記憶體管理
            if batch_idx % 10 == 0:
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                gc.collect()
            
            # 進度顯示
            if batch_idx % 20 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                memory_usage = self.get_memory_usage()
                
                self.log_message(
                    f"Epoch {epoch}/{self.config.num_epochs}, "
                    f"Batch {batch_idx}/{len(dataloader)}, "
                    f"Loss: {loss.item():.4f}, "
                    f"LR: {current_lr:.6f}, "
                    f"Memory: {memory_usage:.1f}MB"
                )
        
        # Epoch統計
        avg_loss = total_loss / len(dataloader)
        current_lr = self.optimizer.param_groups[0]['lr']
        memory_usage = self.get_memory_usage()
        
        # 記錄歷史
        self.train_history['epoch'].append(epoch)
        self.train_history['loss'].append(avg_loss)
        self.train_history['lr'].append(current_lr)
        self.train_history['memory_usage'].append(memory_usage)
        
        self.log_message(
            f"Epoch {epoch} 完成 - "
            f"平均損失: {avg_loss:.4f}"
        )
        
        return avg_loss
    
    def save_checkpoint(self, epoch, loss, filename):
        """保存檢查點"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': {
                'backbone_type': self.config.backbone_type,
                'feature_dim': self.config.feature_dim,
                'temperature': self.config.temperature
            }
        }
        
        filepath = os.path.join(self.output_dir, filename)
        torch.save(checkpoint, filepath)
        self.log_message(f"檢查點已保存: {filepath}")

def train_model(data_root, config_type="balanced", backbone_type="resnet50", resume_from=None, platform="cpu"):
    """主訓練函數"""
    print("🚀 開始無監督時尚模型訓練")
    print("=" * 50)
    
    # 設備設置
    device = torch.device("cpu")
    if platform == "auto":
        device = setup_cuda_optimization() # prefer cuda first
        if device == torch.device("cpu"):
            device = setup_mac_optimization()
    else:
        if platform == "cuda":
            device = setup_cuda_optimization()
        elif platform == "mps":
            device = setup_mac_optimization()
    
    # 創建輸出目錄
    output_dir = create_output_directory()
    
    # 創建配置
    config = TrainingConfig(config_type, backbone_type, device)
    save_config_to_file(config, output_dir)
    
    print(f"📊 訓練配置:")
    print(f"  配置類型: {config.config_type}")
    print(f"  Backbone: {config.backbone_type}")
    print(f"  批次大小: {config.batch_size}")
    print(f"  訓練輪數: {config.num_epochs}")
    print(f"  學習率: {config.learning_rate}")
    print(f"  溫度參數: {config.temperature}")
    
    # 數據預處理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 載入數據集
    print(f"📂 載入數據集: {data_root}")
    dataset = UnsupervisedFashionDataset(data_root, config, transform=transform)
    
    if not dataset.image_paths:
        raise ValueError("無有效圖片可訓練")
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=False,
        drop_last=True
    )
    
    # 創建模型
    print(f"🔧 創建模型: {config.backbone_type}")
    model = UnsupervisedStyleModel(config).to(device)
    
    # 創建訓練器
    trainer = OptimizedUnsupervisedTrainer(model, config, device, output_dir)
    
    # 恢復訓練（如果指定）
    start_epoch = 1
    best_loss = float('inf')
    
    if resume_from and os.path.exists(resume_from):
        print(f"📥 恢復訓練: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['loss']
        trainer.log_message(f"從 epoch {start_epoch-1} 恢復訓練")
    
    # 訓練循環
    trainer.log_message("開始訓練循環")
    
    try:
        for epoch in range(start_epoch, config.num_epochs + 1):
            epoch_loss = trainer.train_epoch(dataloader, epoch)
            
            # 學習率調度
            trainer.scheduler.step()
            
            # 保存最佳模型
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                trainer.save_checkpoint(
                    epoch, epoch_loss, 
                    f'best_model_{backbone_type}_{config_type}.pth'
                )
            
            # 定期保存檢查點
            if epoch % 5 == 0:
                trainer.save_checkpoint(
                    epoch, epoch_loss,
                    f'checkpoint_epoch_{epoch}.pth'
                )
            
            # 記憶體清理
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            gc.collect()
    
    except KeyboardInterrupt:
        trainer.log_message("訓練被中斷")
        trainer.save_checkpoint(
            epoch, epoch_loss,
            f'interrupted_model_{backbone_type}_{config_type}.pth'
        )
    
    # 保存最終模型
    trainer.save_checkpoint(
        config.num_epochs, best_loss,
        f'final_model_{backbone_type}_{config_type}.pth'
    )
    
    # 保存訓練歷史
    save_training_history_plot(trainer.train_history, output_dir)
    
    # 保存訓練歷史JSON
    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(trainer.train_history, f, indent=2)
    
    # 提取特徵並生成dataset_labels.json
    trainer.log_message("提取特徵並生成dataset_labels.json")
    model.eval()
    dataset_labels = []
    
    with torch.no_grad():
        for images1, _, img_paths in dataloader:
            images1 = images1.to(device)
            features = model(images1).cpu().numpy()
            for path, feat in zip(img_paths, features):
                dataset_labels.append({
                    "path": path,
                    "unsupervised_features": feat.tolist()
                })
    
    labels_path = os.path.join(output_dir, 'dataset_labels.json')
    with open(labels_path, "w") as f:
        json.dump(dataset_labels, f, indent=2)
    trainer.log_message(f"已生成 dataset_labels.json，包含 {len(dataset_labels)} 條記錄")
    
    trainer.log_message("訓練完成！")
    print(f"\n✅ 訓練完成！")
    print(f"📁 輸出目錄: {output_dir}")
    print(f"🏆 最佳損失: {best_loss:.4f}")

def main():
    parser = argparse.ArgumentParser(description='時尚無監督學習訓練系統')
    parser.add_argument('--data', type=str, default='../dataset',
                       help='數據集根目錄')
    parser.add_argument('--config', type=str, default='balanced',
                       choices=['minimal', 'balanced', 'performance'],
                       help='訓練配置類型')
    parser.add_argument('--backbone', type=str, default='resnet50',
                       choices=['mobilenet', 'resnet18', 'resnet50', 'efficientnet_b0', 
                               'efficientnet_b2', 'vit_tiny', 'vit_small', 'fashion_resnet'],
                       help='Backbone架構')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢復訓練的檢查點路徑')
    parser.add_argument('--platform', type=str, default='auto',
                        choices=['mps', 'cuda', 'cpu', 'auto'],
                        help='所使用的硬體裝置')
    
    args = parser.parse_args()
    
    train_model(
        data_root=args.data,
        config_type=args.config,
        backbone_type=args.backbone,
        resume_from=args.resume,
        platform=args.platform
    )

if __name__ == "__main__":
    main() 