#!/usr/bin/env python3
"""
時尚AI監督學習訓練系統
支援多種backbone架構的穿搭風格分類訓練
"""

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
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import time
import gc
import psutil
import multiprocessing as mp
import argparse
from datetime import datetime

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

# ==================== 輸出管理 ====================

def create_output_directory(base_dir="result"):
    """創建輸出目錄，格式為 result/[yyyymmddhhmmss]/"""
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
        'num_styles': config.num_styles,
        'num_genders': config.num_genders,
        'feature_dim': config.feature_dim,
        'batch_size': config.batch_size,
        'num_epochs': config.num_epochs,
        'learning_rate': config.learning_rate,
        'max_samples_per_class': config.max_samples_per_class,
        'num_workers': config.num_workers,
        'max_memory_mb': config.max_memory_mb,
        'style_categories': config.style_categories,
        'gender_categories': config.gender_categories,
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
    fig.suptitle('Training History', fontsize=16)
    
    # 損失圖
    axes[0, 0].plot(train_history['epoch'], train_history['loss'], 'b-', label='Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 準確率圖
    axes[0, 1].plot(train_history['epoch'], train_history['style_acc'], 'r-', label='Style Accuracy')
    axes[0, 1].plot(train_history['epoch'], train_history['gender_acc'], 'g-', label='Gender Accuracy')
    axes[0, 1].set_title('Training Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 學習率圖
    axes[1, 0].plot(train_history['epoch'], train_history['lr'], 'm-', label='Learning Rate')
    axes[1, 0].set_title('Learning Rate')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 記憶體使用圖
    axes[1, 1].plot(train_history['epoch'], train_history['memory_usage'], 'c-', label='Memory Usage')
    axes[1, 1].set_title('Memory Usage')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Memory (MB)')
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
    def __init__(self, config_type="balanced", backbone_type="mobilenet"):
        self.config_type = config_type
        self.backbone_type = backbone_type
        
        # 基礎配置
        self.num_styles = 12
        self.num_genders = 2
        self.feature_dim = 1024
        
        # 風格和性別類別
        self.style_categories = [
            'Athleisure', 'BRITISH', 'CASUAL', 'GOTH', 'Kawaii', 
            'Korean', 'MINIMALIST', 'Preppy', 'STREET', 'Streetwear', 
            'Vintage', 'Y2K'
        ]
        self.gender_categories = ['MEN', 'WOMEN']
        
        # 根據配置類型設置參數
        if config_type == "minimal":
            self._set_minimal_config()
        elif config_type == "performance":
            self._set_performance_config()
        else:  # balanced
            self._set_balanced_config()
    
    def _set_minimal_config(self):
        """最小配置 - 適合記憶體有限的情況"""
        self.batch_size = 2
        self.num_epochs = 10
        self.learning_rate = 0.001
        self.max_samples_per_class = 100
        self.num_workers = 0
        self.max_memory_mb = 4000
        
    def _set_balanced_config(self):
        """平衡配置 - 推薦設置"""
        self.batch_size = 8
        self.num_epochs = 30
        self.learning_rate = 0.001
        self.max_samples_per_class = 500
        self.num_workers = 2
        self.max_memory_mb = 8000
        
    def _set_performance_config(self):
        """性能配置 - 適合高性能Mac"""
        self.batch_size = 16
        self.num_epochs = 50
        self.learning_rate = 0.0005
        self.max_samples_per_class = 1000
        self.num_workers = 4
        self.max_memory_mb = 16000

# ==================== Backbone定義 ====================

class FashionBackbone(nn.Module):
    """可配置的Fashion Backbone"""
    def __init__(self, backbone_type='mobilenet', pretrained=True):
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
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
            self.backbone.fc = nn.Identity()
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

class OptimizedOutfitDataset(Dataset):
    def __init__(self, root_dir, config, transform=None, mode='train'):
        self.root_dir = root_dir
        self.config = config
        self.transform = transform
        self.mode = mode
        
        self.style_to_idx = {style: idx for idx, style in enumerate(config.style_categories)}
        self.gender_to_idx = {gender: idx for idx, gender in enumerate(config.gender_categories)}
        
        self.samples = []
        self._load_samples()
        
        print(f"📊 載入了 {len(self.samples)} 個樣本")
        
    def _load_samples(self):
        class_counts = defaultdict(int)
        
        for style in self.config.style_categories:
            for gender in self.config.gender_categories:
                folder_name = f"{style}_{gender}"
                folder_path = os.path.join(self.root_dir, folder_name)
                
                if os.path.exists(folder_path):
                    img_files = [f for f in os.listdir(folder_path) 
                               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    
                    img_files = img_files[:self.config.max_samples_per_class]
                    
                    for img_name in img_files:
                        img_path = os.path.join(folder_path, img_name)
                        
                        if os.path.getsize(img_path) < 1024:
                            continue
                            
                        if self._is_valid_image(img_path):
                            self.samples.append({
                                'path': img_path,
                                'style': style,
                                'gender': gender,
                                'style_idx': self.style_to_idx[style],
                                'gender_idx': self.gender_to_idx[gender]
                            })
                            class_counts[f"{style}_{gender}"] += 1
        
        print("📈 各類別樣本數量:")
        for class_name, count in sorted(class_counts.items()):
            print(f"  {class_name}: {count}")
    
    def _is_valid_image(self, img_path):
        """快速檢查圖片是否有效"""
        try:
            with Image.open(img_path) as img:
                img.verify()
                return True
        except Exception:
            return False
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            image = Image.open(sample['path']).convert('RGB')
            if min(image.size) < 32:
                raise ValueError("圖片太小")
        except Exception as e:
            print(f"⚠️  載入圖片失敗: {sample['path']}")
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        if self.transform:
            image = self.transform(image)
            
        return {
            'image': image,
            'style_idx': sample['style_idx'],
            'gender_idx': sample['gender_idx'],
            'style': sample['style'],
            'gender': sample['gender'],
            'path': sample['path']
        }

# ==================== 模型定義 ====================

class EnhancedStyleClassifier(nn.Module):
    def __init__(self, config):
        super(EnhancedStyleClassifier, self).__init__()
        
        self.backbone = FashionBackbone(
            backbone_type=config.backbone_type, 
            pretrained=True
        )
        
        backbone_features = self.backbone.get_feature_dim()
        self.feature_dim = config.feature_dim
        
        # 特徵投影層
        self.feature_projection = nn.Sequential(
            nn.Linear(backbone_features, config.feature_dim),
            nn.LayerNorm(config.feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 風格分類頭
        self.style_classifier = nn.Sequential(
            nn.Linear(config.feature_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, config.num_styles)
        )
        
        # 性別分類頭
        self.gender_classifier = nn.Sequential(
            nn.Linear(config.feature_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, config.num_genders)
        )
        
        # 投影頭（用於對比學習）
        self.projection_head = nn.Sequential(
            nn.Linear(config.feature_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128)
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
        batch_size = x.size(0)
        if batch_size == 1 and self.training:
            self.eval()
            backbone_features = self.backbone(x)
            features = self.feature_projection(backbone_features)
            
            projected_features = self.projection_head(features)
            projected_features = F.normalize(projected_features, dim=1)
            
            style_logits = self.style_classifier(features)
            gender_logits = self.gender_classifier(features)
            
            self.train()
        else:
            backbone_features = self.backbone(x)
            features = self.feature_projection(backbone_features)
            
            projected_features = self.projection_head(features)
            projected_features = F.normalize(projected_features, dim=1)
            
            style_logits = self.style_classifier(features)
            gender_logits = self.gender_classifier(features)
        
        return {
            'features': features,
            'projected_features': projected_features,
            'style_logits': style_logits,
            'gender_logits': gender_logits
        }

# ==================== 損失函數 ====================

class MemoryEfficientContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features, labels):
        batch_size = features.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=features.device)
        
        chunk_size = min(32, batch_size)
        total_loss = 0.0
        num_chunks = 0
        
        for i in range(0, batch_size, chunk_size):
            end_i = min(i + chunk_size, batch_size)
            chunk_features = features[i:end_i]
            chunk_labels = labels[i:end_i]
            
            similarity_matrix = torch.matmul(chunk_features, features.T) / self.temperature
            label_mask = (chunk_labels.unsqueeze(1) == labels.unsqueeze(0)).float()
            mask = torch.eye(len(chunk_labels), batch_size, device=features.device)
            label_mask = label_mask * (1 - mask)
            
            exp_sim = torch.exp(similarity_matrix)
            log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
            
            positive_mask = label_mask
            if positive_mask.sum() > 0:
                loss = -(log_prob * positive_mask).sum() / positive_mask.sum()
                total_loss += loss
                num_chunks += 1
        
        return total_loss / max(num_chunks, 1)

# ==================== 訓練器 ====================

class MacOptimizedTrainer:
    def __init__(self, model, config, device, output_dir):
        self.model = model
        self.config = config
        self.device = device
        self.output_dir = output_dir
        
        # 損失函數
        self.style_criterion = nn.CrossEntropyLoss()
        self.gender_criterion = nn.CrossEntropyLoss()
        self.contrastive_criterion = MemoryEfficientContrastiveLoss()
        
        # 優化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        # 學習率調度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.num_epochs
        )
        
        # 訓練記錄
        self.train_history = {
            'epoch': [],
            'loss': [],
            'style_acc': [],
            'gender_acc': [],
            'lr': [],
            'memory_usage': []
        }
        
        # 創建訓練日誌文件
        self.log_file = os.path.join(output_dir, 'training_log.txt')
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"訓練開始時間: {datetime.now().isoformat()}\n")
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
        style_correct = 0
        gender_correct = 0
        total_samples = 0
        
        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(self.device)
            style_labels = batch['style_idx'].to(self.device)
            gender_labels = batch['gender_idx'].to(self.device)
            
            # 檢查批次大小
            if images.size(0) == 1:
                continue
            
            self.optimizer.zero_grad()
            
            # 前向傳播
            outputs = self.model(images)
            
            # 計算損失
            style_loss = self.style_criterion(outputs['style_logits'], style_labels)
            gender_loss = self.gender_criterion(outputs['gender_logits'], gender_labels)
            
            # 對比損失（組合風格和性別標籤）
            combined_labels = style_labels * self.config.num_genders + gender_labels
            contrastive_loss = self.contrastive_criterion(
                outputs['projected_features'], 
                combined_labels
            )
            
            # 總損失
            total_batch_loss = style_loss + gender_loss + 0.1 * contrastive_loss
            
            # 反向傳播
            total_batch_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 統計
            total_loss += total_batch_loss.item()
            
            # 計算準確率
            style_pred = outputs['style_logits'].argmax(dim=1)
            gender_pred = outputs['gender_logits'].argmax(dim=1)
            
            style_correct += (style_pred == style_labels).sum().item()
            gender_correct += (gender_pred == gender_labels).sum().item()
            total_samples += style_labels.size(0)
            
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
                    f"Loss: {total_batch_loss.item():.4f}, "
                    f"LR: {current_lr:.6f}, "
                    f"Memory: {memory_usage:.1f}MB"
                )
        
        # Epoch統計
        avg_loss = total_loss / len(dataloader)
        style_acc = style_correct / total_samples
        gender_acc = gender_correct / total_samples
        current_lr = self.optimizer.param_groups[0]['lr']
        memory_usage = self.get_memory_usage()
        
        # 記錄歷史
        self.train_history['epoch'].append(epoch)
        self.train_history['loss'].append(avg_loss)
        self.train_history['style_acc'].append(style_acc)
        self.train_history['gender_acc'].append(gender_acc)
        self.train_history['lr'].append(current_lr)
        self.train_history['memory_usage'].append(memory_usage)
        
        self.log_message(
            f"Epoch {epoch} 完成 - "
            f"平均損失: {avg_loss:.4f}, "
            f"風格準確率: {style_acc:.4f}, "
            f"性別準確率: {gender_acc:.4f}"
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
                'num_styles': self.config.num_styles,
                'num_genders': self.config.num_genders,
                'feature_dim': self.config.feature_dim
            }
        }
        
        filepath = os.path.join(self.output_dir, filename)
        torch.save(checkpoint, filepath)
        self.log_message(f"檢查點已保存: {filepath}")

def train_model(data_root, config_type="balanced", backbone_type="mobilenet", resume_from=None):
    """主訓練函數"""
    print("🚀 開始訓練時尚風格分類模型")
    print("=" * 50)
    
    # 設備設置
    device = setup_mac_optimization()
    
    # 創建輸出目錄
    output_dir = create_output_directory()
    
    # 創建配置
    config = TrainingConfig(config_type, backbone_type)
    save_config_to_file(config, output_dir)
    
    print(f"📊 訓練配置:")
    print(f"  配置類型: {config.config_type}")
    print(f"  Backbone: {config.backbone_type}")
    print(f"  批次大小: {config.batch_size}")
    print(f"  訓練輪數: {config.num_epochs}")
    print(f"  學習率: {config.learning_rate}")
    
    # 數據預處理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 載入數據集
    print(f"📂 載入數據集: {data_root}")
    dataset = OptimizedOutfitDataset(data_root, config, transform=transform)
    
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
    model = EnhancedStyleClassifier(config).to(device)
    
    # 創建訓練器
    trainer = MacOptimizedTrainer(model, config, device, output_dir)
    
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
    
    trainer.log_message("訓練完成！")
    print(f"\n✅ 訓練完成！")
    print(f"📁 輸出目錄: {output_dir}")
    print(f"🏆 最佳損失: {best_loss:.4f}")

def main():
    parser = argparse.ArgumentParser(description='時尚風格分類訓練系統')
    parser.add_argument('--data', type=str, default='../dataset',
                       help='數據集根目錄')
    parser.add_argument('--config', type=str, default='balanced',
                       choices=['minimal', 'balanced', 'performance'],
                       help='訓練配置類型')
    parser.add_argument('--backbone', type=str, default='mobilenet',
                       choices=['mobilenet', 'resnet18', 'resnet50', 'efficientnet_b0', 
                               'efficientnet_b2', 'vit_tiny', 'vit_small', 'fashion_resnet'],
                       help='Backbone架構')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢復訓練的檢查點路徑')
    
    args = parser.parse_args()
    
    train_model(
        data_root=args.data,
        config_type=args.config,
        backbone_type=args.backbone,
        resume_from=args.resume
    )

if __name__ == "__main__":
    main() 