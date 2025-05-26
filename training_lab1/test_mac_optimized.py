import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import numpy as np
import cv2
from PIL import Image
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import time
import gc
import psutil
import multiprocessing as mp

# Mac 特定優化設置
def setup_mac_optimization():
    """設置Mac特定的優化"""
    # 設置多進程啟動方法為spawn（Mac安全）
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    
    # 檢查是否有MPS（Metal Performance Shaders）支持
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("✅ 檢測到 Metal Performance Shaders (MPS) 支持")
        device = torch.device("mps")
    else:
        print("⚠️  MPS 不可用，使用 CPU")
        device = torch.device("cpu")
    
    # 設置線程數（Mac優化）
    torch.set_num_threads(min(8, mp.cpu_count()))
    
    # 禁用某些可能導致問題的優化
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    return device

# ================== 記憶體優化的數據集 ==================

class OptimizedOutfitDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train', max_samples_per_class=500):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.max_samples_per_class = max_samples_per_class
        
        # 定義風格類別
        self.style_categories = [
            'Athleisure', 'BRITISH', 'CASUAL', 'GOTH', 'Kawaii', 
            'Korean', 'MINIMALIST', 'Preppy', 'STREET', 'Streetwear', 
            'Vintage', 'Y2K'
        ]
        
        self.gender_categories = ['MEN', 'WOMEN']
        
        # 建立類別到索引的映射
        self.style_to_idx = {style: idx for idx, style in enumerate(self.style_categories)}
        self.gender_to_idx = {gender: idx for idx, gender in enumerate(self.gender_categories)}
        
        # 載入圖片路徑和標籤（限制數量以節省記憶體）
        self.samples = []
        self._load_samples()
        
        print(f"📊 載入了 {len(self.samples)} 個樣本")
        
    def _load_samples(self):
        class_counts = defaultdict(int)
        
        for style in self.style_categories:
            for gender in self.gender_categories:
                folder_name = f"{style}_{gender}"
                folder_path = os.path.join(self.root_dir, folder_name)
                
                if os.path.exists(folder_path):
                    img_files = [f for f in os.listdir(folder_path) 
                               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    
                    # 限制每個類別的樣本數量
                    img_files = img_files[:self.max_samples_per_class]
                    
                    for img_name in img_files:
                        img_path = os.path.join(folder_path, img_name)
                        
                        # 快速檢查文件大小（跳過過小的文件）
                        if os.path.getsize(img_path) < 1024:  # 小於1KB
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
                img.verify()  # 驗證圖片完整性
                return True
        except Exception:
            return False
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 載入圖片並處理異常
        try:
            image = Image.open(sample['path']).convert('RGB')
            
            # 檢查圖片尺寸
            if min(image.size) < 32:  # 太小的圖片
                raise ValueError("圖片太小")
                
        except Exception as e:
            print(f"⚠️  載入圖片失敗: {sample['path']}")
            # 創建一個預設的彩色圖片
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

# ================== 輕量化模型 ==================

class LightweightStyleClassifier(nn.Module):
    def __init__(self, num_styles=12, num_genders=2, feature_dim=1024):
        super(LightweightStyleClassifier, self).__init__()
        
        # 使用更輕量的backbone
        try:
            # 嘗試使用MobileNetV3
            self.backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
            self.backbone.classifier = nn.Identity()
            backbone_features = 960
        except:
            # 備選方案：使用ResNet18
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.backbone.fc = nn.Identity()
            backbone_features = 512
            
        self.feature_dim = feature_dim
        
        # 特徵投影層
        self.feature_projection = nn.Sequential(
            nn.Linear(backbone_features, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 風格分類頭（簡化）
        self.style_classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_styles)
        )
        
        # 性別分類頭（簡化）
        self.gender_classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_genders)
        )
        
        # 投影頭（用於對比學習）
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128)
        )
        
        # 初始化權重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化權重"""
        for module in [self.feature_projection, self.style_classifier, 
                      self.gender_classifier, self.projection_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 特徵提取
        backbone_features = self.backbone(x)
        features = self.feature_projection(backbone_features)
        
        # 分類預測
        style_logits = self.style_classifier(features)
        gender_logits = self.gender_classifier(features)
        
        # 投影特徵
        projected_features = self.projection_head(features)
        projected_features = F.normalize(projected_features, dim=1)
        
        return {
            'features': features,
            'projected_features': projected_features,
            'style_logits': style_logits,
            'gender_logits': gender_logits
        }

# ================== 記憶體優化的對比學習 ==================

class MemoryEfficientContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(MemoryEfficientContrastiveLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, features, labels):
        batch_size = features.shape[0]
        
        if batch_size < 2:
            return torch.tensor(0.0, device=features.device)
        
        # 計算相似度矩陣（分塊計算以節省記憶體）
        chunk_size = min(32, batch_size)
        similarity_matrix = torch.zeros(batch_size, batch_size, device=features.device)
        
        for i in range(0, batch_size, chunk_size):
            end_i = min(i + chunk_size, batch_size)
            for j in range(0, batch_size, chunk_size):
                end_j = min(j + chunk_size, batch_size)
                
                chunk_sim = torch.matmul(features[i:end_i], features[j:end_j].T) / self.temperature
                similarity_matrix[i:end_i, j:end_j] = chunk_sim
        
        # 建立正樣本mask
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        mask = mask.fill_diagonal_(0)
        
        if mask.sum() == 0:
            return torch.tensor(0.0, device=features.device)
        
        # 計算對比損失
        similarity_matrix = similarity_matrix - similarity_matrix.max(dim=1, keepdim=True)[0].detach()
        exp_sim = torch.exp(similarity_matrix)
        sum_exp_sim = torch.sum(exp_sim, dim=1, keepdim=True)
        
        log_prob = similarity_matrix - torch.log(sum_exp_sim + 1e-8)
        mean_log_prob_pos = torch.sum(mask * log_prob, dim=1) / (torch.sum(mask, dim=1) + 1e-8)
        
        valid_items = torch.sum(mask, dim=1) > 0
        if valid_items.sum() == 0:
            return torch.tensor(0.0, device=features.device)
            
        loss = -mean_log_prob_pos[valid_items].mean()
        
        if torch.isnan(loss):
            return torch.tensor(0.0, device=features.device)
            
        return loss

# ================== Mac優化的訓練器 ==================

class MacOptimizedTrainer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.contrastive_loss = MemoryEfficientContrastiveLoss()
        
        # 記憶體監控
        self.process = psutil.Process()
        
    def get_memory_usage(self):
        """獲取當前記憶體使用量"""
        memory_info = self.process.memory_info()
        return memory_info.rss / 1024 / 1024  # MB
    
    def train_epoch(self, dataloader, optimizer, epoch, max_memory_mb=8000):
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        print(f"🚀 開始訓練 Epoch {epoch}")
        start_time = time.time()
        
        for batch_idx, batch in enumerate(dataloader):
            try:
                # 記憶體檢查
                current_memory = self.get_memory_usage()
                if current_memory > max_memory_mb:
                    print(f"⚠️  記憶體使用過高 ({current_memory:.1f}MB)，執行垃圾回收")
                    gc.collect()
                    if self.device.type == 'mps':
                        torch.mps.empty_cache()
                    elif self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                
                images = batch['image'].to(self.device, non_blocking=True)
                style_labels = batch['style_idx'].to(self.device, non_blocking=True)
                gender_labels = batch['gender_idx'].to(self.device, non_blocking=True)
                
                # 檢查輸入有效性
                if torch.isnan(images).any() or torch.isinf(images).any():
                    print(f"⚠️  跳過batch {batch_idx}: 輸入包含NaN或Inf")
                    continue
                
                optimizer.zero_grad()
                
                # 前向傳播
                with torch.cuda.amp.autocast(enabled=False):  # MPS不支援AMP
                    outputs = self.model(images)
                    
                    # 計算損失
                    style_loss = F.cross_entropy(outputs['style_logits'], style_labels)
                    gender_loss = F.cross_entropy(outputs['gender_logits'], gender_labels)
                    contrastive_loss = self.contrastive_loss(outputs['projected_features'], style_labels)
                    
                    # 檢查損失有效性
                    if torch.isnan(style_loss) or torch.isnan(gender_loss) or torch.isnan(contrastive_loss):
                        print(f"⚠️  跳過batch {batch_idx}: 損失包含NaN")
                        continue
                    
                    # 總損失（調整權重）
                    total_batch_loss = style_loss + 0.2 * gender_loss + 0.05 * contrastive_loss
                
                if torch.isnan(total_batch_loss):
                    print(f"⚠️  跳過batch {batch_idx}: 總損失為NaN")
                    continue
                
                # 反向傳播
                total_batch_loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                total_loss += total_batch_loss.item()
                num_batches += 1
                
                # 進度報告
                if batch_idx % 10 == 0:
                    elapsed = time.time() - start_time
                    memory_usage = self.get_memory_usage()
                    print(f'📊 Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, '
                          f'Loss: {total_batch_loss.item():.4f}, '
                          f'Memory: {memory_usage:.1f}MB, '
                          f'Time: {elapsed:.1f}s')
                
                # 定期清理記憶體
                if batch_idx % 50 == 0:
                    gc.collect()
            
            except Exception as e:
                print(f"❌ 處理batch {batch_idx}時發生錯誤: {str(e)}")
                continue
        
        if num_batches == 0:
            return 0.0
        
        avg_loss = total_loss / num_batches
        epoch_time = time.time() - start_time
        print(f"✅ Epoch {epoch} 完成，平均損失: {avg_loss:.4f}，耗時: {epoch_time:.1f}s")
        
        return avg_loss

# ================== 主程式 ==================

def main():
    print("🍎 Mac 優化訓練開始")
    
    # Mac優化設置
    device = setup_mac_optimization()
    
    # 設定參數（Mac優化）
    data_root = "./dataset"
    batch_size = 8  # 減少batch size以適應Mac記憶體
    learning_rate = 0.0001
    num_epochs = 20  # 減少epoch數
    max_samples_per_class = 300  # 限制每類樣本數
    
    # 輕量化的數據增強
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 直接resize，不使用RandomCrop
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 建立數據集
    print("📂 載入數據集...")
    train_dataset = OptimizedOutfitDataset(
        data_root, 
        transform=train_transform,
        max_samples_per_class=max_samples_per_class
    )
    
    # Mac優化的DataLoader設置
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,  # Mac上設為0避免多進程問題
        pin_memory=False,  # MPS不需要pin_memory
        drop_last=True,
        persistent_workers=False
    )
    
    print(f"📊 數據集大小: {len(train_dataset)}")
    print(f"🔧 使用設備: {device}")
    
    # 初始化輕量化模型
    model = LightweightStyleClassifier()
    model.to(device)
    
    # 計算模型參數數量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📈 模型參數: {total_params:,} (可訓練: {trainable_params:,})")
    
    # 初始化訓練器
    trainer = MacOptimizedTrainer(model, device)
    
    # 優化器設置
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    # 學習率調度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs,
        eta_min=learning_rate * 0.1
    )
    
    # 訓練循環
    print("🚀 開始訓練...")
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\n{'='*50}")
        print(f"📅 Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*50}")
        
        # 訓練一個epoch
        avg_loss = trainer.train_epoch(train_loader, optimizer, epoch)
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
            torch.save(model.state_dict(), 'outfit_model_best_mac.pth')
            print(f"💾 保存最佳模型，損失: {best_loss:.4f}")
        
        # 每5個epoch保存檢查點
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
            }
            torch.save(checkpoint, f'checkpoint_mac_epoch_{epoch+1}.pth')
            print(f"💾 保存檢查點: epoch_{epoch+1}")
        
        # 記憶體清理
        gc.collect()
        if device.type == 'mps':
            torch.mps.empty_cache()
    
    # 保存最終模型
    torch.save(model.state_dict(), 'outfit_model_final_mac.pth')
    
    print("\n🎉 訓練完成！")
    print("📁 模型文件:")
    print("  - outfit_model_best_mac.pth (最佳模型)")
    print("  - outfit_model_final_mac.pth (最終模型)")
    print("  - checkpoint_mac_epoch_*.pth (檢查點)")

def resume_training(checkpoint_path, data_root="./dataset"):
    """從檢查點恢復訓練"""
    print(f"🔄 從檢查點恢復訓練: {checkpoint_path}")
    
    device = setup_mac_optimization()
    
    # 載入檢查點
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 重建模型和優化器
    model = LightweightStyleClassifier()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    print(f"📅 從 epoch {start_epoch} 開始恢復訓練")
    
    # 繼續訓練...
    # (這裡可以添加繼續訓練的邏輯)

if __name__ == "__main__":
    main() 