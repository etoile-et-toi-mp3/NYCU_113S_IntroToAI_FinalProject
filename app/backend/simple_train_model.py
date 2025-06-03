#!/usr/bin/env python3
"""
簡化版穿搭推薦模型訓練系統
只使用 FashionCLIP 提取純粹的時尚語義特徵
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import json
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import psutil
import multiprocessing as mp
import gc
import argparse
import warnings
import time
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# ==================== Mac 優化設置 ====================

def setup_mac_optimization():
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("✅ 檢測到 Metal Performance Shaders (MPS) 支持")
        device = torch.device("mps")
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
        torch.set_num_threads(12)
        print(f"🚀 M4 Pro 優化設置已啟用")
    else:
        print("⚠️ MPS 不可用，使用 CPU")
        device = torch.device("cpu")
        torch.set_num_threads(min(8, mp.cpu_count()))
    
    return device

# ==================== 簡化配置 ====================

class SimpleTrainingConfig:
    def __init__(self, config_type="balanced"):
        self.config_type = config_type
        self.style_categories = [
            'Artsy', 'Athleisure', 'BRITISH', 'CASUAL', 'GOTH',
            'Japanese', 'Kawaii', 'Korean', 'Preppy', 'STREET', 'Vintage'
        ]
        self.gender_categories = ['MEN', 'WOMEN']
        self.style_to_idx = {style: idx for idx, style in enumerate(self.style_categories)}
        self.gender_to_idx = {gender: idx for idx, gender in enumerate(self.gender_categories)}
        
        # 簡化的配置選項
        if config_type == "minimal":
            self.batch_size = 8
            self.num_epochs = 5
            self.learning_rate = 0.001
            self.max_samples_per_class = 200
        elif config_type == "performance":
            self.batch_size = 32
            self.num_epochs = 15
            self.learning_rate = 0.0005
            self.max_samples_per_class = 2000
        else:  # balanced
            self.batch_size = 16
            self.num_epochs = 10
            self.learning_rate = 0.001
            self.max_samples_per_class = 800

# ==================== 簡化的特徵提取 ====================

def is_valid_image(image):
    """檢查圖片是否有效"""
    try:
        image.verify()
        image = image.copy()  # 重新載入，因為verify()會消耗圖片
        pixels = np.array(image)
        if pixels.size == 0 or np.all(pixels == 0) or np.all(pixels == 255):
            return False
        return True
    except:
        return False

def generate_simple_dataset_labels(root_dir, config, device="mps"):
    """
    簡化版數據集標籤生成 - 只使用 FashionCLIP
    """
    print("🔧 載入 FashionCLIP 模型...")
    clip_processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
    clip_model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip").to(device)
    clip_model.eval()
    
    dataset_labels = []
    processed_count = 0
    
    print("📊 開始提取純 FashionCLIP 特徵...")
    
    for style in config.style_categories:
        for gender in config.gender_categories:
            folder = os.path.join(root_dir, f"{style}_{gender}")
            if os.path.exists(folder):
                print(f"📁 處理資料夾: {style}_{gender}")
                
                # 限制每個類別的樣本數量
                image_files = [f for f in os.listdir(folder) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
                image_files = image_files[:config.max_samples_per_class]
                
                for img_name in image_files:
                    img_path = os.path.join(folder, img_name)
                    try:
                        # 載入並驗證圖片
                        image = Image.open(img_path).convert('RGB')
                        
                        if not is_valid_image(image.copy()):
                            continue
                        
                        # 只提取 FashionCLIP 特徵 - 這是關鍵改進！
                        clip_inputs = clip_processor(images=image, return_tensors="pt").to(device)
                        with torch.no_grad():
                            features = clip_model.get_image_features(**clip_inputs)
                            # 標準化特徵向量
                            features = F.normalize(features, p=2, dim=1)
                        
                        # 確保特徵向量維度正確
                        if features.shape[1] != 512:
                            print(f"⚠️ 特徵向量維度異常: {features.shape}")
                            continue
                        
                        # 只保存必要信息
                        dataset_labels.append({
                            "path": img_path,
                            "style": style,
                            "gender": gender,
                            "style_idx": config.style_to_idx[style],
                            "gender_idx": config.gender_to_idx[gender],
                            "features": features.cpu().numpy().flatten().tolist()
                        })
                        
                        processed_count += 1
                        if processed_count % 100 == 0:
                            print(f"  已處理 {processed_count} 張圖片...")
                            
                    except Exception as e:
                        print(f"⚠️ 處理圖片失敗: {img_path}, {e}")
                        continue
    
    print(f"✅ 成功處理 {len(dataset_labels)} 張圖片")
    
    # 保存簡化的標籤文件
    output_file = "simple_dataset_labels.json"
    with open(output_file, "w") as f:
        json.dump(dataset_labels, f, indent=2)
    
    print(f"💾 簡化標籤已保存到: {output_file}")
    return dataset_labels

# ==================== 簡化的數據集類 ====================

class SimpleOutfitDataset(Dataset):
    def __init__(self, labels_file="simple_dataset_labels.json"):
        with open(labels_file, "r") as f:
            self.labels = json.load(f)
        
        print(f"📊 載入 {len(self.labels)} 個樣本")
        
        # 生成更有意義的樣本對
        self.pairs = []
        self._generate_meaningful_pairs()
        
        print(f"🔗 生成 {len(self.pairs)} 個訓練對")
    
    def _generate_meaningful_pairs(self):
        """生成更有意義的正負樣本對"""
        
        # 按性別和風格組織數據
        organized_data = {}
        for sample in self.labels:
            key = f"{sample['gender']}_{sample['style']}"
            if key not in organized_data:
                organized_data[key] = []
            organized_data[key].append(sample)
        
        # 生成正樣本：同性別同風格
        for key, samples in organized_data.items():
            if len(samples) >= 2:
                for i in range(len(samples)):
                    for j in range(i+1, min(i+6, len(samples))):  # 限制正樣本數量
                        self.pairs.append((samples[i], samples[j], 1.0))
        
        # 生成負樣本：同性別不同風格
        gender_groups = {'MEN': [], 'WOMEN': []}
        for sample in self.labels:
            gender_groups[sample['gender']].append(sample)
        
        for gender, samples in gender_groups.items():
            np.random.shuffle(samples)
            for i in range(0, min(len(samples)-1, 1000), 2):  # 限制負樣本數量
                if i+1 < len(samples) and samples[i]['style'] != samples[i+1]['style']:
                    self.pairs.append((samples[i], samples[i+1], 0.0))
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        sample1, sample2, label = self.pairs[idx]
        
        features1 = torch.tensor(sample1['features'], dtype=torch.float32)
        features2 = torch.tensor(sample2['features'], dtype=torch.float32)
        
        return {
            'features1': features1,
            'features2': features2,
            'style1': sample1['style_idx'],
            'style2': sample2['style_idx'],
            'gender1': sample1['gender_idx'],
            'gender2': sample2['gender_idx'],
            'label': torch.tensor(label, dtype=torch.float32),
            'path1': sample1['path'],
            'path2': sample2['path']
        }

# ==================== 簡化的模型 ====================

class SimpleFashionRecommender(nn.Module):
    def __init__(self, config):
        super(SimpleFashionRecommender, self).__init__()
        self.config = config
        
        # 只載入 FashionCLIP
        self.clip_processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
        self.clip_model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
        
        # 凍結 CLIP 參數
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # 簡單的映射層 - 學習時尚搭配的語義關係
        self.fashion_projector = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)  # 最終的時尚語義嵌入
        )
        
        # 風格分類頭（輔助任務）
        self.style_classifier = nn.Linear(64, len(config.style_categories))
        
    def forward(self, features):
        """
        前向傳播
        Args:
            features: 預提取的 FashionCLIP 特徵 [batch_size, 512]
        """
        # 映射到時尚語義空間
        fashion_embedding = self.fashion_projector(features)
        
        # 風格預測（輔助任務）
        style_logits = self.style_classifier(fashion_embedding)
        
        return {
            'fashion_embedding': fashion_embedding,
            'style_logits': style_logits
        }
    
    def extract_image_features(self, image, device):
        """提取單張圖片的特徵"""
        self.eval()
        with torch.no_grad():
            clip_inputs = self.clip_processor(images=image, return_tensors="pt").to(device)
            clip_features = self.clip_model.get_image_features(**clip_inputs)
            clip_features = F.normalize(clip_features, p=2, dim=1)
            
            outputs = self.forward(clip_features)
            return outputs['fashion_embedding'].cpu().numpy()

# ==================== 簡化的訓練器 ====================

class SimpleFashionTrainer:
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        
        # 只訓練我們的映射層
        trainable_params = list(model.fashion_projector.parameters()) + \
                          list(model.style_classifier.parameters())
        
        self.optimizer = torch.optim.Adam(trainable_params, lr=config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.7)
        self.best_loss = float('inf')
    
    def contrastive_loss(self, emb1, emb2, label, margin=1.0):
        """對比學習損失"""
        distance = F.pairwise_distance(emb1, emb2, keepdim=True)
        loss = label * torch.pow(distance, 2) + \
               (1 - label) * torch.pow(torch.clamp(margin - distance, min=0.0), 2)
        return loss.mean()
    
    def style_classification_loss(self, style_logits, style_labels):
        """風格分類損失（輔助任務）"""
        return F.cross_entropy(style_logits, style_labels)
    
    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0.0
        total_contrastive_loss = 0.0
        total_style_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            try:
                self.optimizer.zero_grad()
                
                features1 = batch['features1'].to(self.device)
                features2 = batch['features2'].to(self.device)
                style1 = batch['style1'].to(self.device)
                style2 = batch['style2'].to(self.device)
                label = batch['label'].to(self.device)
                
                # 前向傳播
                outputs1 = self.model(features1)
                outputs2 = self.model(features2)
                
                emb1 = outputs1['fashion_embedding']
                emb2 = outputs2['fashion_embedding']
                style_logits1 = outputs1['style_logits']
                style_logits2 = outputs2['style_logits']
                
                # 對比學習損失 - 主要任務
                contrastive_loss = self.contrastive_loss(emb1, emb2, label)
                
                # 風格分類損失 - 輔助任務
                style_loss = (self.style_classification_loss(style_logits1, style1) + 
                            self.style_classification_loss(style_logits2, style2)) / 2
                
                # 總損失
                total_batch_loss = contrastive_loss + 0.3 * style_loss
                
                # 反向傳播
                total_batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss += total_batch_loss.item()
                total_contrastive_loss += contrastive_loss.item()
                total_style_loss += style_loss.item()
                num_batches += 1
                
                if batch_idx % 20 == 0:
                    print(f"  批次 {batch_idx}/{len(dataloader)}")
                    print(f"    對比損失: {contrastive_loss.item():.4f}")
                    print(f"    風格損失: {style_loss.item():.4f}")
                
                # 記憶體管理
                if self.device.type == 'mps':
                    torch.mps.empty_cache()
                
            except Exception as e:
                print(f"❌ 批次 {batch_idx} 訓練失敗: {e}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_contrastive = total_contrastive_loss / max(num_batches, 1)
        avg_style = total_style_loss / max(num_batches, 1)
        
        return {
            'total_loss': avg_loss,
            'contrastive_loss': avg_contrastive,
            'style_loss': avg_style
        }
    
    def save_checkpoint(self, epoch, losses, save_path):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'losses': losses,
            'config': self.config.__dict__
        }
        torch.save(checkpoint, save_path)
        print(f"💾 檢查點已保存: {save_path}")
        
        if losses['total_loss'] < self.best_loss:
            self.best_loss = losses['total_loss']
            best_path = save_path.replace('.pth', '_best.pth')
            torch.save(checkpoint, best_path)
            print(f"🏆 最佳模型已保存: {best_path}")

# ==================== 主訓練函數 ====================

def train_simple_model(data_root, config_type="balanced", resume_from=None):
    """
    簡化版模型訓練
    """
    device = setup_mac_optimization()
    config = SimpleTrainingConfig(config_type)
    
    print(f"🚀 開始簡化版穿搭推薦模型訓練")
    print(f"📊 配置: {config_type}")
    print(f"🎯 設備: {device}")
    print(f"🔧 特色: 純 FashionCLIP 語義特徵")
    
    # 檢查並生成簡化的數據集標籤
    labels_file = "simple_dataset_labels.json"
    should_regenerate = False
    
    if os.path.exists(labels_file):
        # 檢查現有文件是否有效
        try:
            with open(labels_file, 'r') as f:
                existing_labels = json.load(f)
            
            if not existing_labels or len(existing_labels) == 0:
                print("⚠️ 現有標籤文件為空，需要重新生成")
                should_regenerate = True
            else:
                # 檢查標籤是否匹配當前風格類別
                existing_styles = set(label.get('style', '') for label in existing_labels)
                expected_styles = set(config.style_categories)
                
                if not existing_styles.intersection(expected_styles):
                    print("⚠️ 現有標籤與當前風格類別不匹配，需要重新生成")
                    print(f"   現有風格: {existing_styles}")
                    print(f"   期望風格: {expected_styles}")
                    should_regenerate = True
                else:
                    print(f"✅ 使用現有簡化數據集標籤 ({len(existing_labels)} 個樣本)")
                    
        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠️ 現有標籤文件格式錯誤: {e}")
            should_regenerate = True
    else:
        should_regenerate = True
    
    if should_regenerate:
        print("📝 生成簡化數據集標籤...")
        # 刪除舊文件（如果存在）
        if os.path.exists(labels_file):
            os.remove(labels_file)
        generate_simple_dataset_labels(data_root, config, device)
    
    # 創建數據集和數據載入器
    try:
        dataset = SimpleOutfitDataset()
        
        if len(dataset) == 0:
            print("❌ 數據集為空！請檢查：")
            print(f"   1. 數據集路徑是否正確: {data_root}")
            print(f"   2. 資料夾命名是否正確 (例: Artsy_MEN, Artsy_WOMEN)")
            print(f"   3. 圖片文件是否存在")
            return
            
        print(f"📦 數據集大小: {len(dataset)} 個樣本對")
        
        dataloader = DataLoader(
            dataset, 
            batch_size=config.batch_size, 
            shuffle=True, 
            num_workers=0,  # 簡化為單線程
            pin_memory=True if device.type == 'mps' else False
        )
        
    except Exception as e:
        print(f"❌ 創建數據載入器失敗: {e}")
        return
    
    # 創建模型和訓練器
    model = SimpleFashionRecommender(config).to(device)
    trainer = SimpleFashionTrainer(model, config, device)
    
    start_epoch = 0
    if resume_from and os.path.exists(resume_from):
        print(f"🔄 從檢查點恢復訓練: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"✅ 恢復完成，從第 {start_epoch} 輪開始")
    
    # 開始訓練
    print(f"🎯 開始訓練 {config.num_epochs} 輪...")
    for epoch in range(start_epoch, config.num_epochs):
        print(f"\n=== 第 {epoch+1}/{config.num_epochs} 輪 ===")
        start_time = time.time()
        
        losses = trainer.train_epoch(dataloader, epoch)
        trainer.scheduler.step()
        
        epoch_time = time.time() - start_time
        print(f"✅ 第 {epoch+1} 輪完成")
        print(f"📊 總損失: {losses['total_loss']:.4f}")
        print(f"🔗 對比損失: {losses['contrastive_loss']:.4f}")
        print(f"🎨 風格損失: {losses['style_loss']:.4f}")
        print(f"⏱️  用時: {epoch_time:.2f}秒")
        
        # 保存檢查點
        if (epoch + 1) % 3 == 0:
            save_path = f"simple_fashion_model_epoch_{epoch+1}.pth"
            trainer.save_checkpoint(epoch, losses, save_path)
    
    # 保存最終模型
    final_path = "simple_fashion_model_final.pth"
    trainer.save_checkpoint(config.num_epochs-1, losses, final_path)
    
    print(f"\n🎉 簡化版訓練完成！")
    print(f"📁 模型已保存: {final_path}")
    print(f"🏆 最佳模型: simple_fashion_model_final_best.pth")
    print(f"✨ 特色: 純淨的 FashionCLIP 語義特徵空間")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='簡化版穿搭推薦模型訓練')
    parser.add_argument('--data_root', type=str, required=True, help='數據集根目錄')
    parser.add_argument('--config', type=str, default='balanced', 
                       choices=['minimal', 'balanced', 'performance'], help='訓練配置')
    parser.add_argument('--resume', type=str, help='恢復訓練的檢查點路徑')
    
    args = parser.parse_args()
    train_simple_model(args.data_root, args.config, args.resume) 