#!/usr/bin/env python3
"""
時尚AI無監督學習測試系統
使用訓練好的無監督模型進行相似度推薦
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import timm
from transformers import ViTModel, ViTConfig
from PIL import Image
import numpy as np
import os
import json
import pickle
from datetime import datetime
from sklearn.neighbors import NearestNeighbors
import argparse
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ==================== Mac優化設置 ====================

def setup_mac_optimization():
    """設置Mac優化"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍎 使用 Metal Performance Shaders (MPS)")
    else:
        device = torch.device("cpu")
        print("💻 使用 CPU")
    return device

# ==================== 模型架構定義（與訓練時相同） ====================

class FashionBackbone(nn.Module):
    """可配置的Fashion Backbone（與訓練時相同）"""
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

class UnsupervisedStyleModel(nn.Module):
    """無監督風格模型（與訓練時相同）"""
    def __init__(self, backbone_type='resnet50', feature_dim=128):
        super(UnsupervisedStyleModel, self).__init__()
        
        self.backbone = FashionBackbone(
            backbone_type=backbone_type, 
            pretrained=True
        )
        
        backbone_features = self.backbone.get_feature_dim()
        self.feature_dim = feature_dim
        
        # 投影頭 - 與訓練時架構完全一致
        self.projection = nn.Sequential(
            nn.Linear(backbone_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, feature_dim),
            nn.BatchNorm1d(feature_dim)
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

# ==================== 模型載入工具 ====================

def load_trained_model(model_path, device, backbone_type='resnet50'):
    """載入訓練好的無監督模型"""
    print(f"📂 載入模型: {model_path}")
    
    # 創建模型
    model = UnsupervisedStyleModel(
        backbone_type=backbone_type,
        feature_dim=128
    )
    
    try:
        # 載入權重
        checkpoint = torch.load(model_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ 載入checkpoint，epoch: {checkpoint.get('epoch', '未知')}")
        else:
            model.load_state_dict(checkpoint)
            print("✅ 載入state_dict")
        
        model.to(device)
        model.eval()
        
        return model
        
    except Exception as e:
        print(f"❌ 載入模型失敗: {e}")
        print("\n🔍 檢查模型檔案資訊：")
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            if isinstance(checkpoint, dict):
                print(f"   Checkpoint keys: {list(checkpoint.keys())}")
                if 'model_state_dict' in checkpoint:
                    print(f"   Model keys (first 5): {list(checkpoint['model_state_dict'].keys())[:5]}")
            else:
                print(f"   State dict keys (first 5): {list(checkpoint.keys())[:5]}")
        except Exception as inspect_error:
            print(f"   無法檢查檔案: {inspect_error}")
        raise

# ==================== 推理功能 ====================

def extract_image_features(model, image_path, device):
    """提取單張圖片的特徵"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    try:
        # 載入和預處理圖片
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # 推理
        with torch.no_grad():
            features = model(input_tensor)
            features = F.normalize(features, dim=1)  # 歸一化特徵
            
            result = {
                'features': features.cpu().numpy()[0],
                'image_path': image_path
            }
            
            return result
            
    except Exception as e:
        print(f"❌ 特徵提取失敗: {e}")
        raise

def extract_features_from_dataset_labels(labels_path):
    """從dataset_labels.json載入特徵數據庫"""
    print(f"📁 載入特徵數據庫: {labels_path}")
    
    try:
        with open(labels_path, 'r') as f:
            dataset_labels = json.load(f)
        
        feature_database = {
            'features': [],
            'image_paths': []
        }
        
        for item in dataset_labels:
            feature_database['features'].append(item['unsupervised_features'])
            feature_database['image_paths'].append(item['path'])
        
        feature_database['features'] = np.array(feature_database['features'])
        
        print(f"✅ 載入了 {len(feature_database['features'])} 個特徵向量")
        return feature_database
        
    except Exception as e:
        print(f"❌ 載入特徵數據庫失敗: {e}")
        raise

def find_similar_images(user_features, feature_database, top_k=10):
    """找到最相似的圖片"""
    print(f"🔍 搜索最相似的 {top_k} 張圖片...")
    
    # 歸一化特徵
    user_features_norm = user_features['features'].reshape(1, -1)
    user_features_norm = user_features_norm / np.linalg.norm(user_features_norm)
    
    database_features_norm = feature_database['features']
    database_features_norm = database_features_norm / np.linalg.norm(database_features_norm, axis=1, keepdims=True)
    
    # 計算餘弦相似度
    similarities = np.dot(user_features_norm, database_features_norm.T)[0]
    
    # 找到最相似的
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    similar_images = []
    for i, idx in enumerate(top_indices):
        similar_info = {
            'rank': i + 1,
            'similarity_score': float(similarities[idx]),
            'image_path': feature_database['image_paths'][idx],
            'score': float(similarities[idx] * 10)  # 轉為0-10分
        }
        
        similar_images.append(similar_info)
    
    return similar_images

def create_recommendation_report(user_features, similar_images, output_dir="./recommendations"):
    """創建推薦報告"""
    os.makedirs(output_dir, exist_ok=True)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'user_image': user_features['image_path'],
        'similar_images': similar_images,
        'analysis': {
            'avg_similarity': float(np.mean([img['similarity_score'] for img in similar_images])),
            'max_similarity': float(max([img['similarity_score'] for img in similar_images])),
            'min_similarity': float(min([img['similarity_score'] for img in similar_images]))
        }
    }
    
    # 保存報告
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(output_dir, f"unsupervised_recommendation_report_{timestamp}.json")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"📊 推薦報告已保存: {report_path}")
    
    return report, report_path

def print_analysis_results(user_features, similar_images):
    """打印分析結果"""
    print(f"\n{'='*80}")
    print(f"🎨 無監督相似度分析結果")
    print(f"{'='*80}")
    
    # 用戶圖片信息
    print(f"📷 分析圖片: {user_features['image_path']}")
    
    # 相似圖片
    print(f"\n🔍 最相似的穿搭 (前{min(len(similar_images), 10)}名):")
    for img in similar_images[:10]:
        print(f"  {img['rank']}. {os.path.basename(img['image_path'])}")
        print(f"     相似度: {img['similarity_score']:.4f}")
        print(f"     評分: {img['score']:.2f}/10")
    
    # 統計分析
    similarities = [img['similarity_score'] for img in similar_images]
    print(f"\n📊 相似度統計:")
    print(f"  • 平均相似度: {np.mean(similarities):.4f}")
    print(f"  • 最高相似度: {np.max(similarities):.4f}")
    print(f"  • 最低相似度: {np.min(similarities):.4f}")
    print(f"  • 標準差: {np.std(similarities):.4f}")
    
    # 路徑分析
    path_distribution = defaultdict(int)
    for img in similar_images:
        # 提取風格信息（如果路徑包含風格信息）
        path_parts = img['image_path'].split('/')
        if len(path_parts) > 1:
            folder_name = path_parts[-2]  # 假設倒數第二個是風格文件夾
            path_distribution[folder_name] += 1
    
    if path_distribution:
        print(f"\n🎭 相似圖片風格分佈:")
        for style, count in sorted(path_distribution.items(), key=lambda x: x[1], reverse=True):
            print(f"  • {style}: {count} 張")
    
    print(f"\n{'='*80}")

# ==================== 主程序 ====================

def main():
    parser = argparse.ArgumentParser(description='時尚無監督學習測試系統')
    parser.add_argument('--model', type=str, required=True,
                       help='訓練好的無監督模型路徑')
    parser.add_argument('--image', type=str, required=True,
                       help='要分析的圖片路徑')
    parser.add_argument('--labels', type=str, default='dataset_labels.json',
                       help='特徵數據庫JSON文件路徑')
    parser.add_argument('--backbone', type=str, default='resnet50',
                       choices=['mobilenet', 'resnet18', 'resnet50', 'efficientnet_b0', 
                               'efficientnet_b2', 'vit_tiny', 'vit_small', 'fashion_resnet'],
                       help='Backbone架構類型')
    parser.add_argument('--output-dir', type=str, default='./recommendations',
                       help='結果輸出目錄')
    parser.add_argument('--top-k', type=int, default=10,
                       help='返回最相似的K張圖片')
    
    args = parser.parse_args()
    
    print("🤖 時尚無監督學習測試系統")
    print("=" * 50)
    
    # 檢查文件
    if not os.path.exists(args.model):
        print(f"❌ 模型文件不存在: {args.model}")
        return
    
    if not os.path.exists(args.image):
        print(f"❌ 圖片文件不存在: {args.image}")
        return
    
    if not os.path.exists(args.labels):
        print(f"❌ 特徵數據庫文件不存在: {args.labels}")
        return
    
    try:
        # 設置設備
        device = setup_mac_optimization()
        
        # 載入模型
        model = load_trained_model(args.model, device, args.backbone)
        
        # 提取用戶圖片特徵
        print(f"🔍 分析圖片: {args.image}")
        user_features = extract_image_features(model, args.image, device)
        
        # 載入特徵數據庫
        feature_database = extract_features_from_dataset_labels(args.labels)
        
        # 搜索相似圖片
        similar_images = find_similar_images(user_features, feature_database, args.top_k)
        
        # 生成報告
        report, report_path = create_recommendation_report(
            user_features, similar_images, args.output_dir
        )
        
        # 顯示結果
        print_analysis_results(user_features, similar_images)
        
        print(f"\n✅ 分析完成！報告已保存到: {report_path}")
        
    except Exception as e:
        print(f"❌ 系統運行失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 