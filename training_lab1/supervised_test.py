#!/usr/bin/env python3
"""
時尚AI監督學習測試系統
使用訓練好的模型進行穿搭分析和推薦
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

# ==================== Cuda加速設置 ====================

def setup_cuda_optimization():
    """設置 NVIDIA CUDA 特定的優化"""
    if torch.cuda.is_available():
        print(f"✅ 檢測到 CUDA 支持: {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda")
        
        # CUDA 性能優化選項
        torch.backends.cudnn.benchmark = True  # 適用於輸入尺寸固定的模型
        torch.backends.cudnn.deterministic = False  # 提高速度但結果非完全可重現
    else:
        print("⚠️ CUDA 不可用，使用 CPU")
        device = torch.device("cpu")
    return device

# ==================== 模型架構定義（與訓練時相同） ====================

class FashionBackbone(nn.Module):
    """可配置的Fashion Backbone（與訓練時相同）"""
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

class EnhancedStyleClassifier(nn.Module):
    """增強風格分類器（與訓練時相同）"""
    def __init__(self, backbone_type='mobilenet', num_styles=12, num_genders=2, feature_dim=1024):
        super(EnhancedStyleClassifier, self).__init__()
        
        self.backbone = FashionBackbone(
            backbone_type=backbone_type, 
            pretrained=True
        )
        
        backbone_features = self.backbone.get_feature_dim()
        self.feature_dim = feature_dim
        
        # 特徵投影層
        self.feature_projection = nn.Sequential(
            nn.Linear(backbone_features, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 風格分類頭
        self.style_classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_styles)
        )
        
        # 性別分類頭
        self.gender_classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_genders)
        )
        
        # 投影頭（用於對比學習）
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
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

# ==================== 模型載入工具 ====================

def load_trained_model(model_path, device, backbone_type='mobilenet'):
    """載入訓練好的模型"""
    print(f"📂 載入模型: {model_path}")
    
    # 風格和性別類別
    style_categories = [
        'Artsy', 'Athleisure', 'BRITISH', 'CASUAL', 'GOTH', 'Japanese',
        'Kawaii', 'Korean', 'Preppy', 'STREET', 'Vintage'
    ]
    gender_categories = ['MEN', 'WOMEN']
    
    # 創建模型
    model = EnhancedStyleClassifier(
        backbone_type=backbone_type,
        num_styles=len(style_categories),
        num_genders=len(gender_categories),
        feature_dim=1024
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
        
        return model, style_categories, gender_categories
        
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

def predict_image(model, image_path, style_categories, gender_categories, device):
    """預測單張圖片"""
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
            outputs = model(input_tensor)
            
            # 風格預測
            style_probs = F.softmax(outputs['style_logits'], dim=1)
            style_probs_np = style_probs.cpu().numpy()[0]
            
            # 性別預測
            gender_probs = F.softmax(outputs['gender_logits'], dim=1)
            gender_probs_np = gender_probs.cpu().numpy()[0]
            
            # 獲取最佳預測
            top_style_idx = np.argmax(style_probs_np)
            top_gender_idx = np.argmax(gender_probs_np)
            
            # 計算信心度評分
            style_confidence = style_probs_np[top_style_idx]
            gender_confidence = gender_probs_np[top_gender_idx]
            
            # 綜合評分
            overall_score = (style_confidence * 0.7 + gender_confidence * 0.3) * 10
            
            result = {
                'predicted_style': style_categories[top_style_idx],
                'predicted_gender': gender_categories[top_gender_idx],
                'style_confidence': float(style_confidence),
                'gender_confidence': float(gender_confidence),
                'overall_score': float(overall_score),
                'style_probabilities': {
                    style_categories[i]: float(prob) 
                    for i, prob in enumerate(style_probs_np)
                },
                'gender_probabilities': {
                    gender_categories[i]: float(prob) 
                    for i, prob in enumerate(gender_probs_np)
                },
                'features': outputs['features'].cpu().numpy()[0],
                'projected_features': outputs['projected_features'].cpu().numpy()[0]
            }
            
            return result
            
    except Exception as e:
        print(f"❌ 預測失敗: {e}")
        raise

def extract_features_from_dataset(model, dataset_root, device, max_images_per_class=100):
    """從訓練數據集提取特徵用於相似度搜索"""
    print(f"🔨 從數據集提取特徵: {dataset_root}")
    
    style_categories = [
        'Artsy', 'Athleisure', 'BRITISH', 'CASUAL', 'GOTH', 'Japanese',
        'Kawaii', 'Korean', 'Preppy', 'STREET', 'Vintage'
    ]
    gender_categories = ['MEN', 'WOMEN']
    
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    feature_database = {
        'features': [],
        'projected_features': [],
        'image_paths': [],
        'styles': [],
        'genders': []
    }
    
    total_images = 0
    
    # 掃描數據集
    for style in style_categories:
        for gender in gender_categories:
            folder_name = f"{style}_{gender}"
            folder_path = os.path.join(dataset_root, folder_name)
            
            if not os.path.exists(folder_path):
                continue
            
            print(f"   📁 處理: {folder_name}")
            
            img_files = [f for f in os.listdir(folder_path) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            img_files = img_files[:max_images_per_class]
            
            for img_name in img_files:
                img_path = os.path.join(folder_path, img_name)
                
                try:
                    image = Image.open(img_path).convert('RGB')
                    input_tensor = transform(image).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        features = outputs['features'].cpu().numpy()[0]
                        projected_features = outputs['projected_features'].cpu().numpy()[0]
                    
                    feature_database['features'].append(features)
                    feature_database['projected_features'].append(projected_features)
                    feature_database['image_paths'].append(img_path)
                    feature_database['styles'].append(style)
                    feature_database['genders'].append(gender)
                    
                    total_images += 1
                    
                    if total_images % 50 == 0:
                        print(f"     已處理 {total_images} 張圖片...")
                        
                except Exception as e:
                    print(f"⚠️ 跳過圖片 {img_path}: {e}")
                    continue
    
    if total_images == 0:
        raise ValueError("數據集中沒有找到有效的圖片文件")
    
    feature_database['features'] = np.array(feature_database['features'])
    feature_database['projected_features'] = np.array(feature_database['projected_features'])
    
    print(f"✅ 特徵提取完成，共 {total_images} 張圖片")
    return feature_database

def find_similar_images(user_features, feature_database, top_k=10):
    """找到最相似的圖片"""
    print(f"🔍 搜索最相似的 {top_k} 張圖片...")
    
    searcher = NearestNeighbors(
        n_neighbors=min(top_k * 2, len(feature_database['projected_features'])),
        metric='cosine',
        algorithm='brute'
    )
    
    searcher.fit(feature_database['projected_features'])
    
    distances, indices = searcher.kneighbors(
        user_features['projected_features'].reshape(1, -1)
    )
    
    similar_images = []
    for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
        if i >= top_k:
            break
            
        similar_info = {
            'rank': i + 1,
            'similarity_score': float(1 - distance),
            'image_path': feature_database['image_paths'][idx],
            'style': feature_database['styles'][idx],
            'gender': feature_database['genders'][idx],
            'distance': float(distance)
        }
        
        similar_images.append(similar_info)
    
    return similar_images

def create_recommendation_report(user_prediction, similar_images, output_dir="./recommendations"):
    """創建推薦報告"""
    os.makedirs(output_dir, exist_ok=True)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'user_prediction': {
            'style': user_prediction['predicted_style'],
            'gender': user_prediction['predicted_gender'],
            'style_confidence': user_prediction['style_confidence'],
            'gender_confidence': user_prediction['gender_confidence'],
            'overall_score': user_prediction['overall_score']
        },
        'similar_images': similar_images,
        'style_analysis': {}
    }
    
    # 分析相似圖片的風格分佈
    style_distribution = defaultdict(int)
    for img in similar_images:
        style_distribution[img['style']] += 1
    
    report['style_analysis']['style_distribution'] = dict(style_distribution)
    report['style_analysis']['most_common_style'] = max(style_distribution.items(), key=lambda x: x[1])[0] if style_distribution else None
    
    # 保存報告
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(output_dir, f"recommendation_report_{timestamp}.json")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"📊 推薦報告已保存: {report_path}")
    
    return report, report_path

def print_analysis_results(user_prediction, similar_images):
    """打印分析結果"""
    print(f"\n{'='*80}")
    print(f"👗 穿搭分析結果")
    print(f"{'='*80}")
    
    # 用戶預測結果
    print(f"🎨 預測風格: {user_prediction['predicted_style']} ({user_prediction['style_confidence']:.1%})")
    print(f"👤 預測性別: {user_prediction['predicted_gender']} ({user_prediction['gender_confidence']:.1%})")
    print(f"⭐ 綜合評分: {user_prediction['overall_score']:.1f}/10")
    
    # 風格機率分佈（顯示前5名）
    style_probs = sorted(user_prediction['style_probabilities'].items(), key=lambda x: x[1], reverse=True)
    print(f"\n📊 風格機率分佈 (前5名):")
    for style, prob in style_probs[:5]:
        print(f"  {style}: {prob:.1%}")
    
    # 相似圖片
    print(f"\n🔍 最相似的穿搭 (前{min(len(similar_images), 5)}名):")
    for img in similar_images[:5]:
        print(f"  {img['rank']}. {os.path.basename(img['image_path'])}")
        print(f"     風格: {img['style']}, 性別: {img['gender']}")
        print(f"     相似度: {img['similarity_score']:.3f}")
    
    # 風格建議
    style_distribution = defaultdict(int)
    for img in similar_images:
        style_distribution[img['style']] += 1
    
    if style_distribution:
        print(f"\n💡 風格建議:")
        most_common = max(style_distribution.items(), key=lambda x: x[1])
        if most_common[0] != user_prediction['predicted_style']:
            print(f"  • 考慮嘗試 {most_common[0]} 風格，在相似穿搭中出現 {most_common[1]} 次")
        
        print(f"  • 相似穿搭風格分佈:")
        for style, count in sorted(style_distribution.items(), key=lambda x: x[1], reverse=True):
            print(f"    - {style}: {count} 張")
    
    print(f"\n{'='*80}")

# ==================== 主程序 ====================

def main():
    parser = argparse.ArgumentParser(description='時尚風格分類測試系統')
    parser.add_argument('--model', type=str, required=True,
                       help='訓練好的模型路徑')
    parser.add_argument('--image', type=str, required=True,
                       help='要分析的圖片路徑')
    parser.add_argument('--dataset', type=str, default='../dataset',
                       help='訓練數據集根目錄（用於相似度搜索）')
    parser.add_argument('--backbone', type=str, default='mobilenet',
                       choices=['mobilenet', 'resnet18', 'resnet50', 'efficientnet_b0', 
                               'efficientnet_b2', 'vit_tiny', 'vit_small', 'fashion_resnet'],
                       help='Backbone架構類型')
    parser.add_argument('--platform', type=str, default='auto',
                        choices=['mps', 'cuda', 'cpu', 'auto'],
                        help='所使用的硬體裝置')
    parser.add_argument('--output-dir', type=str, default='./recommendations',
                       help='結果輸出目錄')
    parser.add_argument('--top-k', type=int, default=10,
                       help='返回最相似的K張圖片')
    parser.add_argument('--cache-features', type=str, default='test_features_cache.pkl',
                       help='特徵緩存檔案路徑')
    parser.add_argument('--rebuild-cache', action='store_true',
                       help='重新建立特徵緩存')
    
    args = parser.parse_args()
    
    print("🤖 時尚風格分類測試系統")
    print("=" * 50)
    
    # 檢查文件
    if not os.path.exists(args.model):
        print(f"❌ 模型文件不存在: {args.model}")
        return
    
    if not os.path.exists(args.image):
        print(f"❌ 圖片文件不存在: {args.image}")
        return
    
    try:
        # 設備設置
        device = torch.device("cpu")
        if args.platform == "auto":
            device = setup_cuda_optimization() # prefer cuda first
            if device == torch.device("cpu"):
                device = setup_mac_optimization()
        else:
            if args.platform == "cuda":
                device = setup_cuda_optimization()
            elif args.platform == "mps":
                device = setup_mac_optimization()
        
        # 載入模型
        model, style_categories, gender_categories = load_trained_model(
            args.model, device, args.backbone
        )
        
        # 預測用戶圖片
        print(f"🔍 分析圖片: {args.image}")
        user_prediction = predict_image(
            model, args.image, style_categories, gender_categories, device
        )
        
        # 載入或建立特徵數據庫
        if args.rebuild_cache or not os.path.exists(args.cache_features):
            print("🔨 建立特徵數據庫...")
            feature_database = extract_features_from_dataset(
                model, args.dataset, device
            )
            
            # 保存緩存
            with open(args.cache_features, 'wb') as f:
                pickle.dump(feature_database, f)
            print(f"💾 特徵緩存已保存: {args.cache_features}")
        else:
            print(f"📁 載入特徵緩存: {args.cache_features}")
            with open(args.cache_features, 'rb') as f:
                feature_database = pickle.load(f)
        
        # 搜索相似圖片
        similar_images = find_similar_images(
            user_prediction, feature_database, args.top_k
        )
        
        # 生成報告
        report, report_path = create_recommendation_report(
            user_prediction, similar_images, args.output_dir
        )
        
        # 顯示結果
        print_analysis_results(user_prediction, similar_images)
        
        print(f"\n✅ 分析完成！報告已保存到: {report_path}")
        
    except Exception as e:
        print(f"❌ 系統運行失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 