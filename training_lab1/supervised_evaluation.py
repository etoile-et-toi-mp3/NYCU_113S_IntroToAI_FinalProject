#!/usr/bin/env python3
"""
時尚AI分類準確率測試系統
測試不同backbone模型的分類準確率並生成詳細報告
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
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse
from collections import defaultdict
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ==================== 設備優化設置 ====================

def setup_device(platform="auto"):
    """設置計算設備"""
    if platform == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"✅ 使用 CUDA: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("✅ 使用 Metal Performance Shaders (MPS)")
        else:
            device = torch.device("cpu")
            print("💻 使用 CPU")
    elif platform == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif platform == "mps":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    
    return device

# ==================== 模型架構定義 ====================

class FashionBackbone(nn.Module):
    """Fashion Backbone（與訓練時相同）"""
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
    """增強風格分類器"""
    def __init__(self, backbone_type='mobilenet', num_styles=11, num_genders=2, feature_dim=1024):
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

# ==================== 測試數據載入 ====================

class TestDataLoader:
    """測試數據載入器"""
    def __init__(self, test_data_dir):
        self.test_data_dir = test_data_dir
        self.style_categories = [
            'Artsy', 'Athleisure', 'BRITISH', 'CASUAL', 'GOTH', 'Japanese',
            'Kawaii', 'Korean', 'Preppy', 'STREET', 'Vintage'
        ]
        self.gender_categories = ['MEN', 'WOMEN']
        
        # 創建類別到索引的映射
        all_categories = []
        for style in sorted(self.style_categories):
            for gender in sorted(self.gender_categories):
                all_categories.append(f"{style}_{gender}")
        
        self.category_to_idx = {cat: idx for idx, cat in enumerate(all_categories)}
        self.idx_to_category = {idx: cat for cat, idx in self.category_to_idx.items()}
        
        # 載入測試數據
        self.test_images = self._load_test_images()
        
    def _load_test_images(self):
        """載入測試圖片"""
        test_images = []
        
        if not os.path.exists(self.test_data_dir):
            raise ValueError(f"測試數據目錄不存在: {self.test_data_dir}")
        
        # 按類別名稱排序掃描圖片
        all_files = []
        for filename in os.listdir(self.test_data_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                all_files.append(filename)
        
        # 按檔名排序
        all_files.sort()
        
        for filename in all_files:
            # 解析檔名獲取類別
            category = self._parse_category_from_filename(filename)
            if category and category in self.category_to_idx:
                file_path = os.path.join(self.test_data_dir, filename)
                test_images.append({
                    'path': file_path,
                    'filename': filename,
                    'category': category,
                    'category_idx': self.category_to_idx[category],
                    'style': category.split('_')[0],
                    'gender': category.split('_')[1]
                })
        
        print(f"📊 載入了 {len(test_images)} 張測試圖片")
        
        # 統計每個類別的圖片數量
        category_counts = defaultdict(int)
        for img in test_images:
            category_counts[img['category']] += 1
        
        print("📈 各類別測試圖片數量:")
        for category in sorted(category_counts.keys()):
            print(f"  {category}: {category_counts[category]} 張")
        
        return test_images
    
    def _parse_category_from_filename(self, filename):
        """從檔名解析類別"""
        # 移除副檔名
        base_name = os.path.splitext(filename)[0]
        
        # 嘗試匹配已知的類別模式
        for style in self.style_categories:
            for gender in self.gender_categories:
                category = f"{style}_{gender}"
                if base_name.startswith(category):
                    return category
        
        # 如果沒有直接匹配，嘗試其他解析方式
        parts = base_name.split('_')
        if len(parts) >= 2:
            potential_style = parts[0]
            potential_gender = parts[1]
            
            if potential_style in self.style_categories and potential_gender in self.gender_categories:
                return f"{potential_style}_{potential_gender}"
        
        print(f"⚠️ 無法解析檔名的類別: {filename}")
        return None

# ==================== 模型測試器 ====================

class ModelTester:
    """模型測試器"""
    def __init__(self, device):
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def load_model(self, model_path, backbone_type):
        """載入模型"""
        print(f"📂 載入模型: {model_path}")
        
        # 創建模型
        model = EnhancedStyleClassifier(
            backbone_type=backbone_type,
            num_styles=11,
            num_genders=2,
            feature_dim=1024
        )
        
        try:
            # 載入權重
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                epoch = checkpoint.get('epoch', '未知')
                print(f"✅ 載入checkpoint，epoch: {epoch}")
            else:
                model.load_state_dict(checkpoint)
                print("✅ 載入state_dict")
            
            model.to(self.device)
            model.eval()
            
            return model
            
        except Exception as e:
            print(f"❌ 載入模型失敗: {e}")
            raise
    
    def predict_image(self, model, image_path):
        """預測單張圖片"""
        try:
            # 載入和預處理圖片
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # 推理
            with torch.no_grad():
                outputs = model(input_tensor)
                
                # 獲取預測結果
                style_logits = outputs['style_logits']
                gender_logits = outputs['gender_logits']
                
                style_probs = F.softmax(style_logits, dim=1)
                gender_probs = F.softmax(gender_logits, dim=1)
                
                predicted_style_idx = torch.argmax(style_probs, dim=1).item()
                predicted_gender_idx = torch.argmax(gender_probs, dim=1).item()
                
                style_confidence = style_probs[0][predicted_style_idx].item()
                gender_confidence = gender_probs[0][predicted_gender_idx].item()
                
                return {
                    'predicted_style_idx': predicted_style_idx,
                    'predicted_gender_idx': predicted_gender_idx,
                    'style_confidence': style_confidence,
                    'gender_confidence': gender_confidence,
                    'style_probs': style_probs.cpu().numpy()[0],
                    'gender_probs': gender_probs.cpu().numpy()[0]
                }
                
        except Exception as e:
            print(f"❌ 預測失敗 {image_path}: {e}")
            return None
    
    def test_model(self, model, test_data_loader, backbone_type):
        """測試模型準確率"""
        print(f"🧪 開始測試 {backbone_type} 模型...")
        
        results = {
            'backbone': backbone_type,
            'total_images': len(test_data_loader.test_images),
            'correct_predictions': 0,
            'style_correct': 0,
            'gender_correct': 0,
            'category_results': defaultdict(lambda: {'total': 0, 'correct': 0}),
            'detailed_results': [],
            'confusion_matrix_data': []
        }
        
        style_categories = test_data_loader.style_categories
        gender_categories = test_data_loader.gender_categories
        
        for i, test_image in enumerate(test_data_loader.test_images):
            print(f"  處理 {i+1}/{len(test_data_loader.test_images)}: {test_image['filename']}")
            
            # 預測
            prediction = self.predict_image(model, test_image['path'])
            if prediction is None:
                continue
            
            # 真實標籤
            true_style = test_image['style']
            true_gender = test_image['gender']
            true_style_idx = style_categories.index(true_style)
            true_gender_idx = gender_categories.index(true_gender)
            
            # 預測標籤
            pred_style_idx = prediction['predicted_style_idx']
            pred_gender_idx = prediction['predicted_gender_idx']
            pred_style = style_categories[pred_style_idx]
            pred_gender = gender_categories[pred_gender_idx]
            
            # 判斷準確性
            style_correct = (pred_style_idx == true_style_idx)
            gender_correct = (pred_gender_idx == true_gender_idx)
            overall_correct = style_correct and gender_correct
            
            # 更新統計
            if style_correct:
                results['style_correct'] += 1
            if gender_correct:
                results['gender_correct'] += 1
            if overall_correct:
                results['correct_predictions'] += 1
            
            # 類別統計
            category = test_image['category']
            results['category_results'][category]['total'] += 1
            if overall_correct:
                results['category_results'][category]['correct'] += 1
            
            # 詳細結果
            detail = {
                'filename': test_image['filename'],
                'true_category': category,
                'true_style': true_style,
                'true_gender': true_gender,
                'pred_style': pred_style,
                'pred_gender': pred_gender,
                'style_correct': style_correct,
                'gender_correct': gender_correct,
                'overall_correct': overall_correct,
                'style_confidence': prediction['style_confidence'],
                'gender_confidence': prediction['gender_confidence']
            }
            results['detailed_results'].append(detail)
            
            # 混淆矩陣數據
            results['confusion_matrix_data'].append({
                'true_category_idx': test_image['category_idx'],
                'pred_category_idx': pred_style_idx * 2 + pred_gender_idx,  # 簡化的組合索引
                'true_style_idx': true_style_idx,
                'pred_style_idx': pred_style_idx,
                'true_gender_idx': true_gender_idx,
                'pred_gender_idx': pred_gender_idx
            })
        
        # 計算準確率
        total = results['total_images']
        results['overall_accuracy'] = results['correct_predictions'] / total if total > 0 else 0
        results['style_accuracy'] = results['style_correct'] / total if total > 0 else 0
        results['gender_accuracy'] = results['gender_correct'] / total if total > 0 else 0
        
        print(f"✅ {backbone_type} 測試完成")
        print(f"📊 整體準確率: {results['overall_accuracy']:.4f} ({results['correct_predictions']}/{total})")
        print(f"🎨 風格準確率: {results['style_accuracy']:.4f} ({results['style_correct']}/{total})")
        print(f"👤 性別準確率: {results['gender_accuracy']:.4f} ({results['gender_correct']}/{total})")
        
        return results

# ==================== 結果分析和視覺化 ====================

class ResultAnalyzer:
    """結果分析器"""
    def __init__(self, output_dir="test_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 創建日誌文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(output_dir, f"test_log_{timestamp}.txt")
        self.csv_file = os.path.join(output_dir, f"accuracy_results_{timestamp}.csv")
        
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"分類準確率測試日誌\n")
            f.write(f"開始時間: {datetime.now().isoformat()}\n")
            f.write("="*80 + "\n")
    
    def log_message(self, message):
        """記錄日誌"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_msg + "\n")
    
    def save_results_to_csv(self, all_results):
        """保存結果到CSV"""
        self.log_message(f"💾 保存結果到CSV: {self.csv_file}")
        
        # 準備CSV數據
        csv_data = []
        for result in all_results:
            row = {
                'Backbone': result['backbone'],
                'Overall_Accuracy': result['overall_accuracy'],
                'Style_Accuracy': result['style_accuracy'],
                'Gender_Accuracy': result['gender_accuracy'],
                'Total_Images': result['total_images'],
                'Correct_Predictions': result['correct_predictions'],
                'Style_Correct': result['style_correct'],
                'Gender_Correct': result['gender_correct']
            }
            
            # 添加每個類別的準確率
            for category, cat_result in result['category_results'].items():
                category_accuracy = cat_result['correct'] / cat_result['total'] if cat_result['total'] > 0 else 0
                row[f'{category}_Accuracy'] = category_accuracy
                row[f'{category}_Correct'] = cat_result['correct']
                row[f'{category}_Total'] = cat_result['total']
            
            csv_data.append(row)
        
        # 保存到CSV
        df = pd.DataFrame(csv_data)
        df.to_csv(self.csv_file, index=False, encoding='utf-8-sig')
        
        self.log_message(f"✅ CSV文件已保存: {self.csv_file}")
        return self.csv_file
    
    def create_visualizations(self, all_results):
        """創建視覺化圖表"""
        self.log_message("📊 開始創建視覺化圖表...")
        
        # 設置中文字體
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. 整體準確率比較
        self._plot_overall_accuracy(all_results)
        
        # 2. 各類別準確率熱力圖
        self._plot_category_heatmap(all_results)
        
        # 3. 風格vs性別準確率散佈圖
        self._plot_style_gender_scatter(all_results)
        
        # 4. 詳細準確率柱狀圖
        self._plot_detailed_accuracy_bars(all_results)
        
        self.log_message("✅ 視覺化圖表創建完成")
    
    def _plot_overall_accuracy(self, all_results):
        """繪製整體準確率比較圖"""
        backbones = [r['backbone'] for r in all_results]
        overall_acc = [r['overall_accuracy'] for r in all_results]
        style_acc = [r['style_accuracy'] for r in all_results]
        gender_acc = [r['gender_accuracy'] for r in all_results]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(backbones))
        width = 0.25
        
        bars1 = ax.bar(x - width, overall_acc, width, label='Overall Accuracy', alpha=0.8)
        bars2 = ax.bar(x, style_acc, width, label='Style Accuracy', alpha=0.8)
        bars3 = ax.bar(x + width, gender_acc, width, label='Gender Accuracy', alpha=0.8)
        
        ax.set_xlabel('Backbone')
        ax.set_ylabel('Accuracy')
        ax.set_title('Classification Accuracy Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(backbones, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 添加數值標籤
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, 'overall_accuracy_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.log_message(f"📊 整體準確率比較圖已保存: {plot_path}")
    
    def _plot_category_heatmap(self, all_results):
        """繪製各類別準確率熱力圖"""
        # 準備數據
        backbones = [r['backbone'] for r in all_results]
        
        # 獲取所有類別
        all_categories = set()
        for result in all_results:
            all_categories.update(result['category_results'].keys())
        all_categories = sorted(list(all_categories))
        
        # 創建準確率矩陣
        accuracy_matrix = []
        for result in all_results:
            row = []
            for category in all_categories:
                if category in result['category_results']:
                    cat_result = result['category_results'][category]
                    accuracy = cat_result['correct'] / cat_result['total'] if cat_result['total'] > 0 else 0
                else:
                    accuracy = 0
                row.append(accuracy)
            accuracy_matrix.append(row)
        
        # 繪製熱力圖
        fig, ax = plt.subplots(figsize=(16, 10))
        
        im = ax.imshow(accuracy_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # 設置軸標籤
        ax.set_xticks(np.arange(len(all_categories)))
        ax.set_yticks(np.arange(len(backbones)))
        ax.set_xticklabels(all_categories, rotation=45, ha='right')
        ax.set_yticklabels(backbones)
        
        # 添加數值標籤
        for i in range(len(backbones)):
            for j in range(len(all_categories)):
                text = ax.text(j, i, f'{accuracy_matrix[i][j]:.3f}',
                              ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title("Category-wise Accuracy Heatmap")
        fig.colorbar(im, ax=ax, label='Accuracy')
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, 'category_accuracy_heatmap.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.log_message(f"📊 類別準確率熱力圖已保存: {plot_path}")
    
    def _plot_style_gender_scatter(self, all_results):
        """繪製風格vs性別準確率散佈圖"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        backbones = []
        style_accs = []
        gender_accs = []
        overall_accs = []
        
        for result in all_results:
            backbones.append(result['backbone'])
            style_accs.append(result['style_accuracy'])
            gender_accs.append(result['gender_accuracy'])
            overall_accs.append(result['overall_accuracy'])
        
        # 使用整體準確率作為顏色映射
        scatter = ax.scatter(style_accs, gender_accs, c=overall_accs, 
                           cmap='viridis', s=100, alpha=0.7)
        
        # 添加標籤
        for i, backbone in enumerate(backbones):
            ax.annotate(backbone, (style_accs[i], gender_accs[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax.set_xlabel('Style Accuracy')
        ax.set_ylabel('Gender Accuracy')
        ax.set_title('Style vs Gender Accuracy')
        ax.grid(True, alpha=0.3)
        
        # 添加顏色條
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Overall Accuracy')
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, 'style_gender_scatter.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.log_message(f"📊 風格性別散佈圖已保存: {plot_path}")
    
    def _plot_detailed_accuracy_bars(self, all_results):
        """繪製詳細準確率柱狀圖"""
        # 為每個backbone創建子圖
        n_backbones = len(all_results)
        fig, axes = plt.subplots(2, (n_backbones + 1) // 2, figsize=(20, 12))
        if n_backbones == 1:
            axes = [axes]
        elif n_backbones <= 2:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for idx, result in enumerate(all_results):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            # 準備數據
            categories = sorted(result['category_results'].keys())
            accuracies = []
            
            for category in categories:
                cat_result = result['category_results'][category]
                accuracy = cat_result['correct'] / cat_result['total'] if cat_result['total'] > 0 else 0
                accuracies.append(accuracy)
            
            # 繪製柱狀圖
            bars = ax.bar(range(len(categories)), accuracies, alpha=0.7)
            
            # 設置標籤
            ax.set_title(f'{result["backbone"]} - Category Accuracy')
            ax.set_xlabel('Category')
            ax.set_ylabel('Accuracy')
            ax.set_xticks(range(len(categories)))
            ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
            
            # 添加數值標籤
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=7)
            
            # 添加平均線
            avg_accuracy = np.mean(accuracies)
            ax.axhline(y=avg_accuracy, color='red', linestyle='--', alpha=0.7,
                      label=f'Avg: {avg_accuracy:.3f}')
            ax.legend()
        
        # 隱藏多餘的子圖
        for idx in range(len(all_results), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, 'detailed_category_accuracy.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.log_message(f"📊 詳細類別準確率圖已保存: {plot_path}")
    
    def generate_summary_report(self, all_results):
        """生成總結報告"""
        self.log_message("📋 生成總結報告...")
        
        report_path = os.path.join(self.output_dir, 'summary_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("分類準確率測試總結報告\n")
            f.write("="*80 + "\n")
            f.write(f"測試時間: {datetime.now().isoformat()}\n")
            f.write(f"測試的Backbone數量: {len(all_results)}\n")
            f.write(f"每個模型的測試圖片數量: {all_results[0]['total_images'] if all_results else 0}\n")
            f.write("\n")
            
            # 整體排名
            f.write("📊 整體準確率排名:\n")
            f.write("-" * 50 + "\n")
            sorted_results = sorted(all_results, key=lambda x: x['overall_accuracy'], reverse=True)
            for i, result in enumerate(sorted_results, 1):
                f.write(f"{i:2d}. {result['backbone']:15s} - {result['overall_accuracy']:.4f} "
                       f"({result['correct_predictions']}/{result['total_images']})\n")
            f.write("\n")
            
            # 風格準確率排名
            f.write("🎨 風格準確率排名:\n")
            f.write("-" * 50 + "\n")
            sorted_by_style = sorted(all_results, key=lambda x: x['style_accuracy'], reverse=True)
            for i, result in enumerate(sorted_by_style, 1):
                f.write(f"{i:2d}. {result['backbone']:15s} - {result['style_accuracy']:.4f} "
                       f"({result['style_correct']}/{result['total_images']})\n")
            f.write("\n")
            
            # 性別準確率排名
            f.write("👤 性別準確率排名:\n")
            f.write("-" * 50 + "\n")
            sorted_by_gender = sorted(all_results, key=lambda x: x['gender_accuracy'], reverse=True)
            for i, result in enumerate(sorted_by_gender, 1):
                f.write(f"{i:2d}. {result['backbone']:15s} - {result['gender_accuracy']:.4f} "
                       f"({result['gender_correct']}/{result['total_images']})\n")
            f.write("\n")
            
            # 詳細結果
            f.write("📋 詳細結果:\n")
            f.write("=" * 80 + "\n")
            for result in all_results:
                f.write(f"\n🔧 {result['backbone']}:\n")
                f.write(f"  整體準確率: {result['overall_accuracy']:.4f}\n")
                f.write(f"  風格準確率: {result['style_accuracy']:.4f}\n")
                f.write(f"  性別準確率: {result['gender_accuracy']:.4f}\n")
                f.write(f"  正確預測: {result['correct_predictions']}/{result['total_images']}\n")
                
                f.write("  各類別準確率:\n")
                sorted_categories = sorted(result['category_results'].items())
                for category, cat_result in sorted_categories:
                    accuracy = cat_result['correct'] / cat_result['total'] if cat_result['total'] > 0 else 0
                    f.write(f"    {category:20s}: {accuracy:.4f} ({cat_result['correct']}/{cat_result['total']})\n")
                f.write("\n")
        
        self.log_message(f"📋 總結報告已保存: {report_path}")
        return report_path

# ==================== 主程序 ====================

def main():
    parser = argparse.ArgumentParser(description='時尚AI分類準確率測試系統')
    parser.add_argument('--test_data', type=str, required=True,
                       help='測試數據目錄路徑')
    parser.add_argument('--models_dir', type=str, required=True,
                       help='模型目錄路徑')
    parser.add_argument('--output_dir', type=str, default='test_results',
                       help='結果輸出目錄')
    parser.add_argument('--platform', type=str, default='auto',
                       choices=['auto', 'cuda', 'mps', 'cpu'],
                       help='計算平台')
    parser.add_argument('--models', type=str, nargs='+',
                       help='指定要測試的模型文件（不指定則測試所有.pth文件）')
    
    args = parser.parse_args()
    
    print("🤖 時尚AI分類準確率測試系統")
    print("=" * 80)
    
    # 檢查目錄
    if not os.path.exists(args.test_data):
        print(f"❌ 測試數據目錄不存在: {args.test_data}")
        return
    
    if not os.path.exists(args.models_dir):
        print(f"❌ 模型目錄不存在: {args.models_dir}")
        return
    
    # 設置設備
    device = setup_device(args.platform)
    
    # 初始化組件
    test_data_loader = TestDataLoader(args.test_data)
    model_tester = ModelTester(device)
    result_analyzer = ResultAnalyzer(args.output_dir)
    
    # 獲取要測試的模型列表
    if args.models:
        model_files = args.models
    else:
        # 自動掃描模型目錄
        model_files = [f for f in os.listdir(args.models_dir) if f.endswith('.pth')]
        model_files.sort()
    
    if not model_files:
        print("❌ 沒有找到要測試的模型文件")
        return
    
    result_analyzer.log_message(f"🔍 找到 {len(model_files)} 個模型文件:")
    for model_file in model_files:
        result_analyzer.log_message(f"  - {model_file}")
    
    # 測試所有模型
    all_results = []
    
    for model_file in model_files:
        try:
            result_analyzer.log_message(f"\n🧪 開始測試模型: {model_file}")
            
            # 從檔名解析backbone類型
            backbone_type = extract_backbone_from_filename(model_file)
            result_analyzer.log_message(f"🔧 檢測到backbone類型: {backbone_type}")
            
            # 載入模型
            model_path = os.path.join(args.models_dir, model_file)
            model = model_tester.load_model(model_path, backbone_type)
            
            # 測試模型
            result = model_tester.test_model(model, test_data_loader, backbone_type)
            all_results.append(result)
            
            # 記錄結果
            result_analyzer.log_message(f"✅ {backbone_type} 測試完成:")
            result_analyzer.log_message(f"   整體準確率: {result['overall_accuracy']:.4f}")
            result_analyzer.log_message(f"   風格準確率: {result['style_accuracy']:.4f}")
            result_analyzer.log_message(f"   性別準確率: {result['gender_accuracy']:.4f}")
            
            # 釋放記憶體
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            elif device.type == 'mps':
                torch.mps.empty_cache()
            
        except Exception as e:
            result_analyzer.log_message(f"❌ 測試模型 {model_file} 失敗: {e}")
            continue
    
    if not all_results:
        print("❌ 沒有成功測試任何模型")
        return
    
    # 保存結果
    result_analyzer.log_message(f"\n📊 測試完成，共測試了 {len(all_results)} 個模型")
    
    # 保存CSV結果
    csv_file = result_analyzer.save_results_to_csv(all_results)
    
    # 創建視覺化
    result_analyzer.create_visualizations(all_results)
    
    # 生成總結報告
    summary_report = result_analyzer.generate_summary_report(all_results)
    
    # 顯示最終結果
    print(f"\n🎉 所有測試完成！")
    print(f"📁 結果目錄: {args.output_dir}")
    print(f"📊 CSV結果: {csv_file}")
    print(f"📋 總結報告: {summary_report}")
    print(f"📈 視覺化圖表已保存到結果目錄")
    
    # 顯示排名
    print(f"\n🏆 整體準確率排名:")
    sorted_results = sorted(all_results, key=lambda x: x['overall_accuracy'], reverse=True)
    for i, result in enumerate(sorted_results, 1):
        print(f"  {i}. {result['backbone']:15s} - {result['overall_accuracy']:.4f}")

def extract_backbone_from_filename(filename):
    """從檔名提取backbone類型"""
    filename_lower = filename.lower()
    
    # 定義backbone類型匹配規則
    backbone_patterns = {
        'mobilenet': ['mobilenet'],
        'resnet18': ['resnet18'],
        'resnet50': ['resnet50'],
        'efficientnet_b0': ['efficientnet_b0', 'efficientnetb0'],
        'efficientnet_b2': ['efficientnet_b2', 'efficientnetb2'],
        'vit_tiny': ['vit_tiny', 'vittiny'],
        'vit_small': ['vit_small', 'vitsmall'],
        'fashion_resnet': ['fashion_resnet', 'fashionresnet']
    }
    
    for backbone_type, patterns in backbone_patterns.items():
        for pattern in patterns:
            if pattern in filename_lower:
                return backbone_type
    
    # 如果沒有匹配到，嘗試更寬鬆的匹配
    if 'resnet' in filename_lower:
        if '18' in filename_lower:
            return 'resnet18'
        elif '50' in filename_lower:
            return 'resnet50'
        else:
            return 'resnet18'  # 默認
    elif 'efficient' in filename_lower:
        if 'b2' in filename_lower:
            return 'efficientnet_b2'
        else:
            return 'efficientnet_b0'  # 默認
    elif 'vit' in filename_lower:
        if 'small' in filename_lower:
            return 'vit_small'
        else:
            return 'vit_tiny'  # 默認
    elif 'mobile' in filename_lower:
        return 'mobilenet'
    
    # 如果都沒有匹配到，返回默認值
    print(f"⚠️ 無法從檔名 {filename} 識別backbone類型，使用默認值 mobilenet")
    return 'mobilenet'

if __name__ == "__main__":
    main()