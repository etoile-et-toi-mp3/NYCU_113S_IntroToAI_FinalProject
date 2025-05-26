#!/usr/bin/env python3
"""
Mac 訓練模型測試腳本
用於測試 .pth 模型文件對 test.jpg 的預測結果
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import argparse
import json
from datetime import datetime

# 導入模型類
from test_mac_optimized import LightweightStyleClassifier, setup_mac_optimization

class ModelTester:
    def __init__(self, model_path, device=None):
        """
        初始化模型測試器
        
        Args:
            model_path: 模型文件路徑 (.pth)
            device: 計算設備，None則自動選擇
        """
        self.model_path = model_path
        
        # 設備設置
        if device is None:
            self.device = setup_mac_optimization()
        else:
            self.device = device
        
        # 風格類別
        self.style_categories = [
            'Athleisure', 'BRITISH', 'CASUAL', 'GOTH', 'Kawaii', 
            'Korean', 'MINIMALIST', 'Preppy', 'STREET', 'Streetwear', 
            'Vintage', 'Y2K'
        ]
        
        self.gender_categories = ['MEN', 'WOMEN']
        
        # 圖片預處理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 載入模型
        self.model = self._load_model()
        
    def _load_model(self):
        """載入模型"""
        try:
            print(f"📂 載入模型: {self.model_path}")
            
            # 初始化模型
            model = LightweightStyleClassifier(
                num_styles=len(self.style_categories),
                num_genders=len(self.gender_categories),
                feature_dim=1024
            )
            
            # 載入權重
            if self.model_path.endswith('.pth'):
                state_dict = torch.load(self.model_path, map_location=self.device)
                
                # 如果是檢查點文件，提取模型狀態
                if 'model_state_dict' in state_dict:
                    model.load_state_dict(state_dict['model_state_dict'])
                    print("✅ 從檢查點載入模型")
                    
                    # 顯示檢查點信息
                    if 'epoch' in state_dict:
                        print(f"   Epoch: {state_dict['epoch'] + 1}")
                    if 'loss' in state_dict:
                        print(f"   損失: {state_dict['loss']:.4f}")
                else:
                    model.load_state_dict(state_dict)
                    print("✅ 載入模型權重")
            else:
                raise ValueError("不支援的模型文件格式")
            
            model.to(self.device)
            model.eval()
            
            # 計算模型參數
            total_params = sum(p.numel() for p in model.parameters())
            print(f"📈 模型參數數量: {total_params:,}")
            
            return model
            
        except Exception as e:
            print(f"❌ 載入模型失敗: {e}")
            raise
    
    def preprocess_image(self, image_path):
        """預處理圖片"""
        try:
            # 載入圖片
            image = Image.open(image_path).convert('RGB')
            print(f"📷 載入圖片: {image_path}")
            print(f"   原始尺寸: {image.size}")
            
            # 預處理
            input_tensor = self.transform(image).unsqueeze(0)
            print(f"   處理後尺寸: {input_tensor.shape}")
            
            return input_tensor.to(self.device), image
            
        except Exception as e:
            print(f"❌ 圖片預處理失敗: {e}")
            raise
    
    def predict(self, image_path, top_k=5):
        """
        預測圖片的風格和性別
        
        Args:
            image_path: 圖片路徑
            top_k: 返回前k個預測結果
            
        Returns:
            dict: 預測結果
        """
        try:
            # 預處理圖片
            input_tensor, original_image = self.preprocess_image(image_path)
            
            print(f"\n🔮 開始預測...")
            print(f"   使用設備: {self.device}")
            
            # 模型推論
            with torch.no_grad():
                outputs = self.model(input_tensor)
                
                # 風格預測
                style_logits = outputs['style_logits']
                style_probs = F.softmax(style_logits, dim=1)
                style_probs_np = style_probs.cpu().numpy()[0]
                
                # 性別預測
                gender_logits = outputs['gender_logits']
                gender_probs = F.softmax(gender_logits, dim=1)
                gender_probs_np = gender_probs.cpu().numpy()[0]
                
                # 獲取top-k風格預測
                style_top_indices = np.argsort(style_probs_np)[::-1][:top_k]
                style_predictions = []
                
                for i, idx in enumerate(style_top_indices):
                    style_predictions.append({
                        'rank': i + 1,
                        'style': self.style_categories[idx],
                        'confidence': float(style_probs_np[idx]),
                        'percentage': float(style_probs_np[idx] * 100)
                    })
                
                # 獲取性別預測
                gender_top_indices = np.argsort(gender_probs_np)[::-1]
                gender_predictions = []
                
                for i, idx in enumerate(gender_top_indices):
                    gender_predictions.append({
                        'rank': i + 1,
                        'gender': self.gender_categories[idx],
                        'confidence': float(gender_probs_np[idx]),
                        'percentage': float(gender_probs_np[idx] * 100)
                    })
                
                # 組合結果
                result = {
                    'image_path': image_path,
                    'image_size': original_image.size,
                    'device': str(self.device),
                    'model_path': self.model_path,
                    'timestamp': datetime.now().isoformat(),
                    'style_predictions': style_predictions,
                    'gender_predictions': gender_predictions,
                    'top_style': style_predictions[0]['style'],
                    'top_style_confidence': style_predictions[0]['confidence'],
                    'top_gender': gender_predictions[0]['gender'],
                    'top_gender_confidence': gender_predictions[0]['confidence']
                }
                
                return result
                
        except Exception as e:
            print(f"❌ 預測失敗: {e}")
            raise
    
    def print_results(self, result, detailed=True):
        """打印預測結果"""
        print(f"\n{'='*60}")
        print(f"🎯 預測結果")
        print(f"{'='*60}")
        
        print(f"📷 圖片: {result['image_path']}")
        print(f"📐 尺寸: {result['image_size']}")
        print(f"🔧 設備: {result['device']}")
        print(f"📅 時間: {result['timestamp']}")
        
        print(f"\n🎨 風格預測:")
        print(f"   最可能: {result['top_style']} ({result['top_style_confidence']:.1%})")
        
        if detailed:
            print(f"\n   詳細排名:")
            for pred in result['style_predictions']:
                confidence_bar = "█" * int(pred['percentage'] / 5)
                print(f"   {pred['rank']}. {pred['style']:<12} {pred['percentage']:5.1f}% {confidence_bar}")
        
        print(f"\n👤 性別預測:")
        print(f"   最可能: {result['top_gender']} ({result['top_gender_confidence']:.1%})")
        
        if detailed:
            print(f"\n   詳細排名:")
            for pred in result['gender_predictions']:
                confidence_bar = "█" * int(pred['percentage'] / 5)
                print(f"   {pred['rank']}. {pred['gender']:<6} {pred['percentage']:5.1f}% {confidence_bar}")
        
        print(f"\n{'='*60}")
    
    def save_results(self, result, output_path):
        """保存結果到JSON文件"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"💾 結果已保存到: {output_path}")
        except Exception as e:
            print(f"❌ 保存結果失敗: {e}")

def main():
    parser = argparse.ArgumentParser(description='測試訓練好的模型')
    parser.add_argument('--model', type=str, required=True,
                       help='模型文件路徑 (.pth)')
    parser.add_argument('--image', type=str, default='test.jpg',
                       help='測試圖片路徑 (default: test.jpg)')
    parser.add_argument('--top-k', type=int, default=5,
                       help='顯示前k個預測結果 (default: 5)')
    parser.add_argument('--output', type=str, default=None,
                       help='保存結果的JSON文件路徑')
    parser.add_argument('--simple', action='store_true',
                       help='簡化輸出，只顯示最佳預測')
    
    args = parser.parse_args()
    
    print("🍎 Mac 模型測試器")
    print("=" * 50)
    
    # 檢查文件是否存在
    if not os.path.exists(args.model):
        print(f"❌ 模型文件不存在: {args.model}")
        return
    
    if not os.path.exists(args.image):
        print(f"❌ 圖片文件不存在: {args.image}")
        return
    
    try:
        # 初始化測試器
        tester = ModelTester(args.model)
        
        # 進行預測
        result = tester.predict(args.image, top_k=args.top_k)
        
        # 顯示結果
        tester.print_results(result, detailed=not args.simple)
        
        # 保存結果
        if args.output:
            tester.save_results(result, args.output)
        
        # 簡單總結
        print(f"\n📋 總結:")
        print(f"   風格: {result['top_style']} ({result['top_style_confidence']:.1%})")
        print(f"   性別: {result['top_gender']} ({result['top_gender_confidence']:.1%})")
        
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()

def quick_test(model_path, image_path="test.jpg"):
    """快速測試函數"""
    try:
        tester = ModelTester(model_path)
        result = tester.predict(image_path)
        
        print(f"🎯 快速測試結果:")
        print(f"   圖片: {image_path}")
        print(f"   風格: {result['top_style']} ({result['top_style_confidence']:.1%})")
        print(f"   性別: {result['top_gender']} ({result['top_gender_confidence']:.1%})")
        
        return result
    except Exception as e:
        print(f"❌ 快速測試失敗: {e}")
        return None

if __name__ == "__main__":
    main() 