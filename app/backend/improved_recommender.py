#!/usr/bin/env python3
"""
改進版穿搭推薦器
支援多模態模型比較用戶圖片與推薦圖片，生成結構化改進建議
"""

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import json
import os
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from transformers import (
    CLIPProcessor, CLIPModel,
    AutoProcessor, LlavaNextForConditionalGeneration,
    Blip2Processor, Blip2ForConditionalGeneration,
    InstructBlipProcessor, InstructBlipForConditionalGeneration
)
from simple_train_model import SimpleFashionRecommender, SimpleTrainingConfig
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import logging

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ImprovedFashionRecommendationSystem:
    def __init__(self, model_path="simple_fashion_model_final.pth", 
                 labels_file="simple_dataset_labels.json"):
        """
        改進版推薦系統初始化
        """
        logging.info("🚀 初始化改進版推薦系統...")
        
        # 設置設備
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        logging.info(f"🎯 使用設備: {self.device}")
        
        # 載入配置
        self.config = SimpleTrainingConfig()
        
        # 設置字體
        self._setup_fonts()
        
        # 載入模型
        logging.info("🔧 載入推薦模型...")
        self.model = SimpleFashionRecommender(self.config).to(self.device)
        self.use_trained_model = False
        
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.use_trained_model = True
                logging.info("✅ 訓練模型載入成功")
            except Exception as e:
                logging.error(f"⚠️ 載入訓練模型失敗: {e}")
        
        self.model.eval()
        
        # 載入數據集
        logging.info("📊 載入數據集...")
        with open(labels_file, 'r') as f:
            self.dataset = json.load(f)
        logging.info(f"✅ 載入 {len(self.dataset)} 個樣本")
        
        # 預處理多種特徵
        self._preprocess_multi_features()
        
        # 初始化多模態模型
        self.loaded_ai_models = {}
        self._init_ai_model_configs()
        
        # 初始化詳細的fashion-clip特徵提示詞
        self._init_fashion_features()
        
        logging.info("✅ 改進版推薦系統初始化完成！")
    
    def _init_fashion_features(self):
        """初始化詳細的fashion-clip特徵提示詞"""
        self.fashion_features = {
            'colors': [
                'red clothing', 'blue clothing', 'black clothing', 'white clothing', 
                'gray clothing', 'brown clothing', 'green clothing', 'yellow clothing',
                'pink clothing', 'purple clothing', 'orange clothing', 'beige clothing'
            ],
            'clothing_types': [
                'dress', 't-shirt', 'shirt', 'blouse', 'tank top', 'sweater', 'cardigan',
                'pants', 'jeans', 'skirt', 'shorts', 'leggings', 'jacket', 'coat',
                'blazer', 'hoodie', 'polo shirt', 'vest'
            ],
            'styles': [
                'casual style', 'formal style', 'bohemian style', 'street style',
                'vintage style', 'minimalist style', 'romantic style', 'sporty style',
                'business casual', 'elegant style', 'edgy style', 'preppy style'
            ],
            'patterns': [
                'solid color', 'striped pattern', 'floral pattern', 'plaid pattern',
                'polka dots', 'abstract pattern', 'geometric pattern', 'animal print'
            ],
            'accessories': [
                'handbag', 'backpack', 'hat', 'cap', 'scarf', 'belt', 'watch',
                'sunglasses', 'jewelry', 'necklace', 'earrings', 'bracelet'
            ],
            'shoes': [
                'sneakers', 'boots', 'high heels', 'flats', 'sandals', 'loafers',
                'pumps', 'athletic shoes', 'dress shoes', 'casual shoes'
            ],
            'materials': [
                'cotton fabric', 'denim material', 'silk fabric', 'wool material',
                'leather material', 'synthetic fabric', 'linen fabric', 'knit fabric'
            ],
            'fit': [
                'tight fitting', 'loose fitting', 'oversized clothing', 'fitted clothing',
                'relaxed fit', 'slim fit', 'regular fit', 'cropped clothing'
            ]
        }
    
    def _setup_fonts(self):
        """設置支持中文的字體"""
        try:
            available_fonts = [f.name for f in fm.fontManager.ttflist]
            preferred_fonts = ['Arial Unicode MS', 'PingFang SC', 'DejaVu Sans']
            
            selected_font = None
            for font in preferred_fonts:
                if font in available_fonts:
                    selected_font = font
                    break
            
            if selected_font:
                plt.rcParams['font.sans-serif'] = [selected_font]
            else:
                plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            
            plt.rcParams['axes.unicode_minus'] = False
        except Exception as e:
            logging.error(f"⚠️ 字體設置失敗: {e}")
    
    def _init_ai_model_configs(self):
        """初始化多模態模型配置"""
        logging.info("🧠 初始化多模態模型配置...")
        
        self.model_configs = {
            'rule_based': {
                'name': '規則系統',
                'loaded': True
            },
            'clip': {
                'name': 'FashionCLIP',
                'loaded': True
            },
            'llava': {
                'name': '視覺語言模型 (LLaVA)',
                'model_id': 'llava-hf/llava-v1.6-mistral-7b-hf',
                'loaded': False,
                'processor': None,
                'model': None
            },
            'blip2': {
                'name': '圖像描述模型 (BLIP-2)',
                'model_id': 'Salesforce/blip2-opt-2.7b',
                'loaded': False,
                'processor': None,
                'model': None
            },
            'instructblip': {
                'name': '指令圖像模型 (InstructBLIP)',
                'model_id': 'Salesforce/instructblip-vicuna-7b',
                'loaded': False,
                'processor': None,
                'model': None
            }
        }
        
        logging.info("✅ 多模態模型配置完成")
    
    def _load_ai_models(self, model_keys):
        """載入指定的多模態模型"""
        logging.info(f"🔄 載入多模態模型: {', '.join(model_keys)}")
        
        for model_key in model_keys:
            if model_key in self.loaded_ai_models:
                logging.info(f"✅ {model_key.upper()} 已載入，跳過")
                continue
                
            if model_key not in self.model_configs:
                logging.error(f"❌ 未知模型: {model_key}")
                continue
            
            config = self.model_configs[model_key]
            
            try:
                logging.info(f"🔄 載入 {config['name']} 模型...")
                
                if model_key == 'llava':
                    config['processor'] = AutoProcessor.from_pretrained(config['model_id'], use_fast=True)
                    config['model'] = LlavaNextForConditionalGeneration.from_pretrained(
                        config['model_id'],
                        torch_dtype=torch.float16 if self.device.type == 'mps' else torch.float32,
                        low_cpu_mem_usage=True
                    ).to(self.device)
                
                elif model_key == 'blip2':
                    config['processor'] = Blip2Processor.from_pretrained(config['model_id'], use_fast=True)
                    config['model'] = Blip2ForConditionalGeneration.from_pretrained(
                        config['model_id'],
                        torch_dtype=torch.float16 if self.device.type == 'mps' else torch.float32
                    ).to(self.device)
                
                elif model_key == 'instructblip':
                    config['processor'] = InstructBlipProcessor.from_pretrained(config['model_id'], use_fast=True)
                    config['model'] = InstructBlipForConditionalGeneration.from_pretrained(
                        config['model_id'],
                        torch_dtype=torch.float16 if self.device.type == 'mps' else torch.float32
                    ).to(self.device)
                
                if model_key not in ['rule_based', 'clip']:
                    config['loaded'] = True
                    self.loaded_ai_models[model_key] = config
                    logging.info(f"✅ {config['name']} 載入成功")
                
            except Exception as e:
                logging.error(f"❌ 載入 {config['name']} 失敗: {e}")
                continue
        
        logging.info(f"✅ 共載入 {len(self.loaded_ai_models)} 個多模態模型")
    
    def _preprocess_multi_features(self):
        """預處理多種特徵以改善推薦質量"""
        logging.info("🔄 預處理多級特徵...")
        
        original_features = [sample['features'] for sample in self.dataset]
        self.original_features = np.array(original_features)
        self.original_features = self.original_features / np.linalg.norm(
            self.original_features, axis=1, keepdims=True
        )
        logging.info(f"📊 原始特徵矩陣: {self.original_features.shape}")
        
        self.mapped_features = None
        if self.use_trained_model:
            logging.info("🔄 計算映射特徵...")
            with torch.no_grad():
                original_tensor = torch.tensor(self.original_features, dtype=torch.float32).to(self.device)
                mapped_features = []
                batch_size = 100
                
                for i in range(0, len(original_tensor), batch_size):
                    batch = original_tensor[i:i+batch_size]
                    outputs = self.model.forward(batch)
                    mapped_batch = outputs['fashion_embedding'].cpu().numpy()
                    mapped_features.append(mapped_batch)
                
                self.mapped_features = np.vstack(mapped_features)
                self.mapped_features = self.mapped_features / np.linalg.norm(
                    self.mapped_features, axis=1, keepdims=True
                )
                logging.info(f"📊 映射特徵矩陣: {self.mapped_features.shape}")
        
        logging.info("🔄 計算PCA降維特徵...")
        self.pca = PCA(n_components=128)
        self.pca_features = self.pca.fit_transform(self.original_features)
        self.pca_features = self.pca_features / np.linalg.norm(
            self.pca_features, axis=1, keepdims=True
        )
        logging.info(f"📊 PCA特徵矩陣: {self.pca_features.shape}")
    
    def extract_image_features(self, image_path):
        """提取查詢圖片的多種特徵"""
        try:
            image = Image.open(image_path).convert('RGB')
            
            with torch.no_grad():
                clip_inputs = self.model.clip_processor(images=image, return_tensors="pt").to(self.device)
                clip_features = self.model.clip_model.get_image_features(**clip_inputs)
                clip_features = F.normalize(clip_features, p=2, dim=1)
                
                original_features = clip_features.cpu().numpy().flatten()
                original_features = original_features / np.linalg.norm(original_features)
                
                mapped_features = None
                if self.use_trained_model:
                    outputs = self.model.forward(clip_features)
                    mapped_features = outputs['fashion_embedding'].cpu().numpy().flatten()
                    mapped_features = mapped_features / np.linalg.norm(mapped_features)
                
                pca_features = self.pca.transform([original_features])[0]
                pca_features = pca_features / np.linalg.norm(pca_features)
                
                return {
                    'original': original_features,
                    'mapped': mapped_features,
                    'pca': pca_features
                }
        except Exception as e:
            logging.error(f"❌ 特徵提取失敗: {e}")
            return None
    
    def extract_outfit_attributes(self, image_path):
        """提取圖片的服裝屬性"""
        try:
            image = Image.open(image_path).convert('RGB')
            
            # 使用 FashionCLIP 提取屬性
            text_prompts = [
                "red dress", "blue t-shirt", "black pants", "white shirt", "jeans",
                "casual style", "formal style", "bohemian style", "street style",
                "black shoes", "sneakers", "scarf", "hat", "long sleeves", "floral pattern"
            ]
            clip_inputs = self.model.clip_processor(
                text=text_prompts, images=image, return_tensors="pt", padding=True
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model.clip_model(**clip_inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)[0]
            
            clip_attrs = {prompt: float(prob) for prompt, prob in zip(text_prompts, probs)}
            
            # 使用 BLIP-2 生成描述（如果可用）
            description = ""
            if 'blip2' in self.loaded_ai_models:
                config = self.loaded_ai_models['blip2']
                inputs = config['processor'](
                    images=image,
                    text="Describe the outfit in this image, including clothing types, colors, and style.",
                    return_tensors="pt"
                ).to(self.device)
                with torch.no_grad():
                    outputs = config['model'].generate(
                        **inputs,
                        max_new_tokens=50,
                        num_beams=4
                    )
                description = config['processor'].decode(outputs[0], skip_special_tokens=True)
            
            # 解析描述
            parsed_attrs = {}
            if description:
                description_lower = description.lower()
                colors = ["red", "blue", "black", "white", "gray"]
                clothing_types = ["dress", "t-shirt", "pants", "shirt", "jeans"]
                styles = ["casual", "formal", "bohemian", "street"]
                
                for color in colors:
                    if color in description_lower:
                        parsed_attrs["color"] = color
                        break
                for clothing in clothing_types:
                    if clothing in description_lower:
                        parsed_attrs["clothing_type"] = clothing
                        break
                for style in styles:
                    if style in description_lower:
                        parsed_attrs["style"] = style
                        break
            
            return {
                "clip_attributes": clip_attrs,
                "description": description,
                "parsed_attributes": parsed_attrs
            }
        except Exception as e:
            logging.error(f"❌ 屬性提取失敗: {e}")
            return {"clip_attributes": {}, "description": "", "parsed_attributes": {}}
    
    def find_similar_outfits_improved(self, image_path, gender, top_k=4, style_preference=None, 
                                    similarity_weights={'original': 0.5, 'pca': 0.3, 'mapped': 0.2}):
        """改進版相似穿搭查找，固定返回4張推薦"""
        logging.info(f"🔍 改進版圖片分析: {image_path}")
        
        query_features = self.extract_image_features(image_path)
        if query_features is None:
            return []
        
        filtered_indices = []
        for i, sample in enumerate(self.dataset):
            if sample['gender'] != gender:
                continue
            if style_preference and sample['style'] != style_preference:
                continue
            filtered_indices.append(i)
        
        if not filtered_indices:
            logging.error("❌ 沒有符合條件的樣本")
            return []
        
        similarities = self._calculate_multi_similarity(
            query_features, filtered_indices, similarity_weights
        )
        
        sorted_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in sorted_indices:
            original_idx = filtered_indices[idx]
            sample = self.dataset[original_idx]
            
            detailed_sim = self._calculate_detailed_similarity(
                query_features, original_idx
            )
            
            results.append({
                'path': sample['path'],
                'style': sample['style'],
                'gender': sample['gender'],
                'similarity': float(similarities[idx]),
                'score': float(similarities[idx] * 10),
                'detailed_similarity': detailed_sim,
                'features': sample['features'],
                'clothing_types': sample.get('clothing_types', []),
                'colors': sample.get('colors', [])
            })
        
        logging.info(f"✅ 找到 {len(results)} 個相似穿搭")
        return results
    
    def _calculate_multi_similarity(self, query_features, filtered_indices, weights):
        """計算多級特徵融合相似度"""
        final_similarities = np.zeros(len(filtered_indices))
        
        if 'original' in weights and weights['original'] > 0:
            filtered_original = self.original_features[filtered_indices]
            original_sim = cosine_similarity([query_features['original']], filtered_original)[0]
            final_similarities += weights['original'] * original_sim
        
        if 'pca' in weights and weights['pca'] > 0:
            filtered_pca = self.pca_features[filtered_indices]
            pca_sim = cosine_similarity([query_features['pca']], filtered_pca)[0]
            final_similarities += weights['pca'] * pca_sim
        
        if 'mapped' in weights and weights['mapped'] > 0 and self.mapped_features is not None:
            filtered_mapped = self.mapped_features[filtered_indices]
            mapped_sim = cosine_similarity([query_features['mapped']], filtered_mapped)[0]
            final_similarities += weights['mapped'] * mapped_sim
        
        return final_similarities
    
    def _calculate_detailed_similarity(self, query_features, sample_idx):
        """計算詳細相似度分析"""
        sample_original = self.original_features[sample_idx]
        sample_pca = self.pca_features[sample_idx]
        
        detailed = {
            'visual_similarity': float(cosine_similarity([query_features['original']], [sample_original])[0][0]),
            'main_component_similarity': float(cosine_similarity([query_features['pca']], [sample_pca])[0][0])
        }
        
        if self.mapped_features is not None:
            sample_mapped = self.mapped_features[sample_idx]
            detailed['style_similarity'] = float(cosine_similarity([query_features['mapped']], [sample_mapped])[0][0])
        
        return detailed
    
    def translate_to_chinese(self, text):
        """將英文建議翻譯為中文"""
        translations = {
            "red": "紅色",
            "blue": "藍色",
            "black": "黑色",
            "white": "白色",
            "gray": "灰色",
            "dress": "連衣裙",
            "t-shirt": "T恤",
            "pants": "褲子",
            "jeans": "牛仔褲",
            "shirt": "襯衫",
            "casual": "休閒",
            "bohemian": "波西米亞",
            "formal": "正式",
            "street": "街頭",
            "shoes": "鞋子",
            "sneakers": "運動鞋",
            "scarf": "圍巾",
            "hat": "帽子",
            "accessories": "配飾",
            "color": "顏色",
            "style": "風格",
            "replace": "替換",
            "add": "添加",
            "remove": "移除",
            "adjust": "調整",
            "long sleeves": "長袖",
            "floral pattern": "花卉圖案"
        }
        
        for en, zh in translations.items():
            text = re.sub(r'\b' + en + r'\b', zh, text, flags=re.IGNORECASE)
        
        text = re.sub(r'(\d+)\.', r'\1：', text)
        return text
    
    def _clean_text_output(self, text):
        """清理AI輸出文字，保留結構化格式"""
        if not text or len(text.strip()) < 5:
            return "建議調整關鍵元素以匹配目標風格"
        
        text = re.sub(r'[.]{2,}', '.', text)
        text = re.sub(r'[ ]{2,}', ' ', text)
        text = re.sub(r'^.*?describe.*?:', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^.*?question.*?:', '', text, flags=re.IGNORECASE)
        
        text = text.strip()
        sentences = text.split('. ')
        formatted_text = []
        
        for sentence in sentences:
            if sentence.strip():
                sentence = re.sub(r'(\d+)\.', r'\1:', sentence)
                formatted_text.append(sentence.strip())
        
        return '. '.join(formatted_text) + '.' if formatted_text else "建議調整關鍵元素以匹配目標風格"
    
    def generate_comparison_advice(self, user_image_path, recommendation, ai_models):
        """為單一推薦生成多模態比較建議"""
        logging.info(f"🤖 為推薦 {recommendation['style']} 生成比較建議...")
        
        advice_results = {
            'target_style': recommendation['style'],
            'similarity': recommendation['similarity'],
            'ai_advice': {}
        }
        
        with ThreadPoolExecutor(max_workers=len(ai_models)) as executor:
            future_to_model = {
                executor.submit(self._generate_model_advice, model_key, user_image_path, recommendation): model_key
                for model_key in ai_models if model_key in self.loaded_ai_models or model_key in ['rule_based', 'clip']
            }
            
            for future in as_completed(future_to_model):
                model_key = future_to_model[future]
                try:
                    advice = future.result()
                    advice_results['ai_advice'][model_key] = advice
                except Exception as e:
                    advice_results['ai_advice'][model_key] = f"{self.model_configs[model_key]['name']} 分析失敗: {str(e)}"
        
        return advice_results
    
    def _generate_model_advice(self, model_key, user_image_path, recommendation):
        """根據模型類型生成建議"""
        if model_key == 'rule_based':
            return self._generate_rule_based_advice(user_image_path, recommendation)
        elif model_key == 'clip':
            return self._generate_clip_advice(user_image_path, recommendation)
        elif model_key == 'llava':
            return self._generate_llava_advice(user_image_path, recommendation)
        elif model_key == 'blip2':
            return self._generate_blip2_advice(user_image_path, recommendation)
        elif model_key == 'instructblip':
            return self._generate_instructblip_advice(user_image_path, recommendation)
        else:
            return "不支持的模型類型"
    
    def _generate_rule_based_advice(self, user_image_path, recommendation):
        """生成規則系統建議"""
        try:
            similarity = recommendation['similarity']
            style = recommendation['style']
            colors = recommendation.get('colors', [])
            clothing_types = recommendation.get('clothing_types', [])
            
            advice_parts = []
            
            # 添加相似度評估
            if similarity > 0.8:
                advice_parts.append(f"✨ 您的穿搭與{style}風格非常接近！")
                advice_parts.append("建議進行微調：")
            elif similarity > 0.6:
                advice_parts.append(f"👍 您的穿搭與{style}風格有一定相似性")
                advice_parts.append("建議進行以下調整：")
            else:
                advice_parts.append(f"🎯 要達到{style}風格，需要較大幅度調整：")
            
            # 風格特定建議
            style_advice = {
                'CASUAL': [
                    "• 選擇舒適自然的剪裁",
                    "• 顏色以基本色為主：白、黑、藍、灰",
                    "• 材質選擇棉質、牛仔等",
                    "• 避免過於正式或華麗的元素"
                ],
                'BOHEMIAN': [
                    "• 選擇輕盈、流動的材質",
                    "• 融入花卉圖案或民族元素", 
                    "• 配飾：長項鍊、寬邊帽",
                    "• 顏色以大地色或鮮豔色為主"
                ],
                'STREET': [
                    "• 加入街頭元素：寬鬆剪裁、層次搭配",
                    "• 嘗試oversized上衣或帽T",
                    "• 鞋子選擇運動鞋或街頭風靴子",
                    "• 配飾：棒球帽、背包"
                ],
                'FORMAL': [
                    "• 選擇修身剪裁的西裝或禮服",
                    "• 顏色以黑、白、深藍為主",
                    "• 配飾：領帶、袖扣等正式元素",
                    "• 避免休閒元素"
                ],
                'ARTSY': [
                    "• 嘗試不對稱或特殊剪裁設計",
                    "• 融入藝術感的圖案或色彩",
                    "• 配飾選擇有設計感的單品",
                    "• 敢於嘗試獨特的搭配組合"
                ]
            }
            
            if style in style_advice:
                advice_parts.extend(style_advice[style])
            else:
                # 通用建議
                advice_parts.extend([
                    "• 注意整體搭配的協調性",
                    "• 選擇適合的顏色組合",
                    "• 考慮場合的適切性"
                ])
            
            # 添加具體的顏色和服裝建議
            if colors:
                advice_parts.append(f"🎨 參考色彩：{', '.join(colors[:3])}")
            
            if clothing_types:
                advice_parts.append(f"👔 推薦服裝類型：{', '.join(clothing_types[:3])}")
            
            # 添加相似度評分
            score = int(similarity * 100)
            advice_parts.append(f"📊 目前匹配度：{score}%")
            
            return "\n".join(advice_parts)
            
        except Exception as e:
            logging.error(f"規則系統建議生成失敗: {e}")
            return f"規則系統分析失敗，請稍後再試"
    
    def _generate_clip_advice(self, user_image_path, recommendation):
        """生成改進版 FashionCLIP 基於特徵的 rule_based 比較建議"""
        try:
            logging.info("🔍 開始詳細的FashionCLIP特徵比較...")
            
            # 提取用戶和目標圖片的詳細特徵
            user_features = self.extract_detailed_fashion_features(user_image_path)
            target_features = self.extract_detailed_fashion_features(recommendation['path'])
            
            if not user_features or not target_features:
                return "FashionCLIP 特徵提取失敗"
            
            style = recommendation['style']
            similarity = recommendation['similarity']
            
            advice_parts = []
            advice_parts.append(f"🎯 FashionCLIP 詳細特徵比較分析 (目標風格: {style})")
            
            # 分析每個特徵類別的差異
            for category in self.fashion_features.keys():
                if category in user_features and category in target_features:
                    user_feature = user_features[category]
                    target_feature = target_features[category]
                    
                    user_top = user_feature['top_feature']
                    target_top = target_feature['top_feature']
                    user_score = user_feature['score']
                    target_score = target_feature['score']
                    
                    category_advice = self._analyze_feature_category(
                        category, user_top, target_top, user_score, target_score, user_feature, target_feature
                    )
                    
                    if category_advice:
                        advice_parts.append(category_advice)
            
            # 添加整體建議
            overall_advice = self._generate_overall_clip_advice(user_features, target_features, similarity, style)
            advice_parts.append(overall_advice)
            
            return "\n".join(advice_parts)
            
        except Exception as e:
            return f"FashionCLIP 詳細分析失敗: {str(e)}"

    def _analyze_feature_category(self, category, user_top, target_top, user_score, target_score, user_feature, target_feature):
        """分析特定特徵類別的差異並生成建議"""
        category_names = {
            'colors': '顏色',
            'clothing_types': '服裝類型', 
            'styles': '風格',
            'patterns': '圖案',
            'accessories': '配飾',
            'shoes': '鞋類',
            'materials': '材質',
            'fit': '版型'
        }
        
        category_name = category_names.get(category, category)
        
        # 如果特徵相同，給予正面反饋
        if user_top == target_top:
            if user_score > 0.6 and target_score > 0.6:
                return f"✅ {category_name}: 很好！您的{user_top}與目標完全匹配 (相似度: {user_score:.2f})"
            else:
                return f"⚡ {category_name}: 方向正確但可以更突出{user_top}特徵"
        
        # 如果特徵不同，提供具體建議
        advice = f"🔄 {category_name}: "
        
        if category == 'colors':
            advice += f"您目前的{user_top}可以考慮調整為{target_top}來更符合目標風格"
        elif category == 'clothing_types':
            advice += f"建議將{user_top}替換為{target_top}以達到更好的效果"
        elif category == 'styles':
            advice += f"從{user_top}轉向{target_top}，注重相應的剪裁和搭配"
        elif category == 'patterns':
            advice += f"圖案從{user_top}改為{target_top}會更加合適"
        elif category == 'accessories':
            advice += f"配飾方面建議加入{target_top}元素"
        elif category == 'shoes':
            advice += f"鞋類建議從{user_top}換成{target_top}"
        elif category == 'materials':
            advice += f"材質可以選擇{target_top}而非{user_top}"
        elif category == 'fit':
            advice += f"版型建議調整為{target_top}而非{user_top}"
        
        # 添加信心度信息
        if target_score > 0.7:
            advice += f" (目標特徵明顯: {target_score:.2f})"
        elif target_score > 0.5:
            advice += f" (目標特徵中等: {target_score:.2f})"
        else:
            advice += f" (目標特徵較弱: {target_score:.2f})"
            
        return advice

    def _generate_overall_clip_advice(self, user_features, target_features, similarity, style):
        """生成整體的FashionCLIP建議"""
        advice = f"\n📋 整體建議 (相似度: {similarity:.3f}):\n"
        
        # 計算各類別的匹配度
        matches = 0
        total_categories = 0
        priority_suggestions = []
        
        for category in self.fashion_features.keys():
            if category in user_features and category in target_features:
                total_categories += 1
                if user_features[category]['top_feature'] == target_features[category]['top_feature']:
                    matches += 1
                else:
                    # 根據目標特徵的信心度決定優先級
                    target_score = target_features[category]['score']
                    if target_score > 0.6:
                        priority_suggestions.append((category, target_score))
        
        match_rate = matches / total_categories if total_categories > 0 else 0
        
        if match_rate > 0.7:
            advice += "🎉 您的穿搭已經非常接近目標風格！只需要細微調整"
        elif match_rate > 0.5:
            advice += "👍 基本方向正確，重點改進以下幾個方面"
        else:
            advice += "🎯 需要較大幅度的調整來達到目標風格"
        
        # 按優先級排序建議
        priority_suggestions.sort(key=lambda x: x[1], reverse=True)
        
        if priority_suggestions:
            advice += "\n🔥 優先改進項目："
            for i, (category, score) in enumerate(priority_suggestions[:3], 1):
                category_names = {
                    'colors': '顏色搭配',
                    'clothing_types': '服裝選擇', 
                    'styles': '整體風格',
                    'patterns': '圖案設計',
                    'accessories': '配飾選擇',
                    'shoes': '鞋類搭配',
                    'materials': '材質選擇',
                    'fit': '版型調整'
                }
                category_name = category_names.get(category, category)
                advice += f"\n  {i}. {category_name} (重要度: {score:.2f})"
        
        return advice

    def extract_detailed_fashion_features(self, image_path):
        """提取詳細的時尚特徵用於rule_based比較"""
        try:
            image = Image.open(image_path).convert('RGB')
            
            # 調整圖片大小以提高處理效率
            if max(image.size) > 512:
                ratio = 512 / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            detailed_features = {}
            
            # 為每個特徵類別提取評分
            for category, prompts in self.fashion_features.items():
                clip_inputs = self.model.clip_processor(
                    text=prompts, images=image, return_tensors="pt", padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.clip_model(**clip_inputs)
                    logits_per_image = outputs.logits_per_image
                    probs = logits_per_image.softmax(dim=1)[0]
                
                # 獲取最高分的特徵和其分數
                max_idx = probs.argmax().item()
                max_score = probs[max_idx].item()
                
                detailed_features[category] = {
                    'top_feature': prompts[max_idx],
                    'score': float(max_score),
                    'all_scores': {prompt: float(score) for prompt, score in zip(prompts, probs)}
                }
            
            return detailed_features
        except Exception as e:
            logging.error(f"❌ 詳細特徵提取失敗: {e}")
            return {}

    def create_comparison_image(self, user_image_path, target_image_path, temp_path="temp_comparison.jpg"):
        """創建上下排列的比較圖片供LLaVA分析"""
        try:
            # 載入兩張圖片
            user_image = Image.open(user_image_path).convert('RGB')
            target_image = Image.open(target_image_path).convert('RGB')
            
            # 標準化圖片大小
            target_width = 512
            target_height = 512
            
            user_image = user_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
            target_image = target_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
            
            # 創建新的圖片，高度是兩張圖片的總和加上標籤空間
            label_height = 60
            total_height = (target_height * 2) + (label_height * 2) + 20  # 20是間距
            comparison_image = Image.new('RGB', (target_width, total_height), (255, 255, 255))
            
            # 嘗試載入字體
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
            except:
                try:
                    font = ImageFont.truetype("arial.ttf", 24)
                except:
                    font = ImageFont.load_default()
            
            # 創建繪圖對象
            draw = ImageDraw.Draw(comparison_image)
            
            # 添加標籤和圖片
            y_offset = 10
            
            # 用戶圖片標籤
            draw.text((10, y_offset), "用戶穿搭 (User Outfit)", fill=(0, 0, 0), font=font)
            y_offset += label_height
            
            # 粘貼用戶圖片
            comparison_image.paste(user_image, (0, y_offset))
            y_offset += target_height + 10
            
            # 目標圖片標籤
            draw.text((10, y_offset), "目標穿搭 (Target Outfit)", fill=(0, 0, 0), font=font)
            y_offset += label_height
            
            # 粘貼目標圖片
            comparison_image.paste(target_image, (0, y_offset))
            
            # 保存比較圖片
            comparison_image.save(temp_path, quality=95)
            logging.info(f"✅ 比較圖片已創建: {temp_path}")
            
            return temp_path
        except Exception as e:
            logging.error(f"❌ 創建比較圖片失敗: {e}")
            return None

    def _generate_llava_advice(self, user_image_path, recommendation):
        """生成改進版 LLaVA 雙圖片比較建議"""
        try:
            if 'llava' not in self.loaded_ai_models:
                return "LLaVA模型未載入"
                
            config = self.loaded_ai_models['llava']
            
            # 創建比較圖片
            comparison_path = self.create_comparison_image(user_image_path, recommendation['path'])
            if not comparison_path:
                return "無法創建比較圖片"
            
            try:
                # 載入合併後的圖片
                comparison_image = Image.open(comparison_path).convert('RGB')
                
                # 創建專門的比較提示詞
                style = recommendation['style']
                similarity = recommendation['similarity']
                
                prompt = f"""請仔細分析這張比較圖片，上半部分是用戶的當前穿搭，下半部分是目標{style}風格的穿搭。

請從以下幾個方面進行詳細比較並提供具體的改進建議：

1. 服裝類型比較：分析兩者在上衣、下裝、外套等方面的差異
2. 顏色搭配比較：比較色彩選擇和搭配方式
3. 風格元素比較：分析風格特徵的差異
4. 配飾比較：比較配飾的選擇和搭配
5. 整體效果比較：評價整體協調性和風格統一性

請提供5-8條具體的改進建議，讓用戶的穿搭更接近目標風格。"""

                # 使用正確的LLaVA格式
                conversation = [
                    {
                        "role": "user", 
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
                
                # 格式化prompt
                formatted_prompt = config['processor'].apply_chat_template(
                    conversation, 
                    add_generation_prompt=True
                )
                
                # 處理輸入
                inputs = config['processor'](
                    comparison_image, 
                    formatted_prompt, 
                    return_tensors="pt"
                ).to(self.device)
                
                # 生成回應
                with torch.no_grad():
                    outputs = config['model'].generate(
                        **inputs,
                        max_new_tokens=300,
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.9,
                        pad_token_id=config['processor'].tokenizer.eos_token_id
                    )
                
                # 解碼回應
                response = config['processor'].decode(outputs[0], skip_special_tokens=True)
                
                # 清理回應
                if formatted_prompt in response:
                    response = response.replace(formatted_prompt, "").strip()
                
                # 清理臨時文件
                try:
                    os.remove(comparison_path)
                except:
                    pass
                
                # 格式化回應
                cleaned_response = self._clean_and_format_llava_response(response, style, similarity)
                return self.translate_to_chinese(cleaned_response)
                
            except Exception as e:
                # 清理臨時文件
                try:
                    os.remove(comparison_path)
                except:
                    pass
                raise e
                
        except Exception as e:
            return f"LLaVA 雙圖片比較分析失敗: {str(e)}"

    def _clean_and_format_llava_response(self, response, style, similarity):
        """清理和格式化LLaVA的回應"""
        if not response or len(response.strip()) < 10:
            return f"建議參考{style}風格的特徵進行整體調整"
        
        # 移除多餘的空白和標點
        response = re.sub(r'\s+', ' ', response.strip())
        response = re.sub(r'[.]{2,}', '.', response)
        
        # 確保回應是結構化的
        if not any(marker in response for marker in ['1.', '2.', '1:', '2:', '1、', '2、']):
            # 如果沒有編號，嘗試按句子分割並添加編號
            sentences = [s.strip() for s in response.split('.') if s.strip() and len(s.strip()) > 10]
            if sentences:
                formatted_response = ""
                for i, sentence in enumerate(sentences[:6], 1):
                    formatted_response += f"{i}. {sentence}. "
                response = formatted_response
        
        # 添加相似度信息
        similarity_text = f"\n\n📊 當前相似度: {similarity:.1%}"
        if similarity > 0.8:
            similarity_text += " - 已經非常接近目標風格"
        elif similarity > 0.6:
            similarity_text += " - 有一定相似性，可以進一步優化"
        else:
            similarity_text += " - 需要較大調整來達到目標風格"
        
        return response + similarity_text

    def _generate_blip2_advice(self, user_image_path, recommendation):
        """生成 BLIP-2 建議"""
        try:
            config = self.loaded_ai_models['blip2']
            
            user_image = Image.open(user_image_path).convert('RGB')
            target_image = Image.open(recommendation['path']).convert('RGB')
            
            user_prompt = "Describe the outfit in this image, including clothing types, colors, and style."
            target_prompt = f"Describe the outfit in this image, focusing on its {recommendation['style']} style characteristics."
            
            # 處理用戶圖片描述
            user_inputs = config['processor'](
                images=user_image,
                text=user_prompt,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            with torch.no_grad():
                user_outputs = config['model'].generate(
                    **user_inputs,
                    max_new_tokens=100,
                    num_beams=4
                )
            user_desc = config['processor'].decode(user_outputs[0], skip_special_tokens=True)
            
            # 處理目標圖片描述
            target_inputs = config['processor'](
                images=target_image,
                text=target_prompt,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            with torch.no_grad():
                target_outputs = config['model'].generate(
                    **target_inputs,
                    max_new_tokens=100,
                    num_beams=4
                )
            target_desc = config['processor'].decode(target_outputs[0], skip_special_tokens=True)
            
            style = recommendation['style']
            similarity = recommendation['similarity']
            
            user_attrs = self.extract_outfit_attributes(user_image_path)
            target_attrs = self.extract_outfit_attributes(recommendation['path'])
            
            prompt = f"""
Compare the following two outfit descriptions to provide specific improvement advice:
User's outfit: {user_desc}
Target {style} style outfit: {target_desc}
User attributes: {user_attrs['parsed_attributes']}
Target attributes: {target_attrs['parsed_attributes']}
Similarity score: {similarity:.3f}

Provide actionable advice to transform the user's outfit to match the target style, structured as:
1. Key differences: List specific differences in clothing items.
2. Clothing suggestions: Specify clothing items to change.
3. Color adjustments: Suggest changes in colors or patterns.
4. Accessory suggestions: Recommend accessories to add or remove.
5. Overall styling: Advise on improving the overall look.

Provide detailed and practical advice in English.
"""
            inputs = config['processor'](
                text=prompt,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = config['model'].generate(
                    **inputs,
                    max_new_tokens=200,
                    num_beams=4,
                    temperature=0.8
                )
            
            advice = config['processor'].decode(outputs[0], skip_special_tokens=True)
            advice = advice.replace(prompt, "").strip()
            return self.translate_to_chinese(self._clean_text_output(advice))
        except Exception as e:
            return f"圖像描述模型 (BLIP-2) 分析失敗: {str(e)}"
    
    def _generate_instructblip_advice(self, user_image_path, recommendation):
        """生成 InstructBLIP 建議"""
        try:
            config = self.loaded_ai_models['instructblip']
            
            user_image = Image.open(user_image_path).convert('RGB')
            target_image = Image.open(recommendation['path']).convert('RGB')
            
            user_prompt = "Describe the outfit in this image, including clothing types, colors, and style."
            target_prompt = f"Describe the outfit in this image, focusing on its {recommendation['style']} style characteristics."
            
            # 處理用戶圖片描述
            user_inputs = config['processor'](
                images=user_image,
                text=user_prompt,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            with torch.no_grad():
                user_outputs = config['model'].generate(
                    **user_inputs,
                    max_new_tokens=100,
                    num_beams=4
                )
            user_desc = config['processor'].decode(user_outputs[0], skip_special_tokens=True)
            
            # 處理目標圖片描述
            target_inputs = config['processor'](
                images=target_image,
                text=target_prompt,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            with torch.no_grad():
                target_outputs = config['model'].generate(
                    **target_inputs,
                    max_new_tokens=100,
                    num_beams=4
                )
            target_desc = config['processor'].decode(target_outputs[0], skip_special_tokens=True)
            
            style = recommendation['style']
            similarity = recommendation['similarity']
            
            user_attrs = self.extract_outfit_attributes(user_image_path)
            target_attrs = self.extract_outfit_attributes(recommendation['path'])
            
            prompt = f"""
Compare the following two outfit descriptions to provide specific improvement advice:
User's outfit: {user_desc}
Target {style} style outfit: {target_desc}
User attributes: {user_attrs['parsed_attributes']}
Target attributes: {target_attrs['parsed_attributes']}
Similarity score: {similarity:.3f}

Provide actionable advice to transform the user's outfit to match the target style, structured as:
1. Key differences: List specific differences in clothing items.
2. Clothing suggestions: Specify clothing items to change.
3. Color adjustments: Suggest changes in colors or patterns.
4. Accessory suggestions: Recommend accessories to add or remove.
5. Overall styling: Advise on improving the overall look.

Provide detailed and practical advice in English.
"""
            inputs = config['processor'](
                text=prompt,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = config['model'].generate(
                    **inputs,
                    max_new_tokens=200,
                    num_beams=4,
                    temperature=0.8
                )
            
            advice = config['processor'].decode(outputs[0], skip_special_tokens=True)
            advice = advice.replace(prompt, "").strip()
            return self.translate_to_chinese(self._clean_text_output(advice))
        except Exception as e:
            return f"指令圖像模型 (InstructBLIP) 分析失敗: {str(e)}"

    def analyze_and_recommend_improved(self, image_path, gender, top_k=4, style_preference=None, 
                                     strategy='balanced', ai_models=['rule_based', 'clip', 'llava', 'blip2', 'instructblip']):
        """改進版分析和推薦流程，包含多模態比較"""
        logging.info(f"\n🎯 開始改進版穿搭分析...")
        logging.info(f"📁 圖片: {image_path}")
        logging.info(f"👤 性別: {gender}")
        logging.info(f"🎨 風格偏好: {style_preference or '無限制'}")
        logging.info(f"🔧 策略: {strategy}")
        logging.info(f"🤖 使用多模態模型: {', '.join(ai_models)}")
        
        if not os.path.exists(image_path):
            return {"error": f"圖片文件不存在: {image_path}"}
        
        try:
            # 載入多模態模型
            self._load_ai_models(ai_models)
            
            # 定義策略權重
            strategy_weights = {
                'pure_visual': {'original': 1.0, 'pca': 0.0, 'mapped': 0.0},
                'visual_focused': {'original': 0.7, 'pca': 0.3, 'mapped': 0.0},
                'balanced': {'original': 0.5, 'pca': 0.3, 'mapped': 0.2},
                'style_aware': {'original': 0.3, 'pca': 0.2, 'mapped': 0.5}
            }
            
            weights = strategy_weights.get(strategy, strategy_weights['balanced'])
            
            # 生成推薦
            similar_outfits = self.find_similar_outfits_improved(
                image_path, gender, top_k, style_preference, weights
            )
            
            if not similar_outfits:
                return {
                    "error": "沒有找到相似的穿搭",
                    "suggestion": "請嘗試調整性別或風格偏好設置"
                }
            
            # 生成多模態比較建議
            for outfit in similar_outfits:
                outfit['comparison_advice'] = self.generate_comparison_advice(
                    image_path, outfit, ai_models
                )
            
            # 風格分析
            style_analysis = self._generate_style_analysis(similar_outfits)
            
            # 改進建議
            improved_advice = self._generate_improved_advice(similar_outfits, strategy)
            
            result = {
                "status": "success",
                "input_image": image_path,
                "gender": gender,
                "style_preference": style_preference,
                "strategy": strategy,
                "strategy_weights": weights,
                "similar_outfits": similar_outfits,
                "style_analysis": style_analysis,
                "improved_advice": improved_advice,
                "ai_models_used": ai_models,
                "summary": {
                    "best_match": similar_outfits[0] if similar_outfits else None,
                    "recommended_style": style_analysis.get("dominant_style"),
                    "visual_similarity": similar_outfits[0]['detailed_similarity']['visual_similarity'] if similar_outfits else 0,
                    "style_similarity": similar_outfits[0]['detailed_similarity'].get('style_similarity', 0) if similar_outfits else 0,
                    "total_recommendations": len(similar_outfits)
                }
            }
            
            logging.info(f"✅ 改進版分析完成！")
            return result
        except Exception as e:
            logging.error(f"❌ 分析失敗: {e}")
            return {"error": f"分析失敗: {str(e)}"}
    
    def _generate_style_analysis(self, similar_outfits):
        """生成風格分析"""
        if not similar_outfits:
            return {}
        
        style_distribution = {}
        visual_similarities = []
        style_similarities = []
        
        for outfit in similar_outfits:
            style = outfit['style']
            style_distribution[style] = style_distribution.get(style, 0) + 1
            visual_similarities.append(outfit['detailed_similarity']['visual_similarity'])
            if 'style_similarity' in outfit['detailed_similarity']:
                style_similarities.append(outfit['detailed_similarity']['style_similarity'])
        
        dominant_style = max(style_distribution.items(), key=lambda x: x[1])[0]
        
        return {
            "dominant_style": dominant_style,
            "style_distribution": style_distribution,
            "average_visual_similarity": float(np.mean(visual_similarities)),
            "average_style_similarity": float(np.mean(style_similarities)) if style_similarities else 0,
            "max_visual_similarity": float(max(visual_similarities))
        }
    
    def _generate_improved_advice(self, similar_outfits, strategy):
        """生成改進建議"""
        if not similar_outfits:
            return []
        
        advice = []
        strategy_advice = {
            'pure_visual': "基於純視覺相似性的推薦，注重外觀匹配度",
            'visual_focused': "以視覺相似為主導，兼顧主要視覺成分",
            'balanced': "平衡視覺相似性和風格一致性的推薦",
            'style_aware': "強調風格一致性，同時保持視覺協調"
        }
        
        if strategy in strategy_advice:
            advice.append(f"策略特點: {strategy_advice[strategy]}")
        
        avg_visual = np.mean([r['detailed_similarity']['visual_similarity'] for r in similar_outfits])
        
        if avg_visual > 0.8:
            advice.append("視覺匹配度極佳，可直接參考推薦穿搭")
        elif avg_visual > 0.7:
            advice.append("視覺匹配度良好，建議學習推薦穿搭的搭配技巧")
        else:
            advice.append("推薦穿搭可作為風格轉換的靈感來源")
        
        return advice
    
    def generate_improved_display_image(self, results, output_path="improved_recommendation_display.png", max_recommendations=4):
        """生成包含推薦圖片和多模態建議的展示圖片"""
        if "error" in results:
            logging.error(f"❌ 無法生成顯示圖片: {results['error']}")
            return None
        
        logging.info(f"🎨 生成改進版顯示圖片...")
        
        try:
            recommendations = results['similar_outfits'][:max_recommendations]
            num_recs = len(recommendations)
            total_images = num_recs + 1  # +1 for original image
            ai_models = results['ai_models_used']
            
            # 計算子圖高度：圖片 + 每個模型的建議
            fig_height = 6 + 2 * len(ai_models)  # 圖片6英寸，每個模型建議2英寸
            fig = plt.figure(figsize=(total_images * 4, fig_height))
            
            # 圖片顯示區域
            for i in range(total_images):
                ax = fig.add_subplot(len(ai_models) + 1, total_images, i + 1)
                
                if i == 0:
                    title = "原圖"
                    image_path = results['input_image']
                else:
                    rec = recommendations[i - 1]
                    title = f"推薦 {i}\n{rec['style']}\n相似度: {rec['similarity']:.3f}\n評分: {rec['score']:.1f}/10"
                    image_path = rec['path']
                
                self._show_image_subplot(ax, image_path, title)
            
            # 每個模型的建議區域
            for model_idx, model_key in enumerate(ai_models):
                for rec_idx in range(num_recs):
                    ax = fig.add_subplot(len(ai_models) + 1, total_images, (model_idx + 1) * total_images + rec_idx + 2)
                    advice = recommendations[rec_idx]['comparison_advice']['ai_advice'].get(model_key, "無建議")
                    model_name = self.model_configs[model_key]['name']
                    wrapped_text = textwrap.fill(f"{model_name}：\n{advice}", width=50)
                    ax.text(0.05, 0.95, wrapped_text, fontsize=8, va='top')
                    ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logging.info(f"✅ 顯示圖片已生成: {output_path}")
            return output_path
        except Exception as e:
            logging.error(f"❌ 生成顯示圖片失敗: {e}")
            return None
    
    def _show_image_subplot(self, ax, image_path, title):
        """顯示單張圖片"""
        try:
            if not os.path.exists(image_path):
                ax.text(0.5, 0.5, f"圖片不存在\n{os.path.basename(image_path)}", 
                       ha='center', va='center', bbox=dict(facecolor='lightcoral', alpha=0.7))
            else:
                img = plt.imread(image_path)
                ax.imshow(img)
            
            ax.set_title(title, fontsize=10, fontweight='bold', pad=10)
            ax.axis('off')
        except Exception as e:
            ax.text(0.5, 0.5, f"載入失敗\n{os.path.basename(image_path)}", 
                   ha='center', va='center', bbox=dict(facecolor='lightcoral', alpha=0.7))
            ax.set_title(title, fontsize=10, fontweight='bold', pad=10)
            ax.axis('off')
    
    def print_improved_recommendations(self, results):
        """美化打印推薦結果"""
        if "error" in results:
            print(f"❌ 錯誤: {results['error']}")
            return
        
        print(f"\n{'='*80}")
        print(f"🎯 改進版AI穿搭推薦結果")
        print(f"{'='*80}")
        
        print(f"📸 分析圖片: {results['input_image']}")
        print(f"👤 性別: {results['gender']}")
        print(f"🎨 風格偏好: {results['style_preference'] or '無限制'}")
        print(f"🔧 使用策略: {results['strategy']}")
        print(f"🤖 多模態模型: {', '.join([self.model_configs[m]['name'] for m in results['ai_models_used']])}")
        
        weights = results['strategy_weights']
        print(f"⚖️ 特徵權重: 原始特徵{weights['original']:.1f} + PCA特徵{weights['pca']:.1f} + 映射特徵{weights['mapped']:.1f}")
        
        best_match = results['summary']['best_match']
        print(f"\n🥇 最佳匹配:")
        print(f"  風格: {best_match['style']}")
        print(f"  綜合評分: {best_match['score']:.1f}/10")
        print(f"  視覺相似度: {best_match['detailed_similarity']['visual_similarity']:.3f}")
        
        analysis = results['style_analysis']
        print(f"\n📊 整體分析:")
        print(f"  主導風格: {analysis['dominant_style']}")
        print(f"  平均視覺相似度: {analysis['average_visual_similarity']:.3f}")
        
        print(f"\n🔝 詳細推薦列表:")
        for i, rec in enumerate(results['similar_outfits'], 1):
            print(f"\n  {i}. {rec['style']} - 綜合評分: {rec['score']:.1f}/10")
            print(f"     視覺相似度: {rec['detailed_similarity']['visual_similarity']:.3f}")
            print(f"     圖片檔名: {os.path.basename(rec['path'])}")
            print(f"     多模態比較建議:")
            for model_key, advice in rec['comparison_advice']['ai_advice'].items():
                model_name = self.model_configs[model_key]['name']
                print(f"       - {model_name}: {advice}")

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='改進版AI穿搭推薦系統')
    parser.add_argument('--image', type=str, required=True, help='用戶圖片路徑')
    parser.add_argument('--gender', type=str, required=True, choices=['MEN', 'WOMEN'], help='性別')
    parser.add_argument('--top_k', type=int, default=4, help='推薦數量')
    parser.add_argument('--style', type=str, help='風格偏好')
    parser.add_argument('--model', type=str, default='simple_fashion_model_final.pth', help='模型路徑')
    parser.add_argument('--labels', type=str, default='simple_dataset_labels.json', help='標籤文件路徑')
    parser.add_argument('--strategy', type=str, default='balanced', 
                       choices=['pure_visual', 'visual_focused', 'balanced', 'style_aware'], help='推薦策略')
    parser.add_argument('--ai_models', type=str, nargs='+', 
                       choices=['rule_based', 'clip', 'llava', 'blip2', 'instructblip'],
                       default=['rule_based', 'clip', 'llava', 'blip2', 'instructblip'], help='多模態模型')
    parser.add_argument('--display', type=str, help='生成比較圖片路徑')
    parser.add_argument('--save', type=str, help='保存結果到JSON文件')
    
    args = parser.parse_args()
    
    try:
        system = ImprovedFashionRecommendationSystem(args.model, args.labels)
        
        results = system.analyze_and_recommend_improved(
            args.image, args.gender, args.top_k, args.style, args.strategy, args.ai_models
        )
        
        system.print_improved_recommendations(results)
        
        if args.display:
            display_path = system.generate_improved_display_image(results, args.display)
            if display_path:
                print(f"🖼️ 比較圖片已生成: {display_path}")
        
        if args.save:
            with open(args.save, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"💾 結果已保存到: {args.save}")
            
    except Exception as e:
        print(f"❌ 系統錯誤: {e}")

if __name__ == "__main__":
    main()