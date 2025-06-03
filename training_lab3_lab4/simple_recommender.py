#!/usr/bin/env python3
"""
AI驅動的穿搭推薦器
基於多種視覺語言模型進行深度分析和建議生成
完全移除規則生成，採用純AI驅動方式
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import json
import os
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    CLIPProcessor, CLIPModel,
    Blip2Processor, Blip2ForConditionalGeneration,
    AutoProcessor, LlavaNextForConditionalGeneration,
    AutoTokenizer, AutoModelForCausalLM
)
import sys
sys.path.append('../training_lab2')
from simple_train_model import SimpleFashionRecommender, SimpleTrainingConfig
import warnings
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

class MultiModelAIRecommendationSystem:
    def __init__(self, model_path="simple_fashion_model_final_best.pth", 
                 labels_file="simple_dataset_labels.json"):
        """
        初始化多模型AI推薦系統
        
        Args:
            model_path: 簡化版模型路徑
            labels_file: 簡化版標籤文件
        """
        print("🚀 初始化AI驅動推薦系統...")
        
        # 設置設備
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"🎯 使用設備: {self.device}")
        
        # 載入配置
        self.config = SimpleTrainingConfig()
        
        # 載入模型
        print("🔧 載入簡化版模型...")
        self.model = SimpleFashionRecommender(self.config).to(self.device)
        
        # 檢查是否有訓練好的模型
        self.use_trained_model = False
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.use_trained_model = True
                print(f"✅ 訓練模型載入成功: {model_path}")
            except Exception as e:
                print(f"⚠️ 載入訓練模型失敗: {e}")
                print("將使用預訓練的 FashionCLIP 特徵")
        else:
            print(f"⚠️ 模型文件不存在: {model_path}")
            print("將使用預訓練的 FashionCLIP 特徵")
        
        self.model.eval()
        
        # 載入數據集標籤
        print("📊 載入數據集...")
        if os.path.exists(labels_file):
            with open(labels_file, 'r') as f:
                self.dataset = json.load(f)
            print(f"✅ 載入 {len(self.dataset)} 個樣本")
        else:
            print(f"❌ 標籤文件不存在: {labels_file}")
            self.dataset = []
        
        # 預處理數據集特徵
        if self.dataset:
            self._preprocess_dataset_features()
        
        # 初始化AI模型管理器
        self.ai_models = {}
        self._init_ai_models()
        
        print("✅ AI驅動推薦系統初始化完成！")
    
    def _preprocess_dataset_features(self):
        """預處理數據集特徵矩陣"""
        print("🔄 預處理數據集特徵...")
        
        # 提取所有特徵向量
        features_list = []
        for sample in self.dataset:
            features_list.append(sample['features'])
        
        # 轉換為numpy矩陣並標準化
        self.dataset_features = np.array(features_list)
        self.dataset_features = self.dataset_features / np.linalg.norm(
            self.dataset_features, axis=1, keepdims=True
        )
        
        print(f"📊 數據集特徵矩陣: {self.dataset_features.shape}")
        
        # 如果使用訓練好的模型，需要將數據集特徵也通過映射層
        if self.use_trained_model:
            print("🔄 將數據集特徵通過訓練好的映射層...")
            with torch.no_grad():
                dataset_features_tensor = torch.tensor(self.dataset_features, dtype=torch.float32).to(self.device)
                
                # 分批處理以避免記憶體問題
                batch_size = 100
                mapped_features = []
                
                for i in range(0, len(dataset_features_tensor), batch_size):
                    batch = dataset_features_tensor[i:i+batch_size]
                    outputs = self.model.forward(batch)
                    mapped_batch = outputs['fashion_embedding'].cpu().numpy()
                    mapped_features.append(mapped_batch)
                
                # 合併所有批次
                self.dataset_features = np.vstack(mapped_features)
                # 重新標準化
                self.dataset_features = self.dataset_features / np.linalg.norm(
                    self.dataset_features, axis=1, keepdims=True
                )
                
                print(f"📊 映射後數據集特徵矩陣: {self.dataset_features.shape}")
    
    def _init_ai_models(self):
        """初始化多個AI模型（延遲載入）"""
        print("🧠 準備AI模型管理器...")
        
        # 模型配置
        self.model_configs = {
            'blip2': {
                'name': 'BLIP-2',
                'model_id': 'Salesforce/blip2-opt-2.7b',
                'loaded': False,
                'processor': None,
                'model': None
            },
            'llava': {
                'name': 'LLaVA-Next',
                'model_id': 'llava-hf/llava-v1.6-mistral-7b-hf',
                'loaded': False,
                'processor': None,
                'model': None
            },
            'instructblip': {
                'name': 'InstructBLIP',
                'model_id': 'Salesforce/instructblip-vicuna-7b',
                'loaded': False,
                'processor': None,
                'model': None
            }
        }
        
        print("✅ AI模型管理器準備完成")
    
    def _load_model(self, model_key):
        """動態載入指定模型"""
        if model_key not in self.model_configs:
            print(f"❌ 未知模型: {model_key}")
            return False
        
        config = self.model_configs[model_key]
        if config['loaded']:
            return True
        
        try:
            print(f"🔄 載入 {config['name']} 模型...")
            
            if model_key == 'blip2':
                config['processor'] = Blip2Processor.from_pretrained(config['model_id'])
                config['model'] = Blip2ForConditionalGeneration.from_pretrained(
                    config['model_id'],
                    torch_dtype=torch.float16 if self.device.type == 'mps' else torch.float32,
                    device_map="auto" if self.device.type != 'mps' else None
                )
                if self.device.type == 'mps':
                    config['model'] = config['model'].to(self.device)
            
            elif model_key == 'llava':
                config['processor'] = AutoProcessor.from_pretrained(config['model_id'])
                config['model'] = LlavaNextForConditionalGeneration.from_pretrained(
                    config['model_id'],
                    torch_dtype=torch.float16 if self.device.type == 'mps' else torch.float32,
                    low_cpu_mem_usage=True,
                    device_map="auto" if self.device.type != 'mps' else None
                )
                if self.device.type == 'mps':
                    config['model'] = config['model'].to(self.device)
            
            elif model_key == 'instructblip':
                from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
                config['processor'] = InstructBlipProcessor.from_pretrained(config['model_id'])
                config['model'] = InstructBlipForConditionalGeneration.from_pretrained(
                    config['model_id'],
                    torch_dtype=torch.float16 if self.device.type == 'mps' else torch.float32,
                    device_map="auto" if self.device.type != 'mps' else None
                )
                if self.device.type == 'mps':
                    config['model'] = config['model'].to(self.device)
            
            config['loaded'] = True
            print(f"✅ {config['name']} 載入成功")
            return True
            
        except Exception as e:
            print(f"❌ {config['name']} 載入失敗: {e}")
            return False
    
    def _cleanup_model(self, model_key):
        """清理指定模型釋放記憶體"""
        if model_key not in self.model_configs:
            return
        
        config = self.model_configs[model_key]
        if config['loaded']:
            del config['model']
            del config['processor']
            config['model'] = None
            config['processor'] = None
            config['loaded'] = False
            
            if self.device.type == 'mps':
                torch.mps.empty_cache()
            elif self.device.type == 'cuda':
                torch.cuda.empty_cache()
    
    def extract_image_features(self, image_path):
        """
        提取圖片的FashionCLIP特徵
        
        Args:
            image_path: 圖片路徑
            
        Returns:
            np.array: 標準化的特徵向量
        """
        try:
            # 載入圖片
            image = Image.open(image_path).convert('RGB')
            
            # 提取 FashionCLIP 特徵
            with torch.no_grad():
                clip_inputs = self.model.clip_processor(images=image, return_tensors="pt").to(self.device)
                clip_features = self.model.clip_model.get_image_features(**clip_inputs)
                clip_features = F.normalize(clip_features, p=2, dim=1)
                
                # 根據是否使用訓練模型決定是否通過映射層
                if self.use_trained_model:
                    # 通過訓練好的映射層
                    outputs = self.model.forward(clip_features)
                    fashion_embedding = outputs['fashion_embedding'].cpu().numpy()
                else:
                    # 直接使用原始 FashionCLIP 特徵
                    fashion_embedding = clip_features.cpu().numpy()
                
                # 標準化
                fashion_embedding = fashion_embedding / np.linalg.norm(fashion_embedding)
                
                return fashion_embedding.flatten()
                
        except Exception as e:
            print(f"❌ 特徵提取失敗: {e}")
            return None
    
    def find_similar_outfits(self, image_path, gender, top_k=5, style_preference=None):
        """
        找出最相似的穿搭
        
        Args:
            image_path: 用戶圖片路徑
            gender: 性別過濾
            top_k: 返回前k個結果
            style_preference: 風格偏好
            
        Returns:
            list: 相似穿搭列表
        """
        print(f"🔍 分析圖片: {image_path}")
        
        # 提取查詢圖片特徵
        query_features = self.extract_image_features(image_path)
        if query_features is None:
            return []
        
        # 過濾數據集
        filtered_indices = []
        for i, sample in enumerate(self.dataset):
            # 性別過濾
            if sample['gender'] != gender:
                continue
            
            # 風格過濾（可選）
            if style_preference and sample['style'] != style_preference:
                continue
                
            filtered_indices.append(i)
        
        if not filtered_indices:
            print("❌ 沒有符合條件的樣本")
            return []
        
        # 計算相似度
        filtered_features = self.dataset_features[filtered_indices]
        similarities = cosine_similarity([query_features], filtered_features)[0]
        
        # 排序並選擇前k個
        sorted_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in sorted_indices:
            original_idx = filtered_indices[idx]
            sample = self.dataset[original_idx]
            
            results.append({
                'path': sample['path'],
                'style': sample['style'],
                'gender': sample['gender'],
                'similarity': float(similarities[idx]),
                'score': float(similarities[idx] * 10),  # 轉換為0-10分
                'features': sample['features']
            })
        
        print(f"✅ 找到 {len(results)} 個相似穿搭")
        return results
    
    def extract_detailed_features(self, image_path):
        """
        提取詳細的圖片特徵用於比較分析
        
        Args:
            image_path: 圖片路徑
            
        Returns:
            dict: 包含多種特徵的字典
        """
        try:
            image = Image.open(image_path).convert('RGB')
            
            features = {
                'clip_features': None,
                'image': image,
                'path': image_path
            }
            
            # 提取CLIP特徵
            with torch.no_grad():
                clip_inputs = self.model.clip_processor(images=image, return_tensors="pt").to(self.device)
                clip_features = self.model.clip_model.get_image_features(**clip_inputs)
                clip_features = F.normalize(clip_features, p=2, dim=1)
                features['clip_features'] = clip_features.cpu().numpy().flatten()
            
            return features
            
        except Exception as e:
            print(f"❌ 詳細特徵提取失敗: {e}")
            return None
    
    def analyze_feature_differences(self, user_features, target_features):
        """
        分析用戶圖片與目標圖片的特徵差異
        
        Args:
            user_features: 用戶圖片特徵
            target_features: 目標圖片特徵
            
        Returns:
            dict: 特徵差異分析結果
        """
        try:
            user_clip = user_features['clip_features']
            target_clip = target_features['clip_features']
            
            # 計算整體相似度
            overall_similarity = cosine_similarity([user_clip], [target_clip])[0][0]
            
            # 計算特徵向量差異
            feature_diff = target_clip - user_clip
            diff_magnitude = np.linalg.norm(feature_diff)
            
            # 找出差異最大的維度
            diff_indices = np.argsort(np.abs(feature_diff))[-20:][::-1]  # 前20個差異最大的維度
            
            analysis = {
                'overall_similarity': float(overall_similarity),
                'difference_magnitude': float(diff_magnitude),
                'key_differences': diff_indices.tolist(),
                'feature_diff_vector': feature_diff.tolist()
            }
            
            return analysis
            
        except Exception as e:
            print(f"❌ 特徵差異分析失敗: {e}")
            return {}
    
    def generate_ai_advice_parallel(self, user_image_path, target_outfits, models=['blip2', 'llava']):
        """
        使用多個AI模型並行生成建議
        
        Args:
            user_image_path: 用戶圖片路徑
            target_outfits: 目標穿搭列表
            models: 要使用的模型列表
            
        Returns:
            dict: 多模型生成結果
        """
        print(f"🧠 開始多模型AI分析...")
        print(f"🎯 使用模型: {[self.model_configs[m]['name'] for m in models if m in self.model_configs]}")
        
        results = {}
        
        # 提取用戶圖片特徵
        user_features = self.extract_detailed_features(user_image_path)
        if not user_features:
            return {"error": "無法提取用戶圖片特徵"}
        
        # 對每個目標穿搭進行分析
        for i, target_outfit in enumerate(target_outfits[:3]):  # 只分析前3個最相似的
            print(f"\n📸 分析目標穿搭 {i+1}: {target_outfit['style']}")
            
            # 提取目標圖片特徵
            target_features = self.extract_detailed_features(target_outfit['path'])
            if not target_features:
                continue
            
            # 分析特徵差異
            feature_analysis = self.analyze_feature_differences(user_features, target_features)
            
            # 為每個模型生成建議
            outfit_results = {
                'target_info': target_outfit,
                'feature_analysis': feature_analysis,
                'ai_suggestions': {}
            }
            
            # 並行處理多個模型
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_to_model = {}
                
                for model_key in models:
                    if model_key in self.model_configs:
                        future = executor.submit(
                            self._generate_single_model_advice,
                            model_key,
                            user_image_path,
                            target_outfit['path'],
                            feature_analysis
                        )
                        future_to_model[future] = model_key
                
                # 收集結果
                for future in as_completed(future_to_model):
                    model_key = future_to_model[future]
                    try:
                        advice = future.result(timeout=60)  # 60秒超時
                        outfit_results['ai_suggestions'][model_key] = advice
                    except Exception as e:
                        print(f"❌ {self.model_configs[model_key]['name']} 生成失敗: {e}")
                        outfit_results['ai_suggestions'][model_key] = {
                            'error': str(e),
                            'advice': '生成失敗，請重試'
                        }
            
            results[f'target_{i+1}'] = outfit_results
        
        print("✅ 多模型AI分析完成")
        return results
    
    def _generate_single_model_advice(self, model_key, user_image_path, target_image_path, feature_analysis):
        """
        使用單個模型生成建議
        
        Args:
            model_key: 模型鍵
            user_image_path: 用戶圖片路徑
            target_image_path: 目標圖片路徑
            feature_analysis: 特徵分析結果
            
        Returns:
            dict: 單個模型的建議結果
        """
        try:
            # 載入模型
            if not self._load_model(model_key):
                return {'error': '模型載入失敗'}
            
            config = self.model_configs[model_key]
            
            # 載入圖片
            user_image = Image.open(user_image_path).convert('RGB')
            target_image = Image.open(target_image_path).convert('RGB')
            
            # 根據模型類型生成建議
            if model_key == 'blip2':
                advice = self._generate_blip2_advice(
                    config, user_image, target_image, feature_analysis
                )
            elif model_key == 'llava':
                advice = self._generate_llava_advice(
                    config, user_image, target_image, feature_analysis
                )
            elif model_key == 'instructblip':
                advice = self._generate_instructblip_advice(
                    config, user_image, target_image, feature_analysis
                )
            else:
                advice = {'error': '未支援的模型類型'}
            
            # 清理模型（可選，節省記憶體）
            # self._cleanup_model(model_key)
            
            return advice
            
        except Exception as e:
            return {'error': f'生成失敗: {str(e)}'}
    
    def _generate_blip2_advice(self, config, user_image, target_image, feature_analysis):
        """使用BLIP-2生成建議"""
        try:
            similarity = feature_analysis.get('overall_similarity', 0)
            
            # 構建智能提示
            if similarity > 0.8:
                prompt = "Describe this stylish outfit and suggest how to achieve this look:"
            elif similarity > 0.6:
                prompt = "What styling improvements would you suggest for this outfit?"
            else:
                prompt = "Analyze this fashion style and give specific advice:"
            
            inputs = config['processor'](
                images=target_image,
                text=prompt,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = config['model'].generate(
                    **inputs,
                    max_new_tokens=50,
                    num_beams=2,
                    temperature=0.6,
                    do_sample=True,
                    early_stopping=True,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3
                )
            
            advice_text = config['processor'].decode(outputs[0], skip_special_tokens=True)
            
            # 清理輸出
            advice_text = self._clean_blip2_output(advice_text, prompt)
            
            # 驗證輸出質量
            if self._is_valid_advice(advice_text):
                confidence = 'high' if len(advice_text) > 30 else 'medium'
            else:
                # 如果輸出無效，使用備用策略
                advice_text = self._generate_blip2_fallback(similarity)
                confidence = 'low'
            
            return {
                'model': 'BLIP-2',
                'advice': advice_text,
                'similarity_score': similarity,
                'confidence': confidence
            }
            
        except Exception as e:
            return {'error': f'BLIP-2生成失敗: {str(e)}'}
    
    def _clean_blip2_output(self, output, prompt):
        """改進的BLIP-2輸出清理"""
        try:
            # 移除提示文字
            cleaned = output.replace(prompt, "").strip()
            
            # 移除常見的無用前綴和後綴
            unwanted_patterns = [
                "The image shows", "This image depicts", "In this image", "I can see",
                "The outfit consists of", "The person is wearing", "This person is wearing",
                "Looking at this outfit", "This outfit features", "The style is",
                "Question:", "Answer:", "Caption:", "Description:"
            ]
            
            for pattern in unwanted_patterns:
                if cleaned.lower().startswith(pattern.lower()):
                    cleaned = cleaned[len(pattern):].strip()
            
            # 移除特殊字符和亂碼
            import re
            # 移除多餘的符號和數字雜訊
            cleaned = re.sub(r'[^\w\s.,!?-]', '', cleaned)
            cleaned = re.sub(r'\d+\)', '', cleaned)  # 移除編號
            cleaned = re.sub(r'^[-\s]*', '', cleaned)  # 移除開頭的連字符
            
            # 移除過短或過長的句子
            sentences = cleaned.split('.')
            valid_sentences = []
            for sentence in sentences:
                sentence = sentence.strip()
                if 5 < len(sentence) < 100 and not any(char in sentence for char in ['`', '"', "'"]):
                    valid_sentences.append(sentence)
            
            if valid_sentences:
                cleaned = '. '.join(valid_sentences[:2])  # 只取前兩個有效句子
                if not cleaned.endswith('.'):
                    cleaned += '.'
            else:
                return ""
            
            return cleaned.strip()
            
        except Exception:
            return ""
    
    def _is_valid_advice(self, advice):
        """驗證建議的有效性"""
        if not advice or len(advice) < 10:
            return False
        
        # 檢查是否包含亂碼或無意義內容
        invalid_patterns = [
            r'I have been', r'resistance of', r'first time of',
            r'^-\s*I\s*', r'^\s*[-`"\']+', r'\d+\s*[-\)]+',
            r'world$', r'time$', r'^[^a-zA-Z]*$'
        ]
        
        import re
        for pattern in invalid_patterns:
            if re.search(pattern, advice, re.IGNORECASE):
                return False
        
        # 檢查是否包含合理的時尚詞彙
        fashion_keywords = [
            'style', 'outfit', 'wear', 'fashion', 'clothing', 'dress',
            'shirt', 'pants', 'shoes', 'accessory', 'color', 'fit',
            'layer', 'casual', 'formal', 'trendy', 'classic'
        ]
        
        advice_lower = advice.lower()
        has_fashion_content = any(keyword in advice_lower for keyword in fashion_keywords)
        
        return has_fashion_content
    
    def _generate_blip2_fallback(self, similarity):
        """BLIP-2備用建議生成"""
        if similarity > 0.8:
            return "This outfit has a great foundation. Consider adding one statement piece to elevate the look."
        elif similarity > 0.6:
            return "Try incorporating more trendy accessories or adjusting the fit for a more polished appearance."
        else:
            return "Consider experimenting with different color combinations and layering techniques to enhance your style."
    
    def _generate_llava_advice(self, config, user_image, target_image, feature_analysis):
        """使用LLaVA生成建議"""
        try:
            similarity = feature_analysis.get('overall_similarity', 0)
            
            # LLaVA使用對話格式
            if similarity > 0.8:
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": target_image},
                            {"type": "text", "text": "Look at this outfit and suggest how to achieve this style. What specific clothing items and styling choices make this look work?"}
                        ]
                    }
                ]
            else:
                conversation = [
                    {
                        "role": "user", 
                        "content": [
                            {"type": "image", "image": target_image},
                            {"type": "text", "text": "Describe this outfit style and give specific advice on how to recreate this look. Focus on key clothing pieces, colors, and styling details."}
                        ]
                    }
                ]
            
            prompt = config['processor'].apply_chat_template(conversation, add_generation_prompt=True)
            inputs = config['processor'](
                text=prompt,
                images=target_image,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = config['model'].generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.6,
                    do_sample=True
                )
            
            advice_text = config['processor'].decode(outputs[0], skip_special_tokens=True)
            
            # 清理LLaVA輸出
            advice_text = self._clean_llava_output(advice_text, prompt)
            
            return {
                'model': 'LLaVA-Next',
                'advice': advice_text,
                'similarity_score': similarity,
                'confidence': 'high' if len(advice_text) > 20 else 'medium'
            }
            
        except Exception as e:
            return {'error': f'LLaVA生成失敗: {str(e)}'}
    
    def _clean_llava_output(self, output, prompt):
        """清理LLaVA特殊輸出格式"""
        try:
            # LLaVA的輸出可能包含特殊標記
            cleaned = output.replace(prompt, "").strip()
            
            # 移除LLaVA特有的標記和指令格式
            llava_markers = [
                "<|im_start|>", "<|im_end|>", "assistant\n", "user\n",
                "[INST]", "[/INST]", "<s>", "</s>", "<|endoftext|>"
            ]
            for marker in llava_markers:
                cleaned = cleaned.replace(marker, "")
            
            # 移除重複的指令內容
            import re
            # 移除開頭的指令重複
            instruction_patterns = [
                r'^.*?Look at this outfit and suggest.*?What specific clothing.*?\?\s*',
                r'^.*?Describe this outfit style.*?Focus on key clothing.*?\.\s*',
                r'^.*?\[INST\].*?\[/INST\]\s*',
                r'Look at this outfit and suggest.*?What specific clothing.*?\?\s*',
                r'Describe this outfit style.*?Focus on key clothing.*?\.\s*'
            ]
            
            for pattern in instruction_patterns:
                cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL | re.IGNORECASE)
            
            # 移除開頭重複的問句
            question_patterns = [
                r'^.*?What specific clothing items.*?\?\s*',
                r'^.*?How to achieve this style.*?\?\s*',
                r'^.*?Focus on key clothing pieces.*?\.\s*'
            ]
            
            for pattern in question_patterns:
                cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
            
            # 提取有用的建議內容
            # 首先嘗試找到完整的句子
            sentences = cleaned.split('.')
            useful_sentences = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 10:
                    continue
                    
                # 跳過重複的指令內容
                if any(skip_phrase in sentence.lower() for skip_phrase in [
                    'look at this outfit', 'describe this outfit', 'what specific',
                    'focus on key', 'how to achieve', 'suggest how to'
                ]):
                    continue
                
                # 保留有用的時尚建議
                if any(keyword in sentence.lower() for keyword in [
                    'consider', 'try', 'add', 'wear', 'choose', 'opt for', 
                    'style', 'outfit', 'clothing', 'accessory', 'layer',
                    'jacket', 'shirt', 'pants', 'shoes', 'color',
                    'versatile', 'casual', 'formal', 'trendy', 'classic'
                ]):
                    # 清理格式標記
                    sentence = re.sub(r'\*\*.*?\*\*', '', sentence)  # 移除粗體標記
                    sentence = re.sub(r'^\d+\.\s*', '', sentence)   # 移除編號
                    sentence = sentence.strip()
                    
                    if len(sentence) > 15:
                        useful_sentences.append(sentence)
            
            if useful_sentences:
                # 只取前2個最有用的建議
                cleaned = '. '.join(useful_sentences[:2])
                if not cleaned.endswith('.'):
                    cleaned += '.'
            else:
                # 如果沒有找到好的句子，嘗試提取第一個有意義的段落
                paragraphs = cleaned.split('\n')
                for paragraph in paragraphs:
                    paragraph = paragraph.strip()
                    
                    # 跳過指令內容
                    if any(skip_phrase in paragraph.lower() for skip_phrase in [
                        'look at this outfit', 'describe this outfit', 'what specific',
                        'focus on key', 'how to achieve'
                    ]):
                        continue
                    
                    if len(paragraph) > 30 and any(keyword in paragraph.lower() for keyword in [
                        'outfit', 'style', 'wear', 'clothing', 'fashion'
                    ]):
                        # 提取第一句
                        first_sentence = paragraph.split('.')[0].strip()
                        if len(first_sentence) > 20:
                            cleaned = first_sentence + '.'
                            break
                else:
                    return ""
            
            # 最終清理
            cleaned = cleaned.strip()
            
            # 確保不包含重複的指令
            if any(phrase in cleaned.lower() for phrase in [
                'look at this outfit', 'what specific clothing', 'describe this outfit'
            ]):
                return ""
                
            if len(cleaned) < 25:
                return ""
                
            return cleaned
            
        except Exception:
            return ""
    
    def _generate_instructblip_advice(self, config, user_image, target_image, feature_analysis):
        """使用InstructBLIP生成建議"""
        try:
            similarity = feature_analysis.get('overall_similarity', 0)
            
            if similarity > 0.7:
                instruction = "Provide specific styling tips to refine this outfit and make it more polished."
            else:
                instruction = "Analyze this outfit style and give detailed advice on how to achieve this look."
            
            inputs = config['processor'](
                images=target_image,
                text=instruction,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = config['model'].generate(
                    **inputs,
                    max_new_tokens=90,
                    num_beams=3,
                    temperature=0.7,
                    do_sample=True
                )
            
            advice_text = config['processor'].decode(outputs[0], skip_special_tokens=True)
            advice_text = self._clean_model_output(advice_text, instruction)
            
            return {
                'model': 'InstructBLIP',
                'advice': advice_text,
                'similarity_score': similarity,
                'confidence': 'high' if len(advice_text) > 20 else 'medium'
            }
            
        except Exception as e:
            return {'error': f'InstructBLIP生成失敗: {str(e)}'}
    
    def _clean_model_output(self, output, prompt):
        """清理模型輸出"""
        # 移除提示文字
        cleaned = output.replace(prompt, "").strip()
        
        # 移除常見的無用前綴
        prefixes_to_remove = [
            "The image shows", "This image depicts", "In this image",
            "The outfit consists of", "The person is wearing",
            "Looking at this outfit", "This outfit features"
        ]
        
        for prefix in prefixes_to_remove:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        
        # 移除多餘的標點
        cleaned = cleaned.strip(".,!?:：").strip()
        
        return cleaned if len(cleaned) > 10 else "建議生成中遇到問題，請重試"
    
    def analyze_and_recommend(self, image_path, gender, top_k=5, style_preference=None, 
                            models=['blip2', 'llava']):
        """
        完整的AI驅動分析和推薦流程
        
        Args:
            image_path: 用戶圖片路徑
            gender: 性別
            top_k: 推薦數量
            style_preference: 風格偏好
            models: 使用的AI模型列表
            
        Returns:
            dict: 完整的AI分析結果
        """
        print(f"\n🎯 開始AI驅動穿搭分析...")
        print(f"📁 圖片: {image_path}")
        print(f"👤 性別: {gender}")
        print(f"🎨 風格偏好: {style_preference or '無限制'}")
        print(f"🧠 AI模型: {models}")
        
        # 檢查文件是否存在
        if not os.path.exists(image_path):
            return {"error": f"圖片文件不存在: {image_path}"}
        
        try:
            # 第一步：找出相似穿搭
            similar_outfits = self.find_similar_outfits(
                image_path, gender, top_k, style_preference
            )
            
            if not similar_outfits:
                return {
                    "error": "沒有找到相似的穿搭",
                    "suggestion": "請嘗試調整性別或風格偏好設置"
                }
            
            # 第二步：AI多模型分析
            ai_analysis = self.generate_ai_advice_parallel(
                image_path, similar_outfits, models
            )
            
            # 第三步：綜合結果
            result = {
                "status": "success",
                "input_image": image_path,
                "gender": gender,
                "style_preference": style_preference,
                "similar_outfits": similar_outfits,
                "ai_analysis": ai_analysis,
                "models_used": models,
                "summary": {
                    "best_match": similar_outfits[0] if similar_outfits else None,
                    "total_recommendations": len(similar_outfits),
                    "ai_models_count": len(models)
                }
            }
            
            print(f"✅ AI分析完成！")
            print(f"🏆 最佳匹配: {result['summary']['best_match']['style']} ({result['summary']['best_match']['score']:.1f}分)")
            print(f"🧠 AI模型數量: {result['summary']['ai_models_count']}")
            
            return result
            
        except Exception as e:
            print(f"❌ 分析過程中發生錯誤: {e}")
            return {"error": f"分析失敗: {str(e)}"}
    
    def print_ai_recommendations(self, results):
        """
        美化打印AI推薦結果
        
        Args:
            results: AI分析結果
        """
        if "error" in results:
            print(f"❌ 錯誤: {results['error']}")
            return
        
        print(f"\n{'='*80}")
        print(f"🤖 AI驅動穿搭推薦結果")
        print(f"{'='*80}")
        
        # 基本信息
        print(f"📸 分析圖片: {results['input_image']}")
        print(f"👤 性別: {results['gender']}")
        print(f"🎨 風格偏好: {results['style_preference'] or '無限制'}")
        print(f"🧠 使用的AI模型: {', '.join(results['models_used'])}")
        
        # 最佳匹配
        best_match = results['summary']['best_match']
        print(f"\n🥇 最佳匹配:")
        print(f"  風格: {best_match['style']}")
        print(f"  相似度: {best_match['similarity']:.3f}")
        print(f"  評分: {best_match['score']:.1f}/10")
        print(f"  圖片: {best_match['path']}")
        
        # AI分析結果
        ai_analysis = results['ai_analysis']
        print(f"\n🧠 AI多模型分析結果:")
        
        for target_key, target_data in ai_analysis.items():
            if 'error' in target_data:
                continue
                
            target_info = target_data['target_info']
            feature_analysis = target_data['feature_analysis']
            ai_suggestions = target_data['ai_suggestions']
            
            print(f"\n📸 {target_key.upper()}: {target_info['style']} 風格")
            print(f"  相似度: {feature_analysis.get('overall_similarity', 0):.3f}")
            print(f"  差異程度: {feature_analysis.get('difference_magnitude', 0):.3f}")
            
            # 顯示各AI模型的建議
            for model_key, suggestion in ai_suggestions.items():
                model_name = suggestion.get('model', model_key.upper())
                confidence = suggestion.get('confidence', 'unknown')
                
                print(f"\n  🤖 {model_name} 建議 (信心度: {confidence}):")
                if 'error' in suggestion:
                    print(f"    ❌ {suggestion['error']}")
                else:
                    advice = suggestion.get('advice', '無建議')
                    print(f"    💡 {advice}")
        
        # 模型比較總結
        print(f"\n📊 模型建議比較:")
        self._print_model_comparison(ai_analysis)
        
        print(f"\n{'='*80}")
    
    def _print_model_comparison(self, ai_analysis):
        """打印模型建議比較"""
        model_advice_count = {}
        model_errors = {}
        
        for target_data in ai_analysis.values():
            if 'ai_suggestions' in target_data:
                for model_key, suggestion in target_data['ai_suggestions'].items():
                    model_name = suggestion.get('model', model_key.upper())
                    
                    if model_name not in model_advice_count:
                        model_advice_count[model_name] = 0
                        model_errors[model_name] = 0
                    
                    if 'error' in suggestion:
                        model_errors[model_name] += 1
                    else:
                        model_advice_count[model_name] += 1
        
        for model_name in model_advice_count:
            success_count = model_advice_count[model_name]
            error_count = model_errors[model_name]
            total = success_count + error_count
            success_rate = (success_count / total * 100) if total > 0 else 0
            
            print(f"  {model_name}: 成功 {success_count}/{total} ({success_rate:.1f}%)")


def main():
    """主函數 - AI驅動命令行界面"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AI驅動穿搭推薦系統')
    parser.add_argument('--image', type=str, required=True, help='用戶圖片路徑')
    parser.add_argument('--gender', type=str, required=True, choices=['MEN', 'WOMEN'], help='性別')
    parser.add_argument('--top_k', type=int, default=5, help='推薦數量')
    parser.add_argument('--style', type=str, help='風格偏好')
    parser.add_argument('--model', type=str, default='simple_fashion_model_final_best.pth', help='模型路徑')
    parser.add_argument('--labels', type=str, default='simple_dataset_labels.json', help='標籤文件路徑')
    parser.add_argument('--ai_models', type=str, nargs='+', default=['blip2', 'llava'], 
                       help='使用的AI模型 (可選: blip2, llava, instructblip)')
    parser.add_argument('--save', type=str, help='保存結果到JSON文件')
    
    args = parser.parse_args()
    
    try:
        # 初始化系統
        system = MultiModelAIRecommendationSystem(args.model, args.labels)
        
        # 進行AI分析
        results = system.analyze_and_recommend(
            args.image, 
            args.gender, 
            args.top_k, 
            args.style,
            args.ai_models
        )
        
        # 顯示結果
        system.print_ai_recommendations(results)
        
        # 保存結果
        if args.save:
            with open(args.save, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"💾 結果已保存到: {args.save}")
            
    except Exception as e:
        print(f"❌ 系統錯誤: {e}")


if __name__ == "__main__":
    main() 