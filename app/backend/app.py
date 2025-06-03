#!/usr/bin/env python3
"""
智能穿搭推薦系統 FastAPI 應用
提供基於多模態AI的穿搭分析和推薦服務
優化為線上服務和移動端應用
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import uuid
import asyncio
import logging
from datetime import datetime
import shutil
import json
from pathlib import Path

# 導入我們的推薦系統
from improved_recommender import ImprovedFashionRecommendationSystem

# 設置日誌 - 添加更詳細的格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

# 創建FastAPI應用
app = FastAPI(
    title="智能穿搭推薦系統 API",
    description="基於多模態AI的智能穿搭分析與推薦服務 - 移動端優化版",
    version="2.1.1",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS設置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生產環境中應該設置具體的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 創建必要的目錄
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# 數據集目錄設置
DATASET_DIR = Path("../dataset")
if not DATASET_DIR.exists():
    DATASET_DIR = Path("dataset")  # 備用路徑

# 掛載靜態文件
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# 掛載數據集靜態文件
if DATASET_DIR.exists():
    app.mount("/dataset", StaticFiles(directory=str(DATASET_DIR)), name="dataset")
    logger.info(f"📁 數據集目錄掛載: {DATASET_DIR.absolute()}")
else:
    logger.warning("⚠️ 數據集目錄不存在，圖片可能無法顯示")

# 全域推薦系統實例
recommendation_system = None

# 請求和回應模型
class RecommendationRequest(BaseModel):
    gender: str
    style_preference: Optional[str] = None
    top_k: int = 4
    strategy: str = "balanced"

class AdviceRequest(BaseModel):
    user_image_path: str
    target_image_path: str
    target_style: str
    ai_models: List[str] = ["rule_based", "clip"]

class DetailedSimilarity(BaseModel):
    visual_similarity: float
    main_component_similarity: float
    style_similarity: Optional[float] = None

class RecommendationItem(BaseModel):
    recommendation_id: str
    path: str
    style: str
    gender: str
    similarity: float
    score: float
    detailed_similarity: DetailedSimilarity

class QuickRecommendationResponse(BaseModel):
    status: str
    request_id: str
    input_image_url: str
    analysis_time: float
    recommendations: List[RecommendationItem]
    style_analysis: Dict[str, Any]

class AdviceResponse(BaseModel):
    status: str
    request_id: str
    recommendation_id: str
    target_style: str
    ai_advice: Dict[str, str]
    analysis_time: float

class FeatureAnalysis(BaseModel):
    top_feature: str
    score: float
    all_scores: Dict[str, float]

class ErrorResponse(BaseModel):
    status: str
    error: str
    request_id: str

# 啟動事件
@app.on_event("startup")
async def startup_event():
    """應用啟動時初始化推薦系統"""
    global recommendation_system
    try:
        logger.info("🚀 正在初始化智能穿搭推薦系統...")
        recommendation_system = ImprovedFashionRecommendationSystem()
        logger.info("✅ 推薦系統初始化完成")
    except Exception as e:
        logger.error(f"❌ 推薦系統初始化失敗: {e}")
        raise

# 關閉事件
@app.on_event("shutdown")
async def shutdown_event():
    """應用關閉時的清理工作"""
    logger.info("🔄 正在關閉推薦系統...")

def cleanup_old_files():
    """清理舊文件"""
    try:
        # 清理超過2小時的上傳文件
        for file_path in UPLOAD_DIR.glob("*"):
            if file_path.is_file():
                file_age = datetime.now().timestamp() - file_path.stat().st_mtime
                if file_age > 7200:  # 2小時
                    file_path.unlink()
    except Exception as e:
        logger.error(f"清理文件失敗: {e}")

def convert_path_to_url(file_path: str) -> str:
    """轉換檔案路徑為URL路徑"""
    # 處理相對路徑
    if file_path.startswith("../dataset/"):
        # 轉換為 /dataset/ 路徑
        return file_path.replace("../dataset/", "/dataset/")
    elif file_path.startswith("dataset/"):
        return "/" + file_path
    elif file_path.startswith("/dataset/"):
        return file_path
    else:
        # 如果是絕對路徑，嘗試提取相對部分
        if "dataset" in file_path:
            parts = file_path.split("dataset")
            if len(parts) > 1:
                return "/dataset" + parts[-1]
    
    # 預設返回原路徑
    return file_path

async def process_quick_recommendation(
    image_path: str,
    request_data: RecommendationRequest,
    request_id: str
) -> Dict[str, Any]:
    """快速推薦處理邏輯（不包含文字建議生成）"""
    try:
        start_time = datetime.now()
        logger.info(f"🔍 開始處理推薦請求 {request_id}")
        logger.info(f"📝 請求參數: 性別={request_data.gender}, 風格={request_data.style_preference}, 策略={request_data.strategy}")
        
        # 執行快速推薦分析（不使用AI模型）
        similar_outfits = recommendation_system.find_similar_outfits_improved(
            image_path=image_path,
            gender=request_data.gender,
            top_k=request_data.top_k,
            style_preference=request_data.style_preference,
            similarity_weights=_get_strategy_weights(request_data.strategy)
        )
        
        if not similar_outfits:
            raise HTTPException(status_code=404, detail="沒有找到相似的穿搭")
        
        # 生成風格分析
        style_analysis = _generate_style_analysis(similar_outfits)
        
        # 計算處理時間
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # 輸出推薦結果到terminal
        logger.info("=" * 60)
        logger.info(f"🎯 推薦分析完成 (請求ID: {request_id})")
        logger.info(f"⏱️  分析時間: {processing_time:.2f}秒")
        logger.info(f"👤 目標用戶: {request_data.gender}")
        logger.info(f"🎨 主要風格: {style_analysis.get('dominant_style', 'N/A')}")
        logger.info(f"📊 平均相似度: {style_analysis.get('average_visual_similarity', 0)*100:.1f}%")
        logger.info("")
        logger.info("📸 推薦圖片列表:")
        
        # 格式化回應（添加recommendation_id和URL轉換）
        recommendations = []
        for i, rec in enumerate(similar_outfits):
            recommendation_id = f"{request_id}_rec_{i}"
            
            # 轉換路徑為URL
            original_path = rec["path"]
            url_path = convert_path_to_url(original_path)
            
            # 輸出每個推薦的詳細信息
            logger.info(f"  {i+1}. 風格: {rec['style']}")
            logger.info(f"     原始路徑: {original_path}")
            logger.info(f"     URL路徑: {url_path}")
            logger.info(f"     相似度: {rec['similarity']*100:.1f}%")
            logger.info(f"     評分: {rec['score']:.1f}/10")
            logger.info(f"     推薦ID: {recommendation_id}")
            logger.info("")
            
            recommendations.append({
                "recommendation_id": recommendation_id,
                "path": url_path,  # 使用轉換後的URL路徑
                "style": rec["style"],
                "gender": rec["gender"],
                "similarity": rec["similarity"],
                "score": rec["score"],
                "detailed_similarity": rec["detailed_similarity"]
            })
        
        # 輸出風格分佈
        if 'style_distribution' in style_analysis:
            logger.info("🎨 風格分佈:")
            for style, count in style_analysis['style_distribution'].items():
                logger.info(f"  - {style}: {count} 張")
        
        logger.info("=" * 60)
        
        response_data = {
            "status": "success",
            "request_id": request_id,
            "input_image_url": f"/uploads/{os.path.basename(image_path)}",
            "analysis_time": processing_time,
            "recommendations": recommendations,
            "style_analysis": style_analysis
        }
        
        return response_data
        
    except Exception as e:
        logger.error(f"❌ 處理快速推薦請求失敗: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def _get_strategy_weights(strategy: str) -> Dict[str, float]:
    """獲取策略權重"""
    strategy_weights = {
        'pure_visual': {'original': 1.0, 'pca': 0.0, 'mapped': 0.0},
        'balanced': {'original': 0.5, 'pca': 0.3, 'mapped': 0.2},
        'style_aware': {'original': 0.3, 'pca': 0.2, 'mapped': 0.5}
    }
    return strategy_weights.get(strategy, strategy_weights['balanced'])

def _generate_style_analysis(similar_outfits: List[Dict]) -> Dict[str, Any]:
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
        "average_visual_similarity": float(sum(visual_similarities) / len(visual_similarities)),
        "average_style_similarity": float(sum(style_similarities) / len(style_similarities)) if style_similarities else 0,
        "max_visual_similarity": float(max(visual_similarities))
    }

async def process_advice_generation(
    user_image_path: str,
    target_image_path: str,
    target_style: str,
    ai_models: List[str],
    request_id: str
) -> Dict[str, str]:
    """生成特定推薦的AI建議"""
    try:
        logger.info(f"🤖 開始生成AI建議 (請求ID: {request_id})")
        logger.info(f"🎯 目標風格: {target_style}")
        logger.info(f"🧠 使用模型: {', '.join(ai_models)}")
        
        # 載入所需的AI模型
        recommendation_system._load_ai_models(ai_models)
        
        # 創建模擬推薦對象
        recommendation = {
            'path': target_image_path,
            'style': target_style,
            'similarity': 0.8  # 模擬相似度
        }
        
        # 生成建議
        advice_results = {}
        
        for model_key in ai_models:
            if model_key in recommendation_system.loaded_ai_models or model_key in ['rule_based', 'clip']:
                try:
                    logger.info(f"🔄 正在使用 {model_key} 模型生成建議...")
                    advice = recommendation_system._generate_model_advice(
                        model_key, user_image_path, recommendation
                    )
                    advice_results[model_key] = advice
                    logger.info(f"✅ {model_key} 模型建議生成完成")
                except Exception as e:
                    logger.error(f"❌ {model_key} 模型建議生成失敗: {str(e)}")
                    advice_results[model_key] = f"生成失敗: {str(e)}"
            else:
                advice_results[model_key] = "模型未載入"
        
        # 輸出建議結果
        logger.info("📝 AI建議生成結果:")
        for model, advice in advice_results.items():
            logger.info(f"  🤖 {model}: {advice[:100]}..." if len(advice) > 100 else f"  🤖 {model}: {advice}")
        
        return advice_results
        
    except Exception as e:
        logger.error(f"❌ 生成建議失敗: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", tags=["基本信息"])
async def root():
    """API根端點"""
    return {
        "message": "智能穿搭推薦系統 API - 移動端優化版",
        "version": "2.1.1",
        "status": "運行中",
        "docs": "/docs",
        "features": [
            "快速推薦查找",
            "分離式AI建議生成",
            "Fashion-CLIP 特徵分析",
            "移動端優化",
            "圖片路徑修復"
        ]
    }

@app.get("/health", tags=["健康檢查"])
async def health_check():
    """健康檢查端點"""
    try:
        system_status = "healthy" if recommendation_system else "unavailable"
        return {
            "status": "ok",
            "system": system_status,
            "timestamp": datetime.now().isoformat(),
            "dataset_mounted": DATASET_DIR.exists()
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"服務不可用: {str(e)}")

@app.post("/recommend", response_model=QuickRecommendationResponse, tags=["快速推薦"])
async def quick_recommend(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(..., description="穿搭圖片文件"),
    gender: str = Form(..., description="性別 (MEN/WOMEN)"),
    style_preference: Optional[str] = Form(None, description="風格偏好 (可選)"),
    top_k: int = Form(4, description="推薦數量"),
    strategy: str = Form("balanced", description="推薦策略")
):
    """
    快速推薦端點 - 只返回推薦列表，不生成AI建議
    
    適合移動端應用的快速響應接口
    - **image**: 上傳的穿搭圖片
    - **gender**: 性別選擇 (MEN 或 WOMEN)
    - **style_preference**: 風格偏好 (可選: CASUAL, STREET, FORMAL, BOHEMIAN)
    - **top_k**: 推薦數量 (1-10)
    - **strategy**: 推薦策略 (pure_visual, balanced, style_aware)
    """
    
    # 生成請求ID
    request_id = str(uuid.uuid4())
    
    try:
        logger.info(f"📥 收到新的推薦請求 (ID: {request_id})")
        logger.info(f"📋 檔案名稱: {image.filename}")
        logger.info(f"📋 檔案大小: {image.size if hasattr(image, 'size') else 'N/A'} bytes")
        logger.info(f"📋 內容類型: {image.content_type}")
        
        # 驗證輸入
        if gender not in ["MEN", "WOMEN"]:
            raise HTTPException(status_code=400, detail="性別必須是 MEN 或 WOMEN")
        
        if not (1 <= top_k <= 10):
            raise HTTPException(status_code=400, detail="推薦數量必須在 1-10 之間")
        
        if strategy not in ["pure_visual", "balanced", "style_aware"]:
            raise HTTPException(status_code=400, detail="無效的推薦策略")
        
        # 檢查文件類型
        if not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="文件必須是圖片格式")
        
        # 保存上傳的圖片
        file_extension = os.path.splitext(image.filename)[1]
        if not file_extension:
            file_extension = ".jpg"
        
        filename = f"{request_id}{file_extension}"
        file_path = UPLOAD_DIR / filename
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        logger.info(f"💾 圖片已保存: {file_path}")
        
        # 創建請求對象
        request_data = RecommendationRequest(
            gender=gender,
            style_preference=style_preference,
            top_k=top_k,
            strategy=strategy
        )
        
        # 處理快速推薦請求
        result = await process_quick_recommendation(str(file_path), request_data, request_id)
        
        # 添加後台清理任務
        background_tasks.add_task(cleanup_old_files)
        
        logger.info(f"✅ 推薦請求處理完成 (ID: {request_id})")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 快速推薦失敗 (請求ID: {request_id}): {e}")
        return ErrorResponse(
            status="error",
            error=str(e),
            request_id=request_id
        )

@app.post("/advice", response_model=AdviceResponse, tags=["AI建議生成"])
async def generate_advice(
    user_image_path: str = Form(..., description="用戶圖片路徑"),
    target_image_path: str = Form(..., description="目標推薦圖片路徑"),
    target_style: str = Form(..., description="目標風格"),
    ai_models: str = Form("rule_based,clip", description="AI模型列表，用逗號分隔"),
    recommendation_id: str = Form(..., description="推薦ID")
):
    """
    生成特定推薦的AI建議
    
    為特定的推薦項目生成詳細的穿搭建議
    - **user_image_path**: 用戶圖片的相對路徑
    - **target_image_path**: 目標推薦圖片路徑
    - **target_style**: 目標風格
    - **ai_models**: 使用的AI模型 (rule_based, clip, llava, blip2, instructblip)
    - **recommendation_id**: 來自推薦響應的recommendation_id
    """
    
    request_id = str(uuid.uuid4())
    
    try:
        start_time = datetime.now()
        logger.info(f"🎯 收到建議生成請求 (ID: {request_id})")
        logger.info(f"📸 推薦ID: {recommendation_id}")
        
        # 解析AI模型列表
        model_list = [model.strip() for model in ai_models.split(",")]
        valid_models = ["rule_based", "clip", "llava", "blip2", "instructblip"]
        
        for model in model_list:
            if model not in valid_models:
                raise HTTPException(
                    status_code=400, 
                    detail=f"無效的AI模型: {model}. 有效選項: {', '.join(valid_models)}"
                )
        
        # 構建完整的用戶圖片路徑
        if user_image_path.startswith("/uploads/"):
            full_user_image_path = user_image_path[9:]  # 移除 /uploads/ 前綴
        else:
            full_user_image_path = user_image_path
        
        full_user_image_path = UPLOAD_DIR / full_user_image_path
        
        if not full_user_image_path.exists():
            raise HTTPException(status_code=404, detail="用戶圖片文件不存在")
        
        # 處理目標圖片路徑 - 轉換URL路徑回檔案路徑
        if target_image_path.startswith("/dataset/"):
            # 從URL路徑轉換為實際檔案路徑
            actual_target_path = DATASET_DIR / target_image_path[9:]  # 移除 /dataset/ 前綴
        else:
            actual_target_path = Path(target_image_path)
        
        if not actual_target_path.exists():
            logger.error(f"❌ 目標圖片文件不存在: {actual_target_path}")
            raise HTTPException(status_code=404, detail="目標圖片文件不存在")
        
        # 生成AI建議
        advice_results = await process_advice_generation(
            str(full_user_image_path),
            str(actual_target_path),
            target_style,
            model_list,
            request_id
        )
        
        # 計算處理時間
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        logger.info(f"✅ 建議生成完成 (ID: {request_id}, 耗時: {processing_time:.2f}秒)")
        
        return AdviceResponse(
            status="success",
            request_id=request_id,
            recommendation_id=recommendation_id,
            target_style=target_style,
            ai_advice=advice_results,
            analysis_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 生成建議失敗 (請求ID: {request_id}): {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract-features", tags=["特徵提取"])
async def extract_features(
    image: UploadFile = File(..., description="穿搭圖片文件")
):
    """
    提取圖片的詳細時尚特徵
    
    返回8大類特徵的詳細分析：顏色、服裝類型、風格、圖案、配飾、鞋類、材質、版型
    """
    
    request_id = str(uuid.uuid4())
    
    try:
        # 檢查文件類型
        if not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="文件必須是圖片格式")
        
        # 保存上傳的圖片
        file_extension = os.path.splitext(image.filename)[1] or ".jpg"
        filename = f"{request_id}{file_extension}"
        file_path = UPLOAD_DIR / filename
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        # 提取特徵
        features = recommendation_system.extract_detailed_fashion_features(str(file_path))
        
        if not features:
            raise HTTPException(status_code=500, detail="特徵提取失敗")
        
        return {
            "status": "success",
            "request_id": request_id,
            "features": features,
            "categories": list(features.keys()),
            "summary": {
                "總特徵類別": len(features),
                "最高信心度": max(data["score"] for data in features.values()),
                "主要特徵": {category: data["top_feature"] for category, data in features.items()}
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 特徵提取失敗 (請求ID: {request_id}): {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models", tags=["系統信息"])
async def get_available_models():
    """獲取可用的AI模型列表"""
    return {
        "available_models": {
            "rule_based": {
                "name": "規則系統",
                "description": "基於預設規則的穿搭建議",
                "speed": "快",
                "accuracy": "中等"
            },
            "clip": {
                "name": "FashionCLIP",
                "description": "Fashion-CLIP 詳細特徵分析",
                "speed": "快",
                "accuracy": "高"
            },
            "llava": {
                "name": "視覺語言模型",
                "description": "LLaVA 雙圖片比較分析",
                "speed": "慢",
                "accuracy": "極高"
            },
            "blip2": {
                "name": "圖像描述模型",
                "description": "BLIP-2 圖像理解和描述",
                "speed": "中等",
                "accuracy": "高"
            },
            "instructblip": {
                "name": "指令圖像模型",
                "description": "InstructBLIP 指令式圖像分析",
                "speed": "中等",
                "accuracy": "高"
            }
        },
        "strategies": {
            "pure_visual": "純視覺相似性推薦",
            "balanced": "平衡視覺和風格的推薦",
            "style_aware": "風格導向的推薦"
        },
        "supported_styles": [
            "CASUAL", "STREET", "FORMAL", "BOHEMIAN"
        ]
    }

@app.get("/stats", tags=["系統統計"])
async def get_system_stats():
    """獲取系統統計信息"""
    try:
        upload_count = len(list(UPLOAD_DIR.glob("*")))
        
        return {
            "status": "active",
            "upload_files": upload_count,
            "dataset_available": DATASET_DIR.exists(),
            "dataset_path": str(DATASET_DIR.absolute()),
            "system_info": {
                "dataset_size": len(recommendation_system.dataset) if recommendation_system else 0,
                "feature_categories": 8,
                "supported_models": 5,
                "supported_strategies": 3
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 啟動智能穿搭推薦系統 API 服務器 - 移動端優化版 v2.1.1")
    print("📖 API 文檔: http://localhost:8000/docs")
    print("🔍 API 測試: http://localhost:8000/redoc")
    print(f"📁 數據集路徑: {DATASET_DIR.absolute()}")
    
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    ) 