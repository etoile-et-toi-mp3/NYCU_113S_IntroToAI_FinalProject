# Cuda 訓練配置文件
import os

class CudaTrainingConfig:
    """Cuda 優化的訓練配置"""
    
    # 數據設置
    DATA_ROOT = "./dataset"
    MAX_SAMPLES_PER_CLASS = 300  # 限制每類樣本數以節省記憶體
    
    # 模型設置
    NUM_STYLES = 14
    NUM_GENDERS = 2
    FEATURE_DIM = 1024
    
    # 訓練參數
    BATCH_SIZE = 32  # 增加批次大小以利用 GPU
    LEARNING_RATE = 0.0001
    NUM_EPOCHS = 30
    WEIGHT_DECAY = 1e-4
    
    # 記憶體管理
    MAX_MEMORY_MB = 16384
    MEMORY_CHECK_INTERVAL = 50
    
    # 數據載入器設置（適用於 CUDA）
    NUM_WORKERS = 4  # 根據實際 CPU 核心調整
    PIN_MEMORY = True
    DROP_LAST = True
    PERSISTENT_WORKERS = True
    
    # 優化器設置
    OPTIMIZER_BETAS = (0.9, 0.999)
    SCHEDULER_ETA_MIN_RATIO = 0.1  # 最小學習率比例
    
    # 損失權重
    STYLE_LOSS_WEIGHT = 1.0
    GENDER_LOSS_WEIGHT = 0.2
    CONTRASTIVE_LOSS_WEIGHT = 0.05
    
    # 梯度裁剪
    MAX_GRAD_NORM = 1.0
    
    # 檢查點設置
    SAVE_CHECKPOINT_EVERY = 5  # 每5個epoch保存檢查點
    
    # 日誌設置
    LOG_INTERVAL = 10  # 每10個batch輸出一次日誌
    
    # 數據增強設置
    HORIZONTAL_FLIP_PROB = 0.3
    COLOR_JITTER_BRIGHTNESS = 0.1
    COLOR_JITTER_CONTRAST = 0.1
    
    # 對比學習設置
    CONTRASTIVE_TEMPERATURE = 0.07
    CONTRASTIVE_CHUNK_SIZE = 32
    
    # 文件路徑
    TRAINED_MODEL_FOLDER = "./trained_models"
    BEST_MODEL_PATH = TRAINED_MODEL_FOLDER + "/outfit_model_best_cuda.pth"
    FINAL_MODEL_PATH = TRAINED_MODEL_FOLDER + "/outfit_model_final_cuda.pth"
    CHECKPOINT_FOLDER = "./checkpoint_models"
    CHECKPOINT_PREFIX = CHECKPOINT_FOLDER + "/checkpoint_cuda_epoch_"
    INTERRUPT_PATH = CHECKPOINT_FOLDER + "/interrupted_checkpoint.pth"
    
    def get_device_specific_config(cls):
        """根據設備調整配置"""
        import torch
        import psutil
        
        def get_vram_gb():
            if torch.cuda.is_available():
                return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            return 0

        vram_gb = get_vram_gb()
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)

        print(f"🔍 檢測到 VRAM: {vram_gb:.1f} GB, RAM: {ram_gb:.1f} GB")

        if 16 <= vram_gb:
            print("🚀 自動選用高效能配置 (performance)")
            return QuickConfigs.performance()
        elif 6 <= vram_gb:
            print("⚖️ 自動選用平衡配置 (balanced)")
            return QuickConfigs.balanced()
        elif 0 < vram_gb:
            
            print("🪶 VRAM 不足，自動選用最小配置 (minimal)")
            return QuickConfigs.minimal()
        
        print("⚠️ CUDA 不可用，自動回退至 CPU 配置")
        return QuickConfigs.minimal()
    
    @classmethod
    def print_config(cls):
        """打印當前配置"""
        print("\n📋 當前訓練配置:")
        print(f"  批次大小: {cls.BATCH_SIZE}")
        print(f"  學習率: {cls.LEARNING_RATE}")
        print(f"  訓練輪數: {cls.NUM_EPOCHS}")
        print(f"  每類最大樣本數: {cls.MAX_SAMPLES_PER_CLASS}")
        print(f"  最大記憶體使用: {cls.MAX_MEMORY_MB}MB")
        print(f"  數據工作進程: {cls.NUM_WORKERS}")
        print(f"  特徵維度: {cls.FEATURE_DIM}")
        print()

class QuickConfigs:
    """快速配置預設（CUDA 版本）"""

    @staticmethod
    def minimal():
        """最小配置 - 適用於低記憶體 CUDA 設備或無 GPU 設備"""
        config = CudaTrainingConfig()
        config.BATCH_SIZE = 8
        config.MAX_SAMPLES_PER_CLASS = 200
        config.NUM_EPOCHS = 15
        config.MAX_MEMORY_MB = 4096
        config.NUM_WORKERS = 2
        config.FEATURE_DIM = 512 # adjust feature_dim
        config.PIN_MEMORY = False # turn off pin_memory
        config.PERSISTENT_WORKERS = False # turn off persistent_workers
        return config

    @staticmethod
    def balanced():
        """平衡配置 - 適用於大多數 CUDA 設備"""
        config = CudaTrainingConfig()
        config.BATCH_SIZE = 32
        config.MAX_SAMPLES_PER_CLASS = 500
        config.NUM_EPOCHS = 25
        config.MAX_MEMORY_MB = 8192
        config.NUM_WORKERS = 4
        return config

    @staticmethod
    def performance():
        """性能配置 - 適用於高端 GPU 系統"""
        config = CudaTrainingConfig()
        config.BATCH_SIZE = 64
        config.MAX_SAMPLES_PER_CLASS = 1000
        config.NUM_EPOCHS = 40
        config.MAX_MEMORY_MB = 16384
        config.NUM_WORKERS = 8
        return config

if __name__ == "__main__":
    # 測試配置
    config = CudaTrainingConfig.get_device_specific_config()
    config.print_config() 