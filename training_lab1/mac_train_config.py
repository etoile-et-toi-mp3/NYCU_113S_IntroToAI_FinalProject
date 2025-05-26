# Mac 訓練配置文件
import os

class MacTrainingConfig:
    """Mac 優化的訓練配置"""
    
    # 數據設置
    DATA_ROOT = "./dataset"
    MAX_SAMPLES_PER_CLASS = 300  # 限制每類樣本數以節省記憶體
    
    # 模型設置
    NUM_STYLES = 12
    NUM_GENDERS = 2
    FEATURE_DIM = 1024
    
    # 訓練參數
    BATCH_SIZE = 8  # Mac 適用的小批次大小
    LEARNING_RATE = 0.0001
    NUM_EPOCHS = 20
    WEIGHT_DECAY = 1e-4
    
    # 記憶體管理
    MAX_MEMORY_MB = 8000  # 最大記憶體使用量 (8GB)
    MEMORY_CHECK_INTERVAL = 50  # 每50個batch檢查一次記憶體
    
    # 數據載入器設置
    NUM_WORKERS = 0  # Mac 上設為0避免多進程問題
    PIN_MEMORY = False  # MPS 不需要 pin_memory
    DROP_LAST = True
    PERSISTENT_WORKERS = False
    
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
    BEST_MODEL_PATH = "outfit_model_best_mac.pth"
    FINAL_MODEL_PATH = "outfit_model_final_mac.pth"
    CHECKPOINT_PREFIX = "checkpoint_mac_epoch_"
    
    @classmethod
    def get_device_specific_config(cls):
        """根據設備調整配置"""
        import torch
        import psutil
        
        # 獲取系統記憶體
        total_memory_gb = psutil.virtual_memory().total / (1024**3)
        
        if total_memory_gb < 8:
            # 低記憶體設備
            cls.BATCH_SIZE = 4
            cls.MAX_SAMPLES_PER_CLASS = 200
            cls.MAX_MEMORY_MB = 4000
            print("⚠️  檢測到低記憶體設備，調整配置")
        elif total_memory_gb > 16:
            # 高記憶體設備
            cls.BATCH_SIZE = 12
            cls.MAX_SAMPLES_PER_CLASS = 500
            cls.MAX_MEMORY_MB = 12000
            print("✅ 檢測到高記憶體設備，提升配置")
        
        # 檢查MPS可用性
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("✅ 使用 Metal Performance Shaders (MPS)")
        else:
            # CPU模式下進一步降低配置
            cls.BATCH_SIZE = max(2, cls.BATCH_SIZE // 2)
            cls.MAX_SAMPLES_PER_CLASS = min(150, cls.MAX_SAMPLES_PER_CLASS)
            print("⚠️  使用 CPU 模式，進一步降低配置")
        
        return cls
    
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

# 快速配置預設
class QuickConfigs:
    """快速配置預設"""
    
    @staticmethod
    def minimal():
        """最小配置 - 適用於記憶體很少的設備"""
        config = MacTrainingConfig()
        config.BATCH_SIZE = 2
        config.MAX_SAMPLES_PER_CLASS = 100
        config.NUM_EPOCHS = 10
        config.FEATURE_DIM = 512
        config.MAX_MEMORY_MB = 2000
        return config
    
    @staticmethod
    def balanced():
        """平衡配置 - 適用於大多數Mac設備"""
        config = MacTrainingConfig()
        config.BATCH_SIZE = 8
        config.MAX_SAMPLES_PER_CLASS = 300
        config.NUM_EPOCHS = 20
        config.FEATURE_DIM = 1024
        config.MAX_MEMORY_MB = 8000
        return config
    
    @staticmethod
    def performance():
        """性能配置 - 適用於高端Mac設備"""
        config = MacTrainingConfig()
        config.BATCH_SIZE = 16
        config.MAX_SAMPLES_PER_CLASS = 500
        config.NUM_EPOCHS = 30
        config.FEATURE_DIM = 1024
        config.MAX_MEMORY_MB = 12000
        return config

if __name__ == "__main__":
    # 測試配置
    config = MacTrainingConfig.get_device_specific_config()
    config.print_config() 