# Cuda 優化的服裝風格分類訓練

這是一個專為 Cuda 顯卡優化的服裝風格分類深度學習訓練框架，支援 GPU 加速。

## 🚀 快速開始

### 1. 安裝依賴

```bash
pip install torch torchvision torchaudio
pip install opencv-python scikit-learn psutil pillow numpy matplotlib
```

### 2. 準備數據集

確保您的數據集目錄結構如下：
```
dataset/
├── Artsy_MEN/
├── Artsy_WOMEN/
├── Athleisure_MEN/
├── Athleisure_WOMEN/
├── BRITISH_MEN/
├── BRITISH_WOMEN/
├── CASUAL_MEN/
├── CASUAL_WOMEN/
├── GOTH_MEN/
├── GOTH_WOMEN/
├── Kawaii_MEN/
├── Kawaii_WOMEN/
├── Korean_MEN/
├── Korean_WOMEN/
├── MINIMALIST_MEN/
├── MINIMALIST_WOMEN/
├── Preppy_MEN/
├── Preppy_WOMEN/
├── STREET_MEN/
├── STREET_WOMEN/
├── Streetwear_MEN/
├── Streetwear_WOMEN/
├── Vintage_MEN/
├── Vintage_WOMEN/
├── Y2K_MEN/
└── Y2K_WOMEN/
```

### 3. 開始訓練

#### 平衡配置（推薦）
```bash
python3 run_cuda_training.py --config balanced
```

#### 最小配置（低記憶體設備）
```bash
python3 run_cuda_training.py --config minimal
```

#### 性能配置（高端設備）
```bash
python3 run_cuda_training.py --config performance
```

#### 自定義數據集路徑
```bash
python3 run_cuda_training.py --config balanced --data-root /path/to/your/dataset
```

#### 從檢查點恢復訓練
```bash
python3 run_cuda_training.py --config balanced --resume ./checkpoint/checkpoint_cuda_epoch_10.pth
```

#### 檢查配置（不實際訓練）
```bash
python3 run_cuda_training.py --config balanced --dry-run
```

## ⚙️ 配置說明

### 配置預設

| 配置 | 適用設備 | 批次大小 | 每類樣本數 | 訓練輪數 | 記憶體使用 |
|------|----------|----------|------------|----------|------------|
| minimal | 低顯存顯卡或無顯卡 | 8 | 200 | 15 | ~4GB |
| balanced | 一般顯卡 | 32 | 500 | 25 | ~8GB |
| performance | 高端顯卡 | 64 | 1000 | 40 | ~16GB |

### 自動調整

若不提供 `--config` 選項，系統會根據您的系統配置自動調整參數：
- **記憶體檢測**: 根據系統記憶體調整批次大小
- **GPU 檢測**: 自動使用 Nvidia GPU 加速
- **CPU 回退**: GPU 不可用時自動使用 CPU

## 🔧 GPU 特定優化

### Nvidia GPU Acceleration
- 自動檢測並使用 Nvidia GPU 加速
- 記憶體管理優化

### 記憶體管理
- 動態記憶體監控
- 自動垃圾回收
- 批次大小自適應

### 輕量化模型
- 使用 MobileNetV3 或 ResNet18 作為骨幹網路
- 減少參數數量
- 保持性能的同時降低記憶體使用

## 📊 訓練監控

訓練過程中會顯示：
- 實時損失值
- 記憶體使用量
- 訓練速度
- 學習率變化

## 💾 模型保存

訓練會自動保存：
- `./trained_models/outfit_model_best_cuda.pth`: 最佳模型（基於損失）
- `./trained_models/outfit_model_final_cuda.pth`: 最終模型
- `./checkpoint_models/checkpoint_cuda_epoch_*.pth`: 每5個epoch的檢查點
- `./checkpoint_models/interrupted_checkpoint.pth`: 中斷時的檢查點

## 🛠️ 故障排除

### 常見問題

#### 1. CUDA 不可用
```
⚠️ CUDA 不可用，使用 CPU
```
**解決方案**: 確保使用支援的 GPU 設備

#### 2. 記憶體不足
```
⚠️ 記憶體使用過高，執行垃圾回收
```
**解決方案**: 使用 `--config minimal` 或減少批次大小

#### 3. 數據集未找到
```
❌ 數據集目錄不存在
```
**解決方案**: 檢查數據集路徑和目錄結構

#### 4. 依賴缺失
```
❌ 缺少必要的套件
```
**解決方案**: 安裝缺失的套件 `pip install package_name`

### 性能調優

#### 低顯存設備 (6GB 以下)
```bash
python3 run_cuda_training.py --config minimal
```

#### 高記憶體設備 (16GB 以上)
```bash
python3 run_cuda_training.py --config performance
```

#### 自定義配置
修改 `cuda_train_config.py` 中的參數：
```python
class CudaTrainingConfig:
    BATCH_SIZE = 4  # 調整批次大小
    MAX_SAMPLES_PER_CLASS = 200  # 調整樣本數
    NUM_EPOCHS = 15  # 調整訓練輪數
```

## 📈 模型使用

訓練完成後，可以使用模型進行推論：

```python
from test_cuda_optimized import LightweightStyleClassifier
import torch

# 載入模型
model = LightweightStyleClassifier()
model.load_state_dict(torch.load('./trained_models/outfit_model_best_cuda.pth'))
model.eval()

# 進行推論
# ... (推論代碼)
```

## 🤝 支援

如果遇到問題，請檢查：
1. Python 版本 (建議 3.8+)
2. PyTorch 版本 (建議 2.0+)
3. 可用記憶體空間

## 📝 更新日誌

- **v1.0**: 初始版本，支援 MPS 加速
- **v1.1**: 添加記憶體優化和自動配置調整
- **v1.2**: 改進輕量化模型和訓練穩定性
- **v1.3**: 新增 Cuda 加速版本
