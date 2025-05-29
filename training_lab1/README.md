# 時尚AI系統 - 使用指南

有上標籤(用資料夾分類)的部分
- `supervised_train.py` - 訓練時尚風格分類模型
- `supervised_test.py` - 測試模型並進行穿搭分析推薦

## 🛠️ pip install

```bash
pip install torch torchvision torchaudio
pip install timm transformers
pip install scikit-learn matplotlib
pip install pillow numpy psutil
```

## 目前dataset的風格
支援的風格：`Athleisure`, `BRITISH`, `CASUAL`, `GOTH`, `Kawaii`, `Korean`, `MINIMALIST`, `Preppy`, `STREET`, `Streetwear`, `Vintage`, `Y2K`

## usage:

### 1. 訓練模型

基本訓練：
```bash
python supervised_train.py --data ../dataset
```

選擇不同配置：
```bash
# 最小配置（記憶體有限）
python supervised_train.py --config minimal --backbone mobilenet

# 平衡配置（推薦）
python supervised_train.py --config balanced --backbone resnet18

# 性能配置（高性能Mac）
python supervised_train.py --config performance --backbone efficientnet_b0
```

### 2. 測試模型

```bash
python supervised_test.py --model result/20231201120000/best_model_mobilenet_balanced.pth --image test_image.jpg
```

## 📊 輸出結果

- 訓練輸出：`result/[時間戳]/` 目錄
  - 模型檔案：`best_model_*.pth`
  - 訓練日誌：`training_log.txt`
  - 訓練圖表：`training_history.png`

- 測試輸出：`recommendations/` 目錄
  - 分析報告：`recommendation_report_*.json`

## ⚙️ 參數說明

### 訓練參數
- `--config`: 配置類型 (`minimal`/`balanced`/`performance`)
- `--backbone`: 網路架構 (`mobilenet`/`resnet18`/`resnet50`/`efficientnet_b0`/`efficientnet_b2`/`vit_tiny`/`vit_small`/`fashion_resnet`)
- `--data`: 數據集路徑

### 測試參數
- `--model`: 訓練好的模型路徑
- `--image`: 要分析的圖片路徑
- `--backbone`: 與訓練時相同的網路架構
- `--top-k`: 返回最相似的圖片數量