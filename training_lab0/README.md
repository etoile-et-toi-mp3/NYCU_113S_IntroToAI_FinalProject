# 時尚AI系統 - 快速開始

## 🛠️ 安裝
```bash
pip install torch torchvision timm transformers scikit-learn matplotlib pillow numpy psutil
```

## usage:
```bash
# 訓練
python unsupervised_train.py --data ../dataset --config balanced --backbone resnet50

# 測試
python unsupervised_test.py --model unsupervised_result/[時間戳]/best_model_resnet50_balanced.pth --image test.jpg --labels unsupervised_result/[時間戳]/dataset_labels.json --backbone resnet50 --top-k 5
```

## ⚙️ 參數選項

**通用參數**:
- `--config`: `minimal`(省記憶體) | `balanced`(推薦) | `performance`(高效能)
- `--backbone`: `mobilenet` | `resnet18` | `resnet50` | `efficientnet_b0` | `efficientnet_b2` | `vit_tiny` | `vit_small` | `fashion_resnet`
- `--top-k`: 返回相似圖片數量 (預設10)
