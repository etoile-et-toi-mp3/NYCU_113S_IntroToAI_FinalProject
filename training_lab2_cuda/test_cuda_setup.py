#!/usr/bin/env python3
"""
Cuda 設置測試腳本
用於驗證 Cuda 訓練環境是否正確配置
"""

import sys
import torch
import platform

def test_python_version():
    """測試 Python 版本"""
    print("🐍 Python 版本檢查...")
    version = sys.version_info
    print(f"   當前版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print("   ✅ Python 版本符合要求 (3.8+)")
        return True
    else:
        print("   ❌ Python 版本過低，建議使用 3.8+")
        return False

def test_system_info():
    """測試系統資訊與 CUDA 支援情況"""
    print("\n💻 系統資訊...")
    print(f"   作業系統: {platform.system()}")
    print(f"   版本: {platform.release()}")
    print(f"   架構: {platform.machine()}")

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        device_count = torch.cuda.device_count()
        print(f"   ✅ 檢測到 CUDA 支援: {device_name}")
        print(f"   ✅ CUDA 版本: {cuda_version}")
        print(f"   ✅ 可用 GPU 數量: {device_count}")
        return True
    else:
        print("   ⚠️ 未檢測到 CUDA, 將使用 CPU 執行")
        return False


def test_memory():
    """測試記憶體"""
    print("\n🧠 記憶體檢查...")
    try:
        import psutil
        memory = psutil.virtual_memory()
        total_gb = memory.total / (1024**3)
        available_gb = memory.available / (1024**3)
        
        print(f"   總記憶體: {total_gb:.1f} GB")
        print(f"   可用記憶體: {available_gb:.1f} GB")
        
        if total_gb >= 8:
            print("   ✅ 記憶體充足")
            return True
        elif total_gb >= 4:
            print("   ⚠️ 記憶體較少，建議使用最小配置")
            return True
        else:
            print("   ❌ 記憶體不足，可能影響訓練")
            return False
    except ImportError:
        print("   ❌ 無法檢查記憶體 (缺少 psutil)")
        return False

def test_vram():
    """測試 CUDA VRAM"""
    print("\n🧠 VRAM 檢查...")
    try:
        import torch
        if not torch.cuda.is_available():
            print("   ❌ 未偵測到可用的 CUDA GPU")
            return False

        device = torch.cuda.get_device_properties(0)
        total_vram_gb = device.total_memory / (1024 ** 3)
        device_name = device.name

        print(f"   GPU 裝置: {device_name}")
        print(f"   總 VRAM: {total_vram_gb:.1f} GB")

        if total_vram_gb >= 16:
            print("   ✅ VRAM 充足，適用高效能配置")
            return True
        elif total_vram_gb >= 6:
            print("   ⚠️ VRAM 一般，建議使用平衡配置")
            return True
        else:
            print("   ⚠️ VRAM 偏低，建議使用最小配置")
            return True
    except Exception as e:
        print(f"   ❌ 無法檢查 VRAM: {e}")
        return False

def test_pytorch():
    """測試 PyTorch 與 CUDA"""
    print("\n🔥 PyTorch 檢查...")
    try:
        import torch
        print(f"   PyTorch 版本: {torch.__version__}")
        
        # 檢查 CUDA 支援
        if torch.cuda.is_available():
            device = torch.device("cuda")
            device_name = torch.cuda.get_device_name(0)
            print(f"   ✅ CUDA 可用: {device_name}")
            
            # 測試 CUDA 設備張量操作
            try:
                test_tensor = torch.randn(10, 10, device=device)
                result = torch.matmul(test_tensor, test_tensor.T)
                print("   ✅ CUDA 設備測試成功")
                return True
            except Exception as e:
                print(f"   ❌ CUDA 設備測試失敗: {e}")
                return False
        else:
            print("   ⚠️ CUDA 不可用，將使用 CPU 執行")
            return True  # PyTorch still usable on CPU

    except ImportError:
        print("   ❌ PyTorch 未安裝")
        return False


def test_dependencies():
    """測試依賴套件"""
    print("\n📦 依賴套件檢查...")
    
    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'PIL': 'Pillow',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'sklearn': 'Scikit-learn',
        'psutil': 'PSUtil',
        'matplotlib': 'Matplotlib'
    }
    
    missing_packages = []
    
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} (缺少)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n   缺少的套件: {', '.join(missing_packages)}")
        print("   安裝命令:")
        
        # 特殊處理一些套件名稱
        install_names = []
        for pkg in missing_packages:
            if pkg == 'cv2':
                install_names.append('opencv-python')
            elif pkg == 'PIL':
                install_names.append('pillow')
            elif pkg == 'sklearn':
                install_names.append('scikit-learn')
            else:
                install_names.append(pkg)
        
        print(f"   pip install {' '.join(install_names)}")
        return False
    
    return True

def test_model_creation():
    """測試模型創建與 CUDA 設備轉移"""
    print("\n🤖 模型創建測試...")
    try:
        import torch
        import torch.nn as nn
        import torchvision.models as models

        # 嘗試創建 MobileNetV3 模型
        try:
            model = models.mobilenet_v3_large(weights=None)
            print("   ✅ MobileNetV3 可用")
        except:
            # 若失敗則嘗試 ResNet18
            try:
                model = models.resnet18(weights=None)
                print("   ✅ ResNet18 可用")
            except Exception as e:
                print(f"   ❌ 模型創建失敗: {e}")
                return False

        # 選擇設備：CUDA > CPU
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"   ✅ CUDA 可用: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("   ⚠️ CUDA 不可用，使用 CPU")

        # 將模型轉移到設備
        model.to(device)
        print(f"   ✅ 模型成功轉移到 {device}")

        # 執行一次前向傳播以測試完整流程
        model.eval()
        dummy_input = torch.randn(1, 3, 224, 224, device=device)
        with torch.no_grad():
            output = model(dummy_input)
        print("   ✅ 模型前向傳播測試成功")

        return True

    except Exception as e:
        print(f"   ❌ 模型測試失敗: {e}")
        return False

def test_data_loading():
    """測試數據載入"""
    print("\n📁 數據載入測試...")
    try:
        from torch.utils.data import Dataset, DataLoader
        import torchvision.transforms as transforms
        from PIL import Image
        import numpy as np
        
        # 創建虛擬數據集
        class DummyDataset(Dataset):
            def __init__(self, size=10):
                self.size = size
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                # 創建虛擬圖片
                image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
                image = self.transform(image)
                return {'image': image, 'label': idx % 2}
        
        dataset = DummyDataset()
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
        
        # 測試一個批次
        for batch in dataloader:
            images = batch['image']
            labels = batch['label']
            print(f"   ✅ 數據載入成功，批次形狀: {images.shape}")
            break
        
        return True
        
    except Exception as e:
        print(f"   ❌ 數據載入測試失敗: {e}")
        return False

def main():
    """主測試函數"""
    print("🟩 Cuda 訓練環境檢查")
    print("=" * 50)
    
    tests = [
        ("Python 版本", test_python_version),
        ("系統資訊", test_system_info),
        ("記憶體", test_memory),
        ("顯存", test_vram),
        ("PyTorch", test_pytorch),
        ("依賴套件", test_dependencies),
        ("模型創建", test_model_creation),
        ("數據載入", test_data_loading)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   ❌ 測試 {test_name} 時發生錯誤: {e}")
            results.append((test_name, False))
    
    # 總結
    print("\n" + "=" * 50)
    print("📊 測試結果總結:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通過" if result else "❌ 失敗"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n通過率: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 所有測試通過！您的系統已準備好進行訓練。")
        print("\n建議的下一步:")
        print("1. 準備您的數據集")
        print("2. 運行: python3 run_cuda_training.py --config balanced --dry-run")
        print("3. 開始訓練: python3 run_cuda_training.py --config balanced")
    elif passed >= total * 0.7:
        print("\n⚠️ 大部分測試通過，但有一些問題需要解決。")
        print("請查看上面的錯誤訊息並修復問題。")
    else:
        print("\n❌ 多個測試失敗，請先解決依賴和配置問題。")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 