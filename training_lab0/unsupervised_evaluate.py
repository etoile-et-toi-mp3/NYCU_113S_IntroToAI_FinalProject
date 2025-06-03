#!/usr/bin/env python3
"""
時尚AI無監督學習批量評估系統
正確的準確率計算：預測正確的圖片數量 / (220 × 5) = 正確數 / 1100
"""

import subprocess
import json
import os
import argparse
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import csv

def load_test_data_with_labels(testdata_dir):
    """載入測試數據並生成正確答案"""
    print(f"📂 載入測試數據: {testdata_dir}")
    
    image_files = []
    for file in os.listdir(testdata_dir):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_files.append(file)
    
    image_files.sort()
    print(f"📊 找到 {len(image_files)} 張測試圖片")
    
    test_data = []
    categories = set()
    
    for img_file in image_files:
        img_path = os.path.join(testdata_dir, img_file)
        filename_without_ext = os.path.splitext(img_file)[0]
        
        # 提取類別名稱
        category = None
        for i in range(len(filename_without_ext), 0, -1):
            if not filename_without_ext[i-1:].isdigit():
                category = filename_without_ext[:i]
                break
        
        if category is None:
            category = filename_without_ext
        
        categories.add(category)
        test_data.append({
            'image_path': img_path,
            'filename': img_file,
            'category': category
        })
    
    categories = sorted(list(categories))
    print(f"📋 發現 {len(categories)} 個類別: {categories}")
    
    # 統計每個類別的圖片數量
    category_counts = defaultdict(int)
    for item in test_data:
        category_counts[item['category']] += 1
    
    print("📈 各類別圖片數量:")
    for category in categories:
        print(f"  {category}: {category_counts[category]} 張")
    
    return test_data, categories

def run_unsupervised_test(model_path, image_path, labels_path, backbone_type, top_k, output_dir):
    """調用 unsupervised_test.py 進行單張圖片測試"""
    cmd = [
        'python', 'unsupervised_test.py',
        '--model', model_path,
        '--image', image_path,
        '--labels', labels_path,
        '--backbone', backbone_type,
        '--top-k', str(top_k),
        '--output-dir', output_dir
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            return True, result.stdout, result.stderr
        else:
            return False, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Timeout"
    except Exception as e:
        return False, "", str(e)

def find_latest_report(output_dir):
    """找到最新生成的推薦報告"""
    report_files = []
    for file in os.listdir(output_dir):
        if file.startswith('unsupervised_recommendation_report_') and file.endswith('.json'):
            report_files.append(file)
    
    if not report_files:
        return None
    
    report_files.sort(reverse=True)
    return os.path.join(output_dir, report_files[0])

def find_model_files(models_dir, backbones):
    """自動找到各backbone的模型檔案和特徵數據庫"""
    model_configs = {}
    
    print(f"📁 在目錄中尋找模型和特徵檔案: {models_dir}")
    
    all_files = os.listdir(models_dir)
    
    for backbone in backbones:
        model_file = None
        labels_file = None
        
        # 尋找模型檔案
        for file in all_files:
            if f"best_model_{backbone}_" in file and file.endswith('.pth'):
                model_file = os.path.join(models_dir, file)
                break
        
        # 尋找對應的特徵數據庫檔案
        possible_labels_files = [
            f"{backbone}_dataset_labels.json",
            f"dataset_labels_{backbone}.json",
            "dataset_labels.json"
        ]
        
        for labels_filename in possible_labels_files:
            if labels_filename in all_files:
                labels_file = os.path.join(models_dir, labels_filename)
                break
        
        if model_file and labels_file:
            model_configs[backbone] = {
                'model_path': model_file,
                'labels_path': labels_file
            }
            print(f"  ✅ {backbone}:")
            print(f"     模型: {os.path.basename(model_file)}")
            print(f"     特徵: {os.path.basename(labels_file)}")
        else:
            missing = []
            if not model_file:
                missing.append("模型檔案")
            if not labels_file:
                missing.append("特徵檔案")
            print(f"  ❌ {backbone}: 找不到 {', '.join(missing)}")
    
    return model_configs

def evaluate_single_backbone(backbone_type, model_path, labels_path, test_data, top_k, temp_output_dir):
    """評估單個backbone - 使用正確的準確率計算方式"""
    print(f"\n{'='*60}")
    print(f"🔧 評估 Backbone: {backbone_type}")
    print(f"📂 模型: {os.path.basename(model_path)}")
    print(f"📁 特徵庫: {os.path.basename(labels_path)}")
    print(f"{'='*60}")
    
    if not os.path.exists(model_path):
        print(f"❌ 模型檔案不存在: {model_path}")
        return None
    
    if not os.path.exists(labels_path):
        print(f"❌ 特徵數據庫不存在: {labels_path}")
        return None
    
    results = {
        'backbone': backbone_type,
        'model_path': model_path,
        'labels_path': labels_path,
        'total_test_images': len(test_data),
        'total_predictions': 0,  # 總預測次數 (應該是 test_images * top_k)
        'correct_predictions': 0,  # 正確預測次數
        'failed_tests': 0,
        'category_results': defaultdict(lambda: {
            'total_predictions': 0,  # 該類別的總預測次數
            'correct_predictions': 0,  # 該類別的正確預測次數
            'test_images': 0  # 該類別的測試圖片數
        }),
        'detailed_results': []
    }
    
    # 為每張測試圖片進行預測
    for i, test_item in enumerate(test_data):
        print(f"  測試 {i+1}/{len(test_data)}: {test_item['filename']}")
        
        # 調用 unsupervised_test.py
        success, stdout, stderr = run_unsupervised_test(
            model_path, test_item['image_path'], labels_path, 
            backbone_type, top_k, temp_output_dir
        )
        
        if not success:
            print(f"    ❌ 測試失敗: {stderr}")
            results['failed_tests'] += 1
            continue
        
        # 讀取生成的報告
        report_path = find_latest_report(temp_output_dir)
        if not report_path:
            print(f"    ❌ 找不到報告檔案")
            results['failed_tests'] += 1
            continue
        
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                report = json.load(f)
            
            # 分析結果
            similar_images = report.get('similar_images', [])
            test_category = test_item['category']
            
            # 新的計算方式：計算Top-K中每張圖片
            similar_categories = []
            correct_count = 0  # 這張測試圖片貢獻的正確預測數
            
            # 確保我們處理完整的top_k個結果
            processed_count = 0
            for sim_img in similar_images:
                if processed_count >= top_k:
                    break
                    
                # 從路徑中提取類別
                img_path = sim_img.get('image_path', '')
                path_parts = img_path.split('/')
                if len(path_parts) >= 2:
                    folder_name = path_parts[-2]  # 資料夾名稱就是類別
                    similar_categories.append(folder_name)
                    
                    # 每張相似圖片都是一次預測
                    results['total_predictions'] += 1
                    results['category_results'][test_category]['total_predictions'] += 1
                    processed_count += 1
                    
                    # 如果預測正確
                    if folder_name == test_category:
                        correct_count += 1
                        results['correct_predictions'] += 1
                        results['category_results'][test_category]['correct_predictions'] += 1
            
            # 如果返回的相似圖片少於top_k，補足計數
            if processed_count < top_k:
                missing_predictions = top_k - processed_count
                results['total_predictions'] += missing_predictions
                results['category_results'][test_category]['total_predictions'] += missing_predictions
                print(f"    ⚠️ 只返回了 {processed_count}/{top_k} 張相似圖片")
            
            # 統計測試圖片數
            results['category_results'][test_category]['test_images'] += 1
            
            # 顯示結果
            if correct_count > 0:
                print(f"    ✅ {correct_count}/{top_k} 正確 (Top-{top_k}: {similar_categories})")
            else:
                print(f"    ❌ 0/{top_k} 正確 (Top-{top_k}: {similar_categories})")
            
            # 詳細結果
            results['detailed_results'].append({
                'test_image': test_item['filename'],
                'test_category': test_category,
                'similar_categories': similar_categories,
                'correct_count': correct_count,
                'total_predictions': len(similar_categories),
                'similarities': [img.get('similarity_score', 0) for img in similar_images]
            })
            
            # 清理臨時報告
            os.remove(report_path)
            
        except Exception as e:
            print(f"    ❌ 解析報告失敗: {e}")
            results['failed_tests'] += 1
    
    # 計算準確率
    if results['total_predictions'] > 0:
        results['accuracy'] = results['correct_predictions'] / results['total_predictions']
    else:
        results['accuracy'] = 0.0
    
    # 計算各類別準確率
    results['category_accuracies'] = {}
    for category, stats in results['category_results'].items():
        if stats['total_predictions'] > 0:
            results['category_accuracies'][category] = stats['correct_predictions'] / stats['total_predictions']
        else:
            results['category_accuracies'][category] = 0.0
    
    print(f"🎯 {backbone_type} 評估完成:")
    print(f"  總準確率: {results['accuracy']:.4f} ({results['correct_predictions']}/{results['total_predictions']})")
    print(f"  測試圖片: {results['total_test_images']}, 失敗: {results['failed_tests']}")
    
    return results

def generate_comprehensive_csv_reports_and_visualizations(all_results, output_dir, top_k):
    """生成詳細的CSV報告和視覺化圖表 - 使用正確的計算方式"""
    print(f"\n📊 生成CSV報告和視覺化...")
    
    # 過濾有效結果
    valid_results = [r for r in all_results if r is not None]
    if not valid_results:
        print("❌ 沒有有效的評估結果")
        return None
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 收集所有類別
    all_categories = set()
    for result in valid_results:
        all_categories.update(result['category_accuracies'].keys())
    all_categories = sorted(list(all_categories))
    
    print("📝 生成詳細統計報告...")
    
    # === 1. 詳細預測結果 CSV ===
    detailed_data = []
    for result in valid_results:
        for category in all_categories:
            if category in result['category_results']:
                stats = result['category_results'][category]
                accuracy = result['category_accuracies'].get(category, 0.0)
                
                detailed_data.append({
                    'Backbone': result['backbone'],
                    'Category': category,
                    'Test_Images': stats['test_images'],
                    'Total_Predictions': stats['total_predictions'],
                    'Correct_Predictions': stats['correct_predictions'],
                    'Accuracy': accuracy,
                    'Percentage': f"{accuracy*100:.2f}%",
                    'Expected_Predictions': stats['test_images'] * top_k,
                    'Prediction_Rate': stats['total_predictions'] / (stats['test_images'] * top_k) if stats['test_images'] > 0 else 0
                })
    
    detailed_df = pd.DataFrame(detailed_data)
    detailed_csv_path = os.path.join(output_dir, f'detailed_predictions_{timestamp}.csv')
    detailed_df.to_csv(detailed_csv_path, index=False, encoding='utf-8-sig')
    print(f"📄 詳細預測結果: {detailed_csv_path}")
    
    # === 2. Backbone整體統計 CSV ===
    backbone_summary = []
    for result in valid_results:
        expected_predictions = (result['total_test_images'] - result['failed_tests']) * top_k
        
        backbone_summary.append({
            'Backbone': result['backbone'],
            'Overall_Accuracy': result['accuracy'],
            'Correct_Predictions': result['correct_predictions'],
            'Total_Predictions': result['total_predictions'],
            'Expected_Predictions': expected_predictions,
            'Test_Images': result['total_test_images'],
            'Failed_Tests': result['failed_tests'],
            'Success_Rate': result['total_predictions'] / expected_predictions if expected_predictions > 0 else 0,
            'Percentage': f"{result['accuracy']*100:.2f}%"
        })
    
    backbone_summary.sort(key=lambda x: x['Overall_Accuracy'], reverse=True)
    backbone_df = pd.DataFrame(backbone_summary)
    backbone_csv_path = os.path.join(output_dir, f'backbone_summary_{timestamp}.csv')
    backbone_df.to_csv(backbone_csv_path, index=False, encoding='utf-8-sig')
    print(f"📄 Backbone整體統計: {backbone_csv_path}")
    
    # === 3. 類別整體統計 CSV ===
    category_summary = []
    for category in all_categories:
        total_correct = 0
        total_predictions = 0
        total_test_images = 0
        backbone_accuracies = []
        
        for result in valid_results:
            if category in result['category_results']:
                stats = result['category_results'][category]
                total_correct += stats['correct_predictions']
                total_predictions += stats['total_predictions']
                total_test_images += stats['test_images']
                
                if stats['total_predictions'] > 0:
                    backbone_accuracies.append(stats['correct_predictions'] / stats['total_predictions'])
        
        overall_accuracy = total_correct / total_predictions if total_predictions > 0 else 0.0
        avg_backbone_accuracy = np.mean(backbone_accuracies) if backbone_accuracies else 0.0
        std_backbone_accuracy = np.std(backbone_accuracies) if len(backbone_accuracies) > 1 else 0.0
        expected_predictions = total_test_images * top_k
        
        category_summary.append({
            'Category': category,
            'Overall_Accuracy': overall_accuracy,
            'Total_Correct': total_correct,
            'Total_Predictions': total_predictions,
            'Expected_Predictions': expected_predictions,
            'Test_Images': total_test_images,
            'Avg_Backbone_Accuracy': avg_backbone_accuracy,
            'Std_Backbone_Accuracy': std_backbone_accuracy,
            'Tested_Backbones': len(backbone_accuracies),
            'Percentage': f"{overall_accuracy*100:.2f}%",
            'Prediction_Rate': total_predictions / expected_predictions if expected_predictions > 0 else 0
        })
    
    category_summary.sort(key=lambda x: x['Overall_Accuracy'], reverse=True)
    category_df = pd.DataFrame(category_summary)
    category_csv_path = os.path.join(output_dir, f'category_summary_{timestamp}.csv')
    category_df.to_csv(category_csv_path, index=False, encoding='utf-8-sig')
    print(f"📄 類別整體統計: {category_csv_path}")
    
    # === 4. 生成視覺化圖表 ===
    print("🎨 生成視覺化圖表...")
    
    # 設置圖表樣式
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 圖表1: 綜合評估結果 (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle(f'Unsupervised Learning Evaluation Results (Top-{top_k})\n新計算方式: 正確預測數 / 總預測數 (總預測數 = 圖片數 × {top_k})', 
                 fontsize=16, fontweight='bold')
    
    # 子圖1: Backbone準確率比較
    backbones = [item['Backbone'] for item in backbone_summary]
    accuracies = [item['Overall_Accuracy'] for item in backbone_summary]
    
    bars1 = axes[0,0].bar(backbones, accuracies, color='steelblue', alpha=0.8, edgecolor='navy', linewidth=1)
    axes[0,0].set_title('Backbone Overall Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('Backbone Architecture', fontsize=12)
    axes[0,0].set_ylabel('Accuracy (Correct/Total Predictions)', fontsize=12)
    axes[0,0].set_ylim(0, max(accuracies) * 1.1 if accuracies else 1)
    axes[0,0].tick_params(axis='x', rotation=45)
    axes[0,0].grid(True, alpha=0.3)
    
    for bar, acc in zip(bars1, accuracies):
        axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(accuracies)*0.01, 
                      f'{acc:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 子圖2: 類別準確率分布 (前15名)
    top_categories = category_summary[:15]
    categories = [item['Category'] for item in top_categories]
    cat_accuracies = [item['Overall_Accuracy'] for item in top_categories]
    
    bars2 = axes[0,1].bar(categories, cat_accuracies, color='forestgreen', alpha=0.8, edgecolor='darkgreen', linewidth=1)
    axes[0,1].set_title('Category Overall Accuracy (Top 15)', fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('Category', fontsize=12)
    axes[0,1].set_ylabel('Accuracy', fontsize=12)
    axes[0,1].set_ylim(0, max(cat_accuracies) * 1.1 if cat_accuracies else 1)
    axes[0,1].tick_params(axis='x', rotation=45)
    axes[0,1].grid(True, alpha=0.3)
    
    for bar, acc in zip(bars2, cat_accuracies):
        axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(cat_accuracies)*0.01, 
                      f'{acc:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 子圖3: 預測數量統計
    prediction_data = []
    for item in backbone_summary:
        prediction_data.append({
            'Backbone': item['Backbone'],
            'Correct': item['Correct_Predictions'],
            'Incorrect': item['Total_Predictions'] - item['Correct_Predictions']
        })
    
    pred_df = pd.DataFrame(prediction_data)
    pred_df.set_index('Backbone')[['Correct', 'Incorrect']].plot(kind='bar', stacked=True, ax=axes[1,0], 
                                                                  color=['lightgreen', 'lightcoral'])
    axes[1,0].set_title('Prediction Count Breakdown by Backbone', fontsize=14, fontweight='bold')
    axes[1,0].set_xlabel('Backbone', fontsize=12)
    axes[1,0].set_ylabel('Number of Predictions', fontsize=12)
    axes[1,0].tick_params(axis='x', rotation=45)
    axes[1,0].legend(['Correct', 'Incorrect'])
    axes[1,0].grid(True, alpha=0.3)
    
    # 子圖4: 準確率vs預測率散點圖
    accuracy_vs_rate = []
    for result in valid_results:
        for category in all_categories:
            if category in result['category_results']:
                stats = result['category_results'][category]
                if stats['test_images'] > 0:
                    accuracy = stats['correct_predictions'] / stats['total_predictions'] if stats['total_predictions'] > 0 else 0
                    prediction_rate = stats['total_predictions'] / (stats['test_images'] * top_k)
                    accuracy_vs_rate.append({
                        'Accuracy': accuracy,
                        'Prediction_Rate': prediction_rate,
                        'Backbone': result['backbone'],
                        'Category': category
                    })
    
    if accuracy_vs_rate:
        rate_df = pd.DataFrame(accuracy_vs_rate)
        for backbone in backbones:
            backbone_data = rate_df[rate_df['Backbone'] == backbone]
            axes[1,1].scatter(backbone_data['Prediction_Rate'], backbone_data['Accuracy'], 
                            label=backbone, alpha=0.7, s=50)
        
        axes[1,1].set_title('Accuracy vs Prediction Rate', fontsize=14, fontweight='bold')
        axes[1,1].set_xlabel('Prediction Rate (Actual/Expected Predictions)', fontsize=12)
        axes[1,1].set_ylabel('Accuracy', fontsize=12)
        axes[1,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path1 = os.path.join(output_dir, f'comprehensive_evaluation_{timestamp}.png')
    plt.savefig(plot_path1, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 圖表2: 詳細熱力圖分析
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    fig.suptitle(f'Detailed Accuracy Analysis (Top-{top_k})', fontsize=16, fontweight='bold')
    
    # 左圖: Backbone vs Category 熱力圖
    heatmap_data = detailed_df.pivot(index='Category', columns='Backbone', values='Accuracy')
    
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                ax=ax1, cbar_kws={'label': 'Accuracy'}, linewidths=0.5)
    ax1.set_title('Backbone vs Category Accuracy Heatmap', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Backbone', fontsize=12)
    ax1.set_ylabel('Category', fontsize=12)
    
    # 右圖: 預測數量熱力圖
    prediction_heatmap = detailed_df.pivot(index='Category', columns='Backbone', values='Total_Predictions')
    
    sns.heatmap(prediction_heatmap, annot=True, fmt='d', cmap='Blues', 
                ax=ax2, cbar_kws={'label': 'Total Predictions'}, linewidths=0.5)
    ax2.set_title('Total Predictions Count Heatmap', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Backbone', fontsize=12)
    ax2.set_ylabel('Category', fontsize=12)
    
    plt.tight_layout()
    plot_path2 = os.path.join(output_dir, f'detailed_heatmap_analysis_{timestamp}.png')
    plt.savefig(plot_path2, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"🎨 視覺化圖表已生成:")
    print(f"  📊 綜合評估結果: {plot_path1}")
    print(f"  🔥 詳細熱力圖分析: {plot_path2}")
    
    return {
        'detailed_csv': detailed_csv_path,
        'backbone_csv': backbone_csv_path,
        'category_csv': category_csv_path,
        'plot1': plot_path1,
        'plot2': plot_path2,
        'backbone_summary': backbone_summary,
        'category_summary': category_summary
    }

def create_comprehensive_log(all_results, output_dir, top_k):
    """創建詳細的日誌文件 - 使用正確的計算方式"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(output_dir, f'evaluation_log_{timestamp}.txt')
    
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("無監督學習 Backbone 評估詳細日誌\n")
        f.write("正確計算方式: 準確率 = 正確預測數 / 總預測數\n")
        f.write(f"總預測數 = 測試圖片數 × Top-{top_k}\n")
        f.write("=" * 80 + "\n")
        f.write(f"評估時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Top-K 設定: {top_k}\n")
        f.write(f"評估的 Backbone 數量: {len([r for r in all_results if r is not None])}\n")
        f.write("\n")
        
        valid_results = [r for r in all_results if r is not None]
        
        if not valid_results:
            f.write("❌ 沒有有效的評估結果\n")
            return log_path
        
        # 1. Backbone 整體準確率排名
        f.write("🏆 BACKBONE 整體準確率排名:\n")
        f.write("-" * 80 + "\n")
        f.write("   排名  Backbone               準確率    正確預測/總預測     測試圖片  失敗\n")
        f.write("-" * 80 + "\n")
        
        backbone_summary = []
        for result in valid_results:
            backbone_summary.append({
                'backbone': result['backbone'],
                'accuracy': result['accuracy'],
                'correct': result['correct_predictions'],
                'total': result['total_predictions'],
                'test_images': result['total_test_images'],
                'failed': result['failed_tests']
            })
        
        backbone_summary.sort(key=lambda x: x['accuracy'], reverse=True)
        
        for i, item in enumerate(backbone_summary, 1):
            f.write(f"   {i:2d}.   {item['backbone']:20s}   {item['accuracy']:7.4f}   "
                   f"{item['correct']:4d}/{item['total']:4d}        {item['test_images']:3d}      {item['failed']:2d}\n")
        
        f.write("\n")
        
        # 2. 類別整體準確率排名
        f.write("📊 類別整體準確率排名:\n")
        f.write("-" * 80 + "\n")
        f.write("   排名  類別                   準確率    正確預測/總預測     測試圖片\n")
        f.write("-" * 80 + "\n")
        
        # 收集所有類別
        all_categories = set()
        for result in valid_results:
            all_categories.update(result['category_accuracies'].keys())
        all_categories = sorted(list(all_categories))
        
        category_summary = []
        for category in all_categories:
            total_correct = 0
            total_predictions = 0
            total_test_images = 0
            
            for result in valid_results:
                if category in result['category_results']:
                    stats = result['category_results'][category]
                    total_correct += stats['correct_predictions']
                    total_predictions += stats['total_predictions']
                    total_test_images += stats['test_images']
            
            overall_accuracy = total_correct / total_predictions if total_predictions > 0 else 0.0
            
            category_summary.append({
                'category': category,
                'overall_accuracy': overall_accuracy,
                'total_correct': total_correct,
                'total_predictions': total_predictions,
                'total_test_images': total_test_images
            })
        
        category_summary.sort(key=lambda x: x['overall_accuracy'], reverse=True)
        
        for i, item in enumerate(category_summary, 1):
            f.write(f"   {i:2d}.   {item['category']:20s}   {item['overall_accuracy']:7.4f}   "
                   f"{item['total_correct']:4d}/{item['total_predictions']:4d}        {item['total_test_images']:3d}\n")
        
        f.write("\n")
        
        # 3. 每個 Backbone 的詳細類別表現
        f.write("📋 每個 BACKBONE 的詳細類別表現:\n")
        f.write("=" * 80 + "\n")
        
        for result in backbone_summary:
            backbone = result['backbone']
            backbone_result = next(r for r in valid_results if r['backbone'] == backbone)
            
            f.write(f"\n🔧 {backbone.upper()}:\n")
            f.write("-" * 50 + "\n")
            f.write(f"  整體準確率: {result['accuracy']:.4f} ({result['correct']}/{result['total']})\n")
            f.write(f"  測試圖片: {result['test_images']}, 失敗測試: {result['failed']}\n")
            f.write(f"  預期總預測數: {(result['test_images'] - result['failed']) * top_k}\n")
            f.write("\n  各類別詳細表現:\n")
            f.write("    類別                  準確率    正確/總預測  測試圖片\n")
            f.write("    " + "-" * 55 + "\n")
            
            # 類別表現排序
            category_perfs = []
            for category in all_categories:
                if category in backbone_result['category_results']:
                    stats = backbone_result['category_results'][category]
                    accuracy = backbone_result['category_accuracies'][category]
                    category_perfs.append((category, accuracy, stats['correct_predictions'], 
                                         stats['total_predictions'], stats['test_images']))
            
            category_perfs.sort(key=lambda x: x[1], reverse=True)
            
            for category, accuracy, correct, total, test_imgs in category_perfs:
                f.write(f"    {category:20s}  {accuracy:7.4f}   {correct:3d}/{total:3d}      {test_imgs:2d}\n")
        
        # 4. 統計總結
        f.write("\n" + "=" * 80 + "\n")
        f.write("📈 統計總結:\n")
        f.write("-" * 50 + "\n")
        
        all_accuracies = [item['accuracy'] for item in backbone_summary]
        total_test_images = sum(item['test_images'] for item in backbone_summary) // len(backbone_summary)
        total_expected_predictions = total_test_images * top_k * len(backbone_summary)
        total_actual_predictions = sum(item['total'] for item in backbone_summary)
        total_correct_predictions = sum(item['correct'] for item in backbone_summary)
        
        f.write(f"  測試設定:\n")
        f.write(f"    Top-K: {top_k}\n")
        f.write(f"    測試圖片數: {total_test_images}\n")
        f.write(f"    評估的 Backbone 數: {len(backbone_summary)}\n")
        f.write(f"    預期總預測數: {total_expected_predictions}\n")
        f.write(f"    實際總預測數: {total_actual_predictions}\n")
        f.write(f"    總正確預測數: {total_correct_predictions}\n")
        f.write(f"\n")
        
        f.write(f"  Backbone 表現統計:\n")
        f.write(f"    最高準確率: {max(all_accuracies):.4f} ({backbone_summary[0]['backbone']})\n")
        f.write(f"    最低準確率: {min(all_accuracies):.4f} ({backbone_summary[-1]['backbone']})\n")
        f.write(f"    平均準確率: {np.mean(all_accuracies):.4f}\n")
        f.write(f"    標準差: {np.std(all_accuracies):.4f}\n")
        f.write(f"\n")
        
        all_cat_accuracies = [item['overall_accuracy'] for item in category_summary]
        f.write(f"  類別表現統計:\n")
        f.write(f"    最易分類別: {category_summary[0]['category']} ({category_summary[0]['overall_accuracy']:.4f})\n")
        f.write(f"    最難分類別: {category_summary[-1]['category']} ({category_summary[-1]['overall_accuracy']:.4f})\n")
        f.write(f"    類別平均準確率: {np.mean(all_cat_accuracies):.4f}\n")
        f.write(f"    類別準確率標準差: {np.std(all_cat_accuracies):.4f}\n")
        
        f.write(f"\n")
        f.write("=" * 80 + "\n")
        f.write("計算方式說明:\n")
        f.write("- 準確率 = 正確預測數 / 總預測數\n")
        f.write(f"- 每張測試圖片產生 {top_k} 次預測\n")
        f.write("- 每次預測正確與否獨立計算\n")
        f.write("- 總預測數 = 測試圖片數 × Top-K\n")
        f.write("=" * 80 + "\n")
        f.write("日誌生成完成\n")
        f.write("=" * 80 + "\n")
    
    print(f"📝 詳細日誌已生成: {log_path}")
    return log_path

def main():
    parser = argparse.ArgumentParser(description='無監督學習Backbone批量評估系統（正確計算版）')
    parser.add_argument('--testdata', type=str, required=True,
                       help='測試數據目錄路徑')
    parser.add_argument('--models-dir', type=str, required=True,
                       help='模型和特徵數據庫目錄路徑')
    parser.add_argument('--backbones', type=str, nargs='+', 
                       default=['mobilenet', 'resnet18', 'resnet50', 'efficientnet_b0', 
                               'efficientnet_b2', 'vit_tiny', 'vit_small', 'fashion_resnet'],
                       help='要評估的backbone列表')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Top-K相似度搜尋')
    parser.add_argument('--output-dir', type=str, default='./correct_evaluation_results',
                       help='結果輸出目錄')
    
    args = parser.parse_args()
    
    print("🚀 無監督學習Backbone批量評估系統（正確計算版）")
    print("   準確率 = 正確預測數 / 總預測數 (總預測數 = 測試圖片數 × Top-K)")
    print("=" * 70)
    
    # 檢查必要檔案
    if not os.path.exists('unsupervised_test.py'):
        print("❌ 找不到 unsupervised_test.py，請確保該檔案在同一目錄下")
        return
    
    if not os.path.exists(args.testdata):
        print(f"❌ 測試數據目錄不存在: {args.testdata}")
        return
    
    if not os.path.exists(args.models_dir):
        print(f"❌ 模型目錄不存在: {args.models_dir}")
        return
    
    # 創建輸出目錄
    os.makedirs(args.output_dir, exist_ok=True)
    temp_output_dir = os.path.join(args.output_dir, 'temp_reports')
    os.makedirs(temp_output_dir, exist_ok=True)
    
    try:
        # 載入測試數據
        test_data, categories = load_test_data_with_labels(args.testdata)
        
        # 自動找到模型檔案
        model_configs = find_model_files(args.models_dir, args.backbones)
        
        if not model_configs:
            print("❌ 沒有找到任何有效的模型配置")
            return
        
        print(f"\n📊 評估設置:")
        print(f"  測試圖片數量: {len(test_data)}")
        print(f"  類別數量: {len(categories)}")
        print(f"  Top-K: {args.top_k}")
        print(f"  預期總預測數 (每個backbone): {len(test_data) * args.top_k}")
        print(f"  找到的模型: {list(model_configs.keys())}")
        
        # 評估每個backbone
        all_results = []
        for backbone in args.backbones:
            if backbone in model_configs:
                config = model_configs[backbone]
                result = evaluate_single_backbone(
                    backbone, config['model_path'], config['labels_path'],
                    test_data, args.top_k, temp_output_dir
                )
                all_results.append(result)
            else:
                print(f"\n⚠️ 跳過 {backbone}: 找不到模型或特徵檔案")
                all_results.append(None)
        
        # 生成綜合報告和視覺化
        summary_report = generate_comprehensive_csv_reports_and_visualizations(
            all_results, args.output_dir, args.top_k
        )
        
        # 生成詳細日誌
        log_path = create_comprehensive_log(all_results, args.output_dir, args.top_k)
        
        # 清理臨時目錄
        import shutil
        shutil.rmtree(temp_output_dir)
        
        # 顯示最終排名
        valid_results = [r for r in all_results if r is not None]
        if valid_results:
            print(f"\n🏆 最終 Backbone 排名 (Top-{args.top_k} 相似度搜尋):")
            print("   (準確率 = 正確預測數 / 總預測數)")
            print("-" * 60)
            
            backbone_rankings = []
            for result in valid_results:
                backbone_rankings.append({
                    'backbone': result['backbone'],
                    'accuracy': result['accuracy'],
                    'correct': result['correct_predictions'],
                    'total': result['total_predictions']
                })
            
            backbone_rankings.sort(key=lambda x: x['accuracy'], reverse=True)
            
            for i, item in enumerate(backbone_rankings, 1):
                print(f"  {i}. {item['backbone']:20s}: {item['accuracy']:.4f} "
                      f"({item['correct']:4d}/{item['total']:4d})")
        
        print(f"\n✅ 評估完成！結果已保存到: {args.output_dir}")
        print(f"📊 CSV報告、視覺化圖表和詳細日誌已生成")
        
    except Exception as e:
        print(f"❌ 評估失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()