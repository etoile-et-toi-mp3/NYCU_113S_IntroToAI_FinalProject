# python pinterest_crawler.py --keywords "style_keyword" --num-images 100 --output-dir "output_folder_name"

import os
import time
import argparse
import requests
import re
import random
import urllib.parse
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import logging
import urllib3
requests.packages.urllib3.disable_warnings()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fashion_scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PinterestScraper:
    def __init__(self, output_dir="fashion_images", headless=True):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36")
        chrome_options.add_argument("--disable-notifications")
        chrome_options.add_argument("--start-maximized")
        
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-infobars")
        chrome_options.add_argument("--disable-browser-side-navigation")
        chrome_options.add_argument("--disable-features=TranslateUI")
        chrome_options.add_argument("--disk-cache-size=1073741824")  
        chrome_options.add_argument("--media-cache-size=1073741824") 
        
        try:
            self.driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()),
                options=chrome_options
            )
            self.driver.set_page_load_timeout(30) 
            logger.info("Chrome WebDriver 初始化成功")
        except Exception as e:
            logger.error(f"無法初始化Chrome WebDriver: {e}")
            raise
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7',
            'Referer': 'https://www.google.com/',
        }
        
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.session.verify = False

    def __del__(self):
        if hasattr(self, 'driver'):
            try:
                self.driver.quit()
                logger.info("Chrome WebDriver 已成功關閉")
            except Exception as e:
                logger.error(f"關閉WebDriver出錯: {e}")
    
    def sanitize_filename(self, filename):
        return re.sub(r'[\\/*?:"<>|]', "", filename)
    
    def download_image(self, img_url, save_path):
        try:
            if not img_url.startswith(('http://', 'https://')):
                if img_url.startswith('//'):
                    img_url = 'https:' + img_url
                else:
                    logger.warning(f"無效URL: {img_url}")
                    return False
            
            response = self.session.get(
                img_url, 
                stream=True, 
                timeout=15
            )
            
            if response.status_code == 200:
                content_type = response.headers.get('Content-Type', '')
                if not content_type.startswith('image/'):
                    logger.info(f"跳過非圖像內容 ({content_type}): {img_url}")
                    return False
                
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(8192):
                        f.write(chunk)
                
                file_size = os.path.getsize(save_path)
                if file_size < 1000:
                    logger.info(f"跳過太小的圖像 ({file_size} 位元組): {img_url}")
                    os.remove(save_path)
                    return False
                
                logger.info(f"已下載: {os.path.basename(save_path)} ({file_size} 位元組)")
                return True
            else:
                logger.warning(f"下載失敗 {img_url}, 狀態碼: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"下載 {img_url} 時出錯: {e}")
            return False
    
    def scroll_to_load_more(self, scroll_times=30, scroll_pause=2.0):
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        
        for i in range(scroll_times):
            scroll_amount = random.uniform(0.7, 1.0)
            self.driver.execute_script(f"window.scrollTo(0, document.body.scrollHeight * {scroll_amount});")
            
            time.sleep(random.uniform(scroll_pause * 0.8, scroll_pause * 1.2))
            
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                try:
                    load_more_button = self.driver.find_element(By.XPATH, 
                        "//*[contains(text(), 'Load more') or contains(text(), '載入更多') or contains(@class, 'load-more')]")
                    self.driver.execute_script("arguments[0].click();", load_more_button)
                    time.sleep(random.uniform(2.0, 3.0))
                except:
                    if i > 3:
                        logger.info(f"沒有更多內容可以載入，已滾動 {i+1} 次")
                        break
            
            last_height = new_height
            
            if (i+1) % 5 == 0:
                logger.info(f"已滾動 {i+1}/{scroll_times} 次")
    
    def scrape_pinterest(self, keyword, num_images=1000, start_number=1):
        search_url = f"https://www.pinterest.com/search/pins/?q={urllib.parse.quote(keyword)}%20fashion%20style"
        
        folder_name = self.sanitize_filename(keyword)
        save_folder = os.path.join(self.output_dir, folder_name)
        os.makedirs(save_folder, exist_ok=True)
        
        logger.info(f"在Pinterest上搜尋 '{keyword}'")
        
        try:
            self.driver.get(search_url)
            
            try:
                WebDriverWait(self.driver, 20).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "img[srcset]"))
                )
            except Exception:
                logger.warning("等待Pinterest圖像載入逾時")
            
            try:
                close_buttons = self.driver.find_elements(By.XPATH, "//button[contains(@class, 'closeup-close') or contains(text(), 'Close') or contains(text(), '關閉')]")
                if close_buttons:
                    for button in close_buttons:
                        self.driver.execute_script("arguments[0].click();", button)
                        time.sleep(1)
            except Exception:
                pass
            
            scroll_times = max(40, num_images // 8)  
            self.scroll_to_load_more(scroll_times=scroll_times, scroll_pause=2.5)
            
            img_elements = self.driver.find_elements(By.CSS_SELECTOR, "img[srcset], img.hCL, div.GrowthUnauthPinImage img")
            logger.info(f"在Pinterest上找到 {len(img_elements)} 個圖像元素")
            
            img_urls = []
            for img in img_elements:
                try:
                    srcset = img.get_attribute("srcset")
                    if srcset:
                        parts = srcset.split(',')
                        for part in parts:
                            if '4x' in part or '3x' in part or '2x' in part:  
                                url = part.strip().split(' ')[0]
                                if url:
                                    img_urls.append(url)
                                    break
                        else:  
                            largest_img = parts[-1].strip().split(' ')[0]
                            if largest_img:
                                img_urls.append(largest_img)
                    else:
                        src = img.get_attribute("src")
                        if src and not src.startswith("data:"):
                            src = re.sub(r'/(236x|474x)/', '/originals/', src)
                            img_urls.append(src)
                except Exception as e:
                    logger.error(f"提取Pinterest圖像URL時出錯: {e}")
            
            unique_urls = []
            seen = set()
            for url in img_urls:
                if url not in seen:
                    seen.add(url)
                    unique_urls.append(url)
            
            logger.info(f"找到 {len(unique_urls)} 個唯一Pinterest圖像URL")
            
            downloaded_count = 0
            current_number = start_number
            for i, img_url in enumerate(unique_urls):
                if downloaded_count >= num_images:
                    break
                
                file_extension = os.path.splitext(img_url.split('?')[0])[1]
                if not file_extension or len(file_extension) > 5:
                    file_extension = '.jpg'
                
                filename = f"{keyword}{current_number:03d}{file_extension}"
                save_path = os.path.join(save_folder, filename)
                
                if self.download_image(img_url, save_path):
                    downloaded_count += 1
                    current_number += 1
                
                time.sleep(random.uniform(0.2, 0.8))
                
                if downloaded_count % 50 == 0:
                    logger.info(f"已從Pinterest下載 {downloaded_count} 張圖像")
            
            logger.info(f"已為關鍵字 '{keyword}' 從Pinterest下載 {downloaded_count} 張圖像")
            return downloaded_count, current_number - 1  
            
        except Exception as e:
            logger.error(f"為關鍵字 '{keyword}' 抓取Pinterest時出錯: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return 0, start_number
    
    def scrape_keywords(self, keywords, num_images=1000, retry_count=3):
        results = {}
        total_downloaded = 0
        
        for keyword in keywords:
            logger.info(f"\n{'='*50}\n處理關鍵字: {keyword}\n{'='*50}")
            
            folder_name = self.sanitize_filename(keyword)
            save_folder = os.path.join(self.output_dir, folder_name)
            
            start_number = 1
            if os.path.exists(save_folder):
                existing_files = [f for f in os.listdir(save_folder) if os.path.isfile(os.path.join(save_folder, f))]
                if existing_files:
                    numbers = []
                    pattern = re.compile(f"{keyword}([0-9]+)\\.")
                    for file in existing_files:
                        match = pattern.search(file)
                        if match:
                            try:
                                num = int(match.group(1))
                                numbers.append(num)
                            except ValueError:
                                pass
                    
                    if numbers:
                        start_number = max(numbers) + 1
                        logger.info(f"找到現有檔案，從編號 {start_number} 開始")
            
            downloaded = 0
            last_number = start_number - 1
            retry = 0
            
            while downloaded < num_images and retry < retry_count:
                logger.info(f"為關鍵字 '{keyword}' 嘗試下載 {num_images - downloaded} 張圖像 (嘗試 {retry + 1}/{retry_count})")
                
                batch_downloaded, last_number = self.scrape_pinterest(
                    keyword, 
                    num_images - downloaded,
                    start_number=last_number + 1
                )
                
                downloaded += batch_downloaded
                
                if batch_downloaded == 0:
                    retry += 1
                else:
                    retry = 0
                
                if downloaded >= num_images or (retry > 0 and batch_downloaded == 0):
                    break
            
            results[keyword] = downloaded
            total_downloaded += downloaded
            logger.info(f"關鍵字 '{keyword}' 總共下載了 {downloaded} 張圖像")
        
        return total_downloaded, results

def batch_download(keywords, num_images_per_keyword=1000, batch_size=3, output_dir="fashion_images", headless=True):
    total_results = {}
    total_downloaded = 0
    
    keyword_batches = [keywords[i:i+batch_size] for i in range(0, len(keywords), batch_size)]
    
    for batch_index, keyword_batch in enumerate(keyword_batches):
        logger.info(f"\n{'#'*70}\n批次 {batch_index+1}/{len(keyword_batches)}: {keyword_batch}\n{'#'*70}")
        
        try:
            scraper = PinterestScraper(output_dir=output_dir, headless=headless)
            
            batch_downloaded, batch_results = scraper.scrape_keywords(
                keyword_batch, num_images_per_keyword
            )
            
            total_results.update(batch_results)
            total_downloaded += batch_downloaded
            
            logger.info(f"\n{'='*50}\n批次 {batch_index+1} 下載完成! 此批次下載: {batch_downloaded} 張圖像\n{'='*50}")
            
            del scraper
            
            if batch_index < len(keyword_batches) - 1:
                logger.info(f"暫停 10 秒後開始下一批...")
                time.sleep(10)
                
        except Exception as e:
            logger.error(f"處理批次 {batch_index+1} 時出錯: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    return total_downloaded, total_results

def main():
    parser = argparse.ArgumentParser(description='從Pinterest抓取時尚圖像')
    parser.add_argument('--keywords', nargs='+', required=True, help='要搜尋的關鍵字清單')
    parser.add_argument('--num-images', type=int, default=1000, help='每個關鍵字要下載的圖像數量')
    parser.add_argument('--output-dir', default='fashion_images', help='儲存圖像的基礎目錄')
    parser.add_argument('--no-headless', action='store_true', help='以可見模式執行瀏覽器（非無頭）')
    parser.add_argument('--batch-size', type=int, default=3, help='每批處理的關鍵字數量')
    parser.add_argument('--retry-count', type=int, default=3, help='每個關鍵字嘗試的最大次數')
    
    args = parser.parse_args()
    
    if args.keywords:
        try:
            keywords = []
            for k in args.keywords:
                if isinstance(k, bytes):
                    k = k.decode('utf-8')
                keywords.append(k)
            args.keywords = keywords
        except Exception as e:
            logger.warning(f"警告: 處理關鍵字時出錯: {e}")
    
    logger.info(f"使用關鍵字啟動Pinterest爬蟲: {args.keywords}")
    logger.info(f"每個關鍵字的圖像數量: {args.num_images}")
    logger.info(f"批次大小: {args.batch_size}, 重試次數: {args.retry_count}")
    
    try:
        total_downloaded, results = batch_download(
            keywords=args.keywords,
            num_images_per_keyword=args.num_images,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            headless=not args.no_headless
        )
        
        logger.info("\n" + "="*50)
        logger.info(f"下載完成! 總共下載的圖像: {total_downloaded}")
        logger.info(f"圖像儲存在: {os.path.abspath(args.output_dir)}")
        
        logger.info("\n摘要:")
        for keyword, count in results.items():
            logger.info(f"  {keyword}: {count} 張圖像")
    
    except Exception as e:
        logger.error(f"主執行中出錯: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()