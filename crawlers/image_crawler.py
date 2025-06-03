import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import random
import re
from datetime import datetime

class URLImageCrawler:
    def __init__(self, base_url, output_dir='downloaded_images'):
        self.base_url = base_url
        # Create a unique folder name based on timestamp and URL
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitize URL to create a valid folder name
        sanitized_url = re.sub(r'[^\w\-]', '_', base_url.replace('https://', '').replace('http://', ''))
        # Limit folder name length
        sanitized_url = sanitized_url[:50]  # Limit to 50 characters
        self.output_dir = os.path.join(output_dir, f"{timestamp}_{sanitized_url}")
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Create main output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Create specific directory for this URL
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created directory: {self.output_dir}")

    def download_image(self, img_url, filename):
        try:
            # Skip data URLs
            if img_url.startswith('data:'):
                return False
                
            response = requests.get(img_url, headers=self.headers)
            if response.status_code == 200:
                filepath = os.path.join(self.output_dir, filename)
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                print(f"Successfully downloaded: {filename}")
                return True
        except Exception as e:
            print(f"Error downloading {img_url}: {str(e)}")
        return False

    def extract_images(self, url):
        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code != 200:
                print(f"Failed to fetch {url}")
                return []

            soup = BeautifulSoup(response.text, 'html.parser')
            images = []

            # Find all img tags
            for img in soup.find_all('img'):
                src = img.get('src')
                if src and not src.startswith('data:'):  # Skip data URLs
                    # Convert relative URLs to absolute URLs
                    img_url = urljoin(url, src)
                    # Get filename from URL
                    filename = os.path.basename(urlparse(img_url).path)
                    if not filename:
                        filename = f"image_{int(time.time())}_{random.randint(1000, 9999)}.jpg"
                    
                    images.append((img_url, filename))

            return images

        except Exception as e:
            print(f"Error processing {url}: {str(e)}")
            return []

    def crawl(self):
        print(f"Starting to crawl: {self.base_url}")
        images = self.extract_images(self.base_url)
        
        print(f"Found {len(images)} images")
        for img_url, filename in images:
            self.download_image(img_url, filename)
            # Add a small delay to be respectful to the server
            time.sleep(0.5)

def main():
    print("Welcome to the Image Crawler!")
    print("Paste URLs to crawl images from them.")
    print("Type 'exit' or 'quit' to end the program.")
    print("-" * 50)

    while True:
        # Get URL from user input
        target_url = input("\nEnter URL to crawl (or 'exit' to quit): ").strip()
        
        # Check for exit command
        if target_url.lower() in ['exit', 'quit']:
            print("Thank you for using the Image Crawler!")
            break
            
        # Validate URL
        if not target_url.startswith(('http://', 'https://')):
            print("Please enter a valid URL starting with http:// or https://")
            continue
            
        try:
            print(f"\nProcessing: {target_url}")
            crawler = URLImageCrawler(target_url)
            crawler.crawl()
            print("-" * 50)
        except Exception as e:
            print(f"Error processing URL: {str(e)}")
            print("Please try another URL.")
            continue

if __name__ == "__main__":
    main()