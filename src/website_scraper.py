import os
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time
import zipfile
 
class NCERTScraper:
    
    def __init__(self, chromedriver_path="/usr/local/bin/chromedriver", base_url="https://ncert.nic.in/textbook.php?"):
        """Initialize with chromedriver path and base URL."""
        self.chromedriver_path = chromedriver_path
        self.base_url = base_url
        self.zip_folder = "books"
        os.makedirs(self.zip_folder, exist_ok=True)
        
    def _setup_driver(self):
        """Set up and return a headless Chrome driver."""
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        service = Service(self.chromedriver_path)
        return webdriver.Chrome(service=service, options=chrome_options)
        
    def _find_download_link(self, url):
        """Find download link for NCERT textbook from URL."""
        driver = self._setup_driver()
        driver.get(url)
        time.sleep(7)
        
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")
        driver.quit()
        
        download_link = None
        for a_tag in soup.find_all("a"):
            if "Download complete book" in a_tag.get_text():
                download_link = a_tag["href"]
                break
                
        if download_link and download_link.startswith(".."):
            download_link = "https://ncert.nic.in/" + download_link.replace("..", "")
            
        return download_link
        
    def _download_file(self, download_link):
        """Download file from the given link."""
        response = requests.get(download_link, stream=True)
        if response.status_code == 200:
            zip_filename = os.path.join(self.zip_folder, download_link.split("/")[-1])
            with open(zip_filename, "wb") as zip_file:
                for chunk in response.iter_content(1024):
                    zip_file.write(chunk)
            return True
        return False
    
    def scrape_data(self):
        book_codes = [
            "fees1=0-14",
            "gess1=0-8",
            "hess2=0-8",
            "iess3=0-5",
            "jess3=0-5"
        ]
        
        success_count = 0
        for code in book_codes:
            url = self.base_url + code
            print(f"Processing {url}")
            
            download_link = self._find_download_link(url)
            if download_link:
                print(f"Found download link: {download_link}")
                if self._download_file(download_link):
                    success_count += 1
                    print(f"Downloaded book {success_count} of {len(book_codes)}")
                else:
                    print(f"Failed to download from {download_link}")
            else:
                print(f"No download link found for {url}")
                
        print(f"Downloaded {success_count} of {len(book_codes)} books")
        return self.zip_folder
        
    def extract_zip(self):
        """Extract all ZIP files in the zip folder."""
        extract_folder = os.path.join(self.zip_folder, "extracted")
        os.makedirs(extract_folder, exist_ok=True)
        
        zip_files = [f for f in os.listdir(self.zip_folder) if f.endswith(".zip")]
        
        if not zip_files:
            print("No ZIP files found to extract.")
            return
            
        for zip_file in zip_files:
            zip_path = os.path.join(self.zip_folder, zip_file)
            print(f"Extracting {zip_file}...")
            
            try:
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(extract_folder)
                print(f"Successfully extracted {zip_file}")
            except Exception as e:
                print(f"Error extracting {zip_file}: {str(e)}")
                
        return extract_folder