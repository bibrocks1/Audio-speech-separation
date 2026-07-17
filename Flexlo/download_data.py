import os
import sys
import urllib.request
import time

URLS = {
    "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "train-clean-100": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
    "rirs_noises": "https://www.openslr.org/resources/28/rirs_noises.zip"
}

last_update_time = 0

def report_hook(count, block_size, total_size):
    global start_time, last_update_time
    if count == 0:
        start_time = time.time()
        last_update_time = time.time()
        return
        
    current_time = time.time()
    duration = current_time - start_time
    progress_size = int(count * block_size)
    percent = min(int(count * block_size * 100 / total_size), 100) if total_size > 0 else 0
    
    # Throttle updates to at most once per second
    if current_time - last_update_time >= 1.0 or percent == 100:
        speed = int(progress_size / (1024 * 1024 * duration)) if duration > 0 else 0
        sys.stdout.write(f"\rDownloading... {percent}% | {progress_size / (1024 * 1024):.1f} MB / {total_size / (1024 * 1024):.1f} MB | Speed: {speed} MB/s | Time: {duration:.1f}s")
        sys.stdout.flush()
        last_update_time = current_time

def download_file(url, dest_dir):
    filename = url.split("/")[-1]
    dest_path = os.path.join(dest_dir, filename)
    
    if os.path.exists(dest_path):
        print(f"\n{filename} already exists at {dest_path}. Skipping download.")
        return dest_path
        
    print(f"\nStarting download of {url} to {dest_path}")
    
    # Ensure destination directory exists
    os.makedirs(dest_dir, exist_ok=True)
    
    try:
        urllib.request.urlretrieve(url, dest_path, report_hook)
        print(f"\nSuccessfully downloaded {filename}")
    except Exception as e:
        print(f"\nError downloading {filename}: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        sys.exit(1)
        
    return dest_path

def main():
    if '__file__' in globals():
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    else:
        base_dir = os.getcwd()
    raw_dir = os.path.join(base_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    print("Dataset Downloader for Speech Separation Setup")
    print(f"Destination folder: {raw_dir}")
    print("-" * 50)
    
    downloaded = {}
    for name, url in URLS.items():
        downloaded[name] = download_file(url, raw_dir)
        
    print("\nAll downloads finished successfully!")

if __name__ == "__main__":
    main()
