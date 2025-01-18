import os
import urllib.request
import zipfile
import shutil
from tqdm import tqdm

def download_tiny_imagenet():
    # Create directory for the dataset
    os.makedirs('tiny-imagenet-200', exist_ok=True)
    
    # URL for Tiny ImageNet
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    zip_path = 'tiny-imagenet-200.zip'
    
    # Download if not already present
    if not os.path.exists(zip_path):
        print(f"Downloading Tiny ImageNet from {url}")
        urllib.request.urlretrieve(url, zip_path)
        print("Download completed!")
    
    # Extract the dataset
    if not os.path.exists('tiny-imagenet-200/train'):
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('.')
        print("Extraction completed!")
        
        # Clean up
        os.remove(zip_path)
    
    print("\nDataset structure:")
    print("- tiny-imagenet-200/train: Training images (200 classes, 500 images per class)")
    print("- tiny-imagenet-200/val: Validation images (10,000 images)")
    print("- tiny-imagenet-200/test: Test images (10,000 images)")

if __name__ == "__main__":
    download_tiny_imagenet()