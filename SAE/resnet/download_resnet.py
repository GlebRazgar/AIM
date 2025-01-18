import urllib.request
import os

def download_resnet18():
    # URL for the pretrained ResNet-18 model
    url = 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet18-imagenet.pth'
    
    # Create a directory to save the model
    save_dir = 'pretrained_models'
    os.makedirs(save_dir, exist_ok=True)
    
    # Path to save the downloaded model
    save_path = os.path.join(save_dir, 'resnet18-imagenet.pth')
    
    # Download the model if it doesn't exist
    if not os.path.exists(save_path):
        print(f"Downloading ResNet-18 model from {url}")
        try:
            urllib.request.urlretrieve(url, save_path)
            print(f"Model saved to {save_path}")
        except Exception as e:
            print(f"Error downloading model: {e}")
    else:
        print(f"Model already exists at {save_path}")

if __name__ == "__main__":
    download_resnet18()