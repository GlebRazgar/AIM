import os
import subprocess
import zipfile
import sys
import stat

def setup_kaggle():
    """Setup Kaggle credentials"""
    print("Setting up Kaggle credentials...")
    
    # Create .kaggle directory
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    
    # Copy kaggle.json to the right location
    try:
        subprocess.run(["cp", "kaggle.json", kaggle_dir], check=True)
        # Set correct permissions (600)
        kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")
        os.chmod(kaggle_json_path, stat.S_IRUSR | stat.S_IWUSR)
        print("Kaggle credentials setup complete!")
    except subprocess.CalledProcessError:
        print("Error: Failed to copy kaggle.json")
        print("Make sure kaggle.json exists in the current directory")
        sys.exit(1)
    except Exception as e:
        print(f"Error during Kaggle setup: {str(e)}")
        sys.exit(1)

def setup_imagenet_mini():
    """Download and setup ImageNet Mini dataset"""
    os.makedirs("imagenet-mini", exist_ok=True)
    os.chdir("imagenet-mini")
    
    print("Starting ImageNet Mini setup...")
    
    try:
        print("Downloading dataset...")
        subprocess.run(["kaggle", "datasets", "download", "-d", "ifigotin/imagenetmini-1000"], 
                      check=True)
    except subprocess.CalledProcessError:
        print("Error: Failed to download dataset. Make sure kaggle is installed and configured.")
        print("Run: pip install kaggle")
        print("And ensure your kaggle.json is in ~/.kaggle/")
        sys.exit(1)
    
    try:
        print("Unzipping dataset (this may take a few minutes)...")
        with zipfile.ZipFile("imagenetmini-1000.zip", 'r') as zip_ref:
            for file_info in zip_ref.filelist:
                if file_info.filename.startswith('imagenet-mini/'):
                    new_filename = file_info.filename.replace('imagenet-mini/', '', 1)
                    if new_filename:
                        with zip_ref.open(file_info) as source, \
                             open(new_filename, 'wb') as target:
                            target.write(source.read())
        
        os.remove("imagenetmini-1000.zip")
        print("Setup complete! Dataset is ready in imagenet-mini directory")
        
        if os.path.exists("train"):
            n_train_classes = len(os.listdir("train"))
            print(f"Number of training classes: {n_train_classes}")
    
    except zipfile.BadZipFile:
        print("Error: Downloaded file is corrupted or not a zip file")
        sys.exit(1)
    except Exception as e:
        print(f"Error during extraction: {str(e)}")
        sys.exit(1)

def main():
    print("Starting dataset setup process...")
    setup_kaggle()
    setup_imagenet_mini()
    print("Dataset setup completed successfully!")

if __name__ == "__main__":
    main()