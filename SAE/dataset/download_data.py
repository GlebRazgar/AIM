import os
import subprocess
import zipfile
import sys

def setup_imagenet_mini():
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
            # Extract contents directly without creating another imagenet-mini folder
            for file_info in zip_ref.filelist:
                if file_info.filename.startswith('imagenet-mini/'):
                    # Remove the 'imagenet-mini/' prefix when extracting
                    new_filename = file_info.filename.replace('imagenet-mini/', '', 1)
                    if new_filename:  # Skip the root directory entry
                        with zip_ref.open(file_info) as source, \
                             open(new_filename, 'wb') as target:
                            target.write(source.read())
        
        # Remove zip file after extraction
        os.remove("imagenetmini-1000.zip")
        print("Setup complete! Dataset is ready in imagenet-mini directory")
        
        # Print basic dataset info
        if os.path.exists("train"):
            n_train_classes = len(os.listdir("train"))
            print(f"Number of training classes: {n_train_classes}")
    
    except zipfile.BadZipFile:
        print("Error: Downloaded file is corrupted or not a zip file")
        sys.exit(1)
    except Exception as e:
        print(f"Error during extraction: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    setup_imagenet_mini()