import torch
from torchvision import transforms, models
from PIL import Image
import os
from tqdm import tqdm
import json

def load_model():
    # Load standard ResNet18 with pretrained weights from torchvision
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
    return model

def process_image(image_path):
    # Standard ImageNet preprocessing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def run_inference():
    model = load_model()
    
    # Path to ImageNet-mini training data
    # Updated path to ImageNet-mini training data
    data_path = os.path.join('..', 'dataset', 'imagenet-mini', 'train')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please run download_data.py first.")


    # Store results
    results = {}
    
    # Process each class folder
    class_folders = os.listdir(data_path)
    for class_folder in tqdm(class_folders, desc="Processing classes"):
        class_path = os.path.join(data_path, class_folder)
        results[class_folder] = {'total': 0, 'correct': 0}
        
        # Process each image in the class folder
        for image_file in os.listdir(class_path):
            if not image_file.lower().endswith(('.jpg', '.jpeg')):
                continue
                
            image_path = os.path.join(class_path, image_file)
            
            try:
                # Prepare image
                input_tensor = process_image(image_path)
                if torch.cuda.is_available():
                    input_tensor = input_tensor.cuda()
                
                # Run inference
                with torch.no_grad():
                    output = model(input_tensor)
                    _, predicted = output.max(1)
                
                # Update results
                results[class_folder]['total'] += 1
                # ImageNet classes are numbered from 0-999
                if predicted.item() == int(class_folder[1:].split('_')[0]):
                    results[class_folder]['correct'] += 1
                    
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                continue
    
    # Calculate and print results
    total_images = sum(d['total'] for d in results.values())
    total_correct = sum(d['correct'] for d in results.values())
    
    print("\nInference Results:")
    print(f"Total images processed: {total_images}")
    print(f"Total correct predictions: {total_correct}")
    print(f"Accuracy: {(total_correct/total_images)*100:.2f}%")
    
    # Save detailed results to file
    with open('inference_results.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    run_inference()