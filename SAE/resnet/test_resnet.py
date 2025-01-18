import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import urllib.request
import os

def download_test_image():
    os.makedirs('test_images', exist_ok=True)
    
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image_path = "test_images/cat.jpg"
    
    if not os.path.exists(image_path):
        print(f"Downloading test image from {image_url}")
        urllib.request.urlretrieve(image_url, image_path)
        print(f"Image saved to {image_path}")
    
    return image_path

class CustomResNet18(nn.Module):
    def __init__(self):
        super(CustomResNet18, self).__init__()
        
        # First layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Fixed input dimensions - the first layer takes 128 channels as input
        self.layer1 = self._make_layer(128, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1000)

    def _make_layer(self, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        
        layers = []
        layers.append(BasicBlock(inplanes, planes, stride, downsample))
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes))
        
        return nn.Sequential(*layers)
   

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

def test_resnet18(image_path):
    # Load the model
    model = CustomResNet18()
    state_dict = torch.load('pretrained_models/resnet18-imagenet.pth')
    
    # Remove the unexpected keys before loading
    for key in list(state_dict.keys()):
        if '_tmp_running_mean' in key or '_tmp_running_var' in key or '_running_iter' in key:
            del state_dict[key]
            
    model.load_state_dict(state_dict)
    model.eval()
    
    # Load and preprocess the image
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    img = Image.open(image_path)
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)
    
    # Get predictions
    with torch.no_grad():
        output = model(batch_t)
    
    # Get top 5 predictions
    _, indices = torch.sort(output[0], descending=True)
    percentages = torch.nn.functional.softmax(output[0], dim=0) * 100
    
    # Download class labels from torchvision
    from torchvision.models import ResNet18_Weights
    weights = ResNet18_Weights.DEFAULT
    class_labels = weights.meta["categories"]
    
    print("\nTop 5 predictions:")
    for idx in indices[:5]:
        class_idx = idx.item()
        print(f"{class_labels[class_idx]}: {percentages[class_idx]:.2f}%")

if __name__ == "__main__":
    image_path = download_test_image()
    test_resnet18(image_path)