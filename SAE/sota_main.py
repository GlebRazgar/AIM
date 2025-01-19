import torch
from torchvision import transforms, models

# from PyUnity.ultralytics.ultralytics import YOLO

from train_main import sae_trainer as trainer


# import mlflow

seed = 124
torch.manual_seed(seed)


# Load standard ResNet18 with pretrained weights from torchvision
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
# Get the class mapping
weights = models.ResNet18_Weights.IMAGENET1K_V1
categories = weights.meta["categories"]
class_labels = weights.meta["categories"]  # These are human-readable names

# Create mappings
idx_to_label = {i: label for i, label in enumerate(class_labels)}
label_to_idx = {label: i for i, label in enumerate(class_labels)}

model.eval()
if torch.cuda.is_available():
    model = model.cuda()


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOAD_MODEL = True

experiment_name = "mech_interp"

def main():
    path = 'dataset/imagenet-mini' 

    sae_trainer = trainer(data_path=path, device=DEVICE, nc=11, model=model, experiment_name=experiment_name)

    sae_trainer.train_sae()
    sae_trainer.inspect_sae()


if __name__ == "__main__":
    main()