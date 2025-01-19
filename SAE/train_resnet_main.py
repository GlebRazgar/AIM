import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets

import numpy as np
from tqdm import tqdm
import os

# Import your custom ResNet
from SAE.resnet.resnet_class import CustomResNet18

# Import your favorite SAE (pick whichever)
from sae import SparseAutoencoder

# This is the GDN-based code
from gdn import GDN

seed = 12
torch.manual_seed(seed)

def save_checkpoint(state, filename="my_checkpoint.pth"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

class ResNetValidator:
    """
    Lightly adapted version of YOLO-based 'DSIAC_validator' but for ResNet classification + SAEs.
    """

    def __init__(self, 
                 data_path=None, 
                 device='cpu', 
                 LOAD_MODEL=False, 
                 experiment_name="",
                 sae_class=SparseAutoencoder,
                 layer_to_hook="layer3"):  
        self.device = device
        self.data_path = data_path
        self.experiment_name = experiment_name
        self.layer_to_hook = layer_to_hook

        # Load classification dataset (TinyImageNet as example)
        self.train_dataset, self.val_dataset = self.get_datasets(data_path)
        self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True, num_workers=2)
        self.val_loader   = DataLoader(self.val_dataset,   batch_size=32, shuffle=False, num_workers=2)

        # Build the ResNet
        self.model = self.get_model().to(self.device)

        # Freeze the ResNet
        for param in self.model.parameters():
            param.requires_grad = False

        self.sae = sae_class(input_channels=256, num_filters=64).to(self.device)  
        # Example: if hooking layer3, dimension might be 256 channels, so we set input_channels=256

        self.optim_sae = torch.optim.Adam(self.sae.parameters(), lr=1e-3)

        # Optionally load pretrained SAE
        if LOAD_MODEL:
            ckpt_path = f"/path/to/sae_checkpoint.pth"
            load_checkpoint(torch.load(ckpt_path, map_location=self.device), self.sae)

        # Hook storage
        self.feature_storage = None
        # Register forward hook
        target_layer = getattr(self.model, self.layer_to_hook)  # e.g. model.layer3
        target_layer.register_forward_hook(self._save_hook)

    def get_datasets(self, data_path):
        """
        Example of using TinyImageNet from torchvision.datasets.ImageFolder
        (assuming you put the 'tiny-imagenet-200/train_flat' and 'tiny-imagenet-200/val' in data_path).
        """
        # Simple transforms
        transform = transforms.Compose([
            transforms.Resize((64, 64)), 
            transforms.ToTensor()
        ])

        train_dir = os.path.join(data_path, 'tiny-imagenet-200', 'train_flat')
        val_dir   = os.path.join(data_path, 'tiny-imagenet-200', 'val', 'images')

        train_dataset = datasets.ImageFolder(train_dir, transform=transform)
        val_dataset   = datasets.ImageFolder(val_dir, transform=transform)

        return train_dataset, val_dataset

    def get_model(self):
        """Return a CustomResNet18 model (with 1000 output classes, as normal)."""
        model = CustomResNet18()
        # Optionally load pretrained Imagenet weights if you have them
        # But for simplicity, let's just random-init
        return model

    def _save_hook(self, module, input, output):
        """
        PyTorch forward hook that captures the output (feature map) of the chosen layer.
        """
        # output is the layer's feature map [batch_size, 256, H, W], for example
        self.feature_storage = output.detach()

    def train_sae(self, epochs=5):
        """
        Example: train the SAE on features extracted from the chosen ResNet layer.
        """
        self.model.eval()  # We do NOT train ResNet

        for epoch in range(epochs):
            epoch_loss = 0.0
            for images, labels in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                images = images.to(self.device)

                # 1) Forward pass ResNet
                # self.feature_storage will be set by the hook
                _ = self.model(images)

                # 2) Retrieve the hook features
                if self.feature_storage is None:
                    continue
                features = self.feature_storage  # shape e.g. [32, 256, H, W]

                # 3) Encode/Decode with SAE
                z, xhat = self.sae(features)

                # 4) Compute a reconstruction loss
                loss = nn.MSELoss()(xhat, features)

                self.optim_sae.zero_grad()
                loss.backward()
                self.optim_sae.step()

                epoch_loss += loss.item()

            print(f"Epoch {epoch+1} - Avg Loss: {epoch_loss/len(self.train_loader):.4f}")

        # Optionally save the SAE checkpoint
        checkpoint = {
            "state_dict": self.sae.state_dict()
        }
        save_checkpoint(checkpoint, filename=f"sae_checkpoint_{self.experiment_name}.pth")

    def validate_sae(self):
        """
        Example “validation” pass. Usually for autoencoders, you’d just measure reconstruction error on val set.
        """
        self.model.eval()
        self.sae.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                _ = self.model(images)

                if self.feature_storage is None:
                    continue
                features = self.feature_storage

                # Pass through SAE
                _, xhat = self.sae(features)
                loss = nn.MSELoss()(xhat, features)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(self.val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

    def inspect_sae(self):
        """Do any special inspection/visualization here."""
        print("Inspecting SAE outputs... (not yet implemented)")