# import mlflow
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from torchvision import transforms, models
import torchvision
import scipy
import matplotlib.pyplot as plt

import numpy as np
from sae import SparseAutoencoder, SparseAutoencoder2, SparseAutoencoder3, SparseAutoencoder4, SparseAutoencoder5, SparseAutoencoder7, SparseAutoencoder8, SparseAutoencoder9
import cv2
import matplotlib.patches as patches
import pickle
# from PyUnity.agents.common.shap_util import OD2Score, SuperPixler, CastNumpy
import matplotlib as mpl

import wandb
from datetime import datetime

import umap
# from umap.parametric_umap import ParametricUMAP
import pandas as pd
import seaborn as sns
import math
import random

from matplotlib.colors import Normalize

seed = 12  # 124
torch.manual_seed(seed)

def save_checkpoint(state, filename="my_checkpoint.pth"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


class sae_trainer():  # pass as argument 'trainer' when calling model.train()

    def __init__(self, model, data_path=None, nc=3, ckpt=None, device='cpu', experiment_name=""):
        super().__init__()
        self.device = device
        self.data_path = data_path
        self.nc = nc
        self.thresh = 0.4
        self.val_dataset, self.val_loader = self.get_dataloader(data_path, mode="test")  
        self.layer = 1
        self.num_models = 1  # size of ensemble of models
        self.sae = SparseAutoencoder4(input_channels=64).to(self.device)  
        self.optim_sae = torch.optim.SGD(self.sae.parameters(), lr=1e-5)
        self.epochs = 10
        self.model = model
        self.models = [model]  # Store model in list for compatibility with existing code
        # Move model to the correct device
        self.model.to(self.device)

        wandb.init(
        project="sae-resnet",
        name=f"sae-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config={
            "architecture": "SparseAutoencoder4",
            "learning_rate": 1e-3,
            "epochs": self.epochs,
            "batch_size": 8,
            "lambda": 10,
            "rho": 0.05,
            "device": device,
            "resnet_layer": 3
            }
        )

        pbar = tqdm(self.val_loader, leave=True)

        os.makedirs("results/trained_models", exist_ok=True)
        
        #self.LOAD_SAE_MODEL_FILE = "/Users/nmital/Results/trained_models/SAE_cbe1c6005e5b472bbd9b975bff4b52a2_[6]"
        #self.LOAD_SAE_MODEL_FILE = f"/Users/nmital/Results/trained_models/SAE[3]_cbe1c6005e5b472bbd9b975bff4b52a2_[10]_yolo_layer[{self.layer}]"
        self.SAVE_SAE_MODEL_FILE = "results/trained_models/SAE_cbe1c6005e5b472bbd9b975bff4b52a2"
        # print(self.LOAD_MODEL_FILE)
        
        self.model.to(self.device)
        '''if LOAD_MODEL:
            load_checkpoint(torch.load(self.LOAD_SAE_MODEL_FILE+f".pth", map_location=torch.device('cpu')), self.sae)
            self.sae.to(self.device)'''

    # def build_dataset(self, img_path=None, mode="train", batch=None):
    #     """
    #     Build YOLO Dataset.

    #     Args:
    #         img_path (str): Path to the folder containing images.
    #         mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
    #         batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
    #     """

    #     real_range = [[0, 20], [160, 200], [340,
    #                                         360]]  # [[20, 30], [60, 70], [100, 110], [140, 150], [180, 190], [220, 230], [260, 270], [300, 310],[340, 350]]
    #     # test_range = [[80,100], [260, 280]] #[[0, 5], [40, 45], [80, 85], [120, 125], [160, 165], [200, 205], [240, 245], [280, 285],[320, 325]]


    #     test_range = [[260, 261]]  # [[70, 72], [80, 82], [90, 92], [100, 102], [110,112]]  # [[70,71], [80,81], [90,91], [100,101], [110,111],[250,251], [260,261], [270,271], [280,281], [290,291]] # for tsne
    #     real_tgts = ['BTR70']  # ['BTR70', 'T72', 'BRDM2', 'BMP2', '2S3', 'ZSU23_4', 'SUV', 'PICKUP']
    #     syn_tgts = ['ZSU23_4']
    #     ds1 = DSIAC_sae(path=self.data_path, set_type='test', time='night',
    #                  ranges={'real': [1500], 'synthetic': [1500]},
    #                  targets={'real': real_tgts, 'synthetic': syn_tgts, 'unknown': []},
    #                  aspect_range={'real': test_range, 'synthetic': test_range},
    #                  data_type=['real'],
    #                  synthetic_folder="Synthetic_sae2")
    #     indices = list(np.random.randint(low=0, high=ds1.__len__(), size=1))
    #     ds1 = torch.utils.data.Subset(ds1, indices)

    #     test_range = [[74,75]]
    #     real_tgts = ['ZSU23_4']
    #     syn_tgts = ['SUV']
    #     ds2 = DSIAC_sae(path=self.data_path, set_type='test', time='night',
    #                 ranges={'real': [1500], 'synthetic': [1500]},
    #                 targets={'real': real_tgts, 'synthetic': syn_tgts, 'unknown': []},
    #                 aspect_range={'real': test_range, 'synthetic': test_range},
    #                 data_type=['synthetic'], synthetic_folder="Synthetic_sae5")
    #     indices = list(np.random.randint(low=0, high=ds2.__len__(), size=int(1)))
    #     ds2 = torch.utils.data.Subset(ds2, indices)

    #     test_range = [[74,75]]
    #     real_tgts = ['ZSU23_4']
    #     syn_tgts = ['SUV']
    #     ds3 = DSIAC_sae(path=self.data_path, set_type='test', time='night',
    #                 ranges={'real': [1500], 'synthetic': [1500]},
    #                 targets={'real': real_tgts, 'synthetic': syn_tgts, 'unknown': []},
    #                 aspect_range={'real': test_range, 'synthetic': test_range},
    #                 data_type=['synthetic'], synthetic_folder="Synthetic_sae6")
    #     indices = list(np.random.randint(low=0, high=ds3.__len__(), size=int(1)))
    #     ds3 = torch.utils.data.Subset(ds3, indices)

    #     test_range = [[72,
    #                    73]]  # [[70, 72], [80, 82], [90, 92], [100, 102], [110,112]]  # [[70,71], [80,81], [90,91], [100,101], [110,111],[250,251], [260,261], [270,271], [280,281], [290,291]] # for tsne
    #     real_tgts = ['SUV']  # ['BTR70', 'T72', 'BRDM2', 'BMP2', '2S3', 'ZSU23_4', 'SUV', 'PICKUP']
    #     syn_tgts = ['ZSU23_4']
    #     ds5 = DSIAC_sae(path=self.data_path, set_type='test', time='night',
    #                     ranges={'real': [1500], 'synthetic': [1500]},
    #                     targets={'real': real_tgts, 'synthetic': syn_tgts, 'unknown': []},
    #                     aspect_range={'real': test_range, 'synthetic': test_range},
    #                     data_type=['real'],
    #                     synthetic_folder="Synthetic_sae2")
    #     indices = list(np.random.randint(low=0, high=ds5.__len__(), size=1))
    #     ds5 = torch.utils.data.Subset(ds5, indices)

    #     ds = torch.utils.data.ConcatDataset([ds1, ds2, ds3, ds5])
    #     return ds


    def visualize_activations(self, features, sparse_features, epoch):
        # Create directory for saving visualizations
        os.makedirs("results/activations", exist_ok=True)
        
        # 1. Feature Maps Visualization
        n_features = min(16, sparse_features.shape[1])  # Show first 16 features or less
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        for idx in range(n_features):
            i, j = idx//4, idx%4
            feature_map = sparse_features[0, idx].detach().cpu().numpy()
            axes[i, j].imshow(feature_map, cmap='viridis')
            axes[i, j].axis('off')
        plt.suptitle(f'SAE Feature Activations - Epoch {epoch}')
        
        # Save to wandb and local directory
        plt.savefig(f'results/activations/features_epoch_{epoch}.png')
        wandb.log({
            "feature_maps": wandb.Image(plt.gcf()),
            "epoch": epoch
        })
        plt.close()

        # 2. Activation Statistics
        avg_activation = torch.mean(sparse_features, dim=(0,2,3))
        wandb.log({
            "activation/mean": wandb.Histogram(avg_activation.detach().cpu().numpy()),
            "activation/sparsity": (sparse_features == 0).float().mean().item(),
            "epoch": epoch
        })

    def get_dataloader(self, dataset_path, batch_size=8, rank=0, mode="train"):
        """Construct and return dataloader."""

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(dataset_path, 'train'),
            transform=transform)
        
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
            drop_last=True)
        
        return dataset, data_loader  # Return both dataset and loader


    def visualise_layers(self):
        loop = tqdm(self.val_loader, leave=True)
        dataset = []
        labels = []
        arr_pad = 0
        for idx, data in enumerate(loop):
            x = data["img"]
            x = x.to(self.device)
            features = []
            for j in range(23):
                features += [self.extract_features(self.models[0], x, layer_index=j)]
                # features_flat = torch.flatten(features[-1])
            print("features computed")
            return data, features

    def plot_features(self, data, features, spf):
        #plt.close('all')
        spf = spf.detach().numpy()
        spf_ind = spf >= self.thresh
        spf2 = np.where(spf_ind == True, spf, 0)
        features = features.detach().numpy()
        w_gt = data['bboxes'][0, 2]
        h_gt = data['bboxes'][0, 3]
        x_gt = data['bboxes'][0, 0] - w_gt / 2
        y_gt = data['bboxes'][0, 1] - h_gt / 2
        rect0 = patches.Rectangle((float(torch.floor(x_gt * 512)), float(torch.floor(y_gt * 256))),
                                  float(torch.ceil(w_gt * 512)), float(torch.ceil(h_gt * 256)), linewidth=2,
                                  edgecolor='y', facecolor='none')
        rect1 = patches.Rectangle((float(torch.floor(x_gt * 64*2)), float(torch.floor(y_gt * 32*2))),
                                  float(torch.ceil(w_gt * 64*2)), float(torch.ceil(h_gt * 32*2)), linewidth=2,
                                  edgecolor='y', facecolor='none')
        rect2 = patches.Rectangle((float(torch.floor(x_gt * 64)), float(torch.floor(y_gt * 32))),
                                  float(torch.ceil(w_gt * 64)), float(torch.ceil(h_gt * 32)), linewidth=2,
                                  edgecolor='y', facecolor='none')
        fig1, ax1 = plt.subplots(1, 1)
        #fig2, ax2 = plt.subplots(1, 1)
        fig0, ax0 = plt.subplots(1, 1)

        ax0.imshow(data['img'][0,0,:,:], cmap='gray', vmin=0, vmax=1.0)
        ax0.add_patch(rect0)

        spf = cv2.resize(spf, dsize=(features.shape[1], features.shape[0]), interpolation=cv2.INTER_NEAREST_EXACT)
        ax1.imshow(spf, cmap='gray', vmin=0, vmax=1.5)
        #ax1.imshow(features, cmap='gray')
        ax1.add_patch(rect1)

        #spf2 = cv2.resize(spf2, dsize=(features.shape[1], features.shape[0]), interpolation=cv2.INTER_NEAREST_EXACT)
        #ax2.imshow(spf2, cmap='gray', vmin=0, vmax=1.0)
        #ax2.add_patch(rect2)


    def inspect_sae(self, data=None, ref_features=[]):

        # x = torch.nn.Parameter(torch.zeros(size=(1,1, 128,256)), requires_grad=True)
        if data:
            x = [data['img']]

            x[0] = torch.nn.Parameter(x[0], requires_grad=False)
            x[0] = x[0].to(self.device)
        else:
            loop = tqdm(self.val_loader, leave=True)
            data_list = []
            x = []
            for idx, data in enumerate(loop):
                data_list += [data]
                x += [data['img']]
                x[-1] = torch.nn.Parameter(x[-1], requires_grad=True)
                x[-1] = x[-1].to(self.device)

        for model in self.models:
            for param in model.parameters():
                param.requires_grad = False

        layer = self.layer

        num_steps = 2  # how many optim steps to take
        loss_fn = torch.nn.MSELoss(reduction='mean')
        sae_features_original = None

        sae_features = []
        features_list = []
        reconstructed_features_list = []
        for i in range(len(x)):
            features = self.extract_features(self.models[0], x[i], layer_index=layer)
            sparse_features, features_reconstruction = self.sae(features)
            sparse_features_flat = torch.reshape(sparse_features, shape=[sparse_features.shape[0],
                                                                         sparse_features.shape[1] *
                                                                         sparse_features.shape[2] *
                                                                         sparse_features.shape[3]])
            sparse_features_flat = torch.norm(sparse_features_flat, p=1, dim=0) / sparse_features_flat.shape[0]
            spf = sparse_features / 0.5 #(torch.max(sparse_features_flat))
            features_list += [features]
            reconstructed_features_list += [features_reconstruction]
            sae_features += [spf]

        indices = [0, 2, 3]
        #features_ind, common_features_ind, spf_bb, spf_thres = self.correlate_neurons(data_list, indices, sae_features)

    def train_sae(self):
        for model in self.models:
            for param in model.parameters():
                param.requires_grad = False

        loss_fn1 = torch.nn.MSELoss(reduction='mean')
        lmbda = 5  # regularisation coefficient
        rho = 0.1
        
        for epoch in range(self.epochs):
            loop = tqdm(self.val_loader, leave=True)
            epoch_loss1 = 0.0
            epoch_loss2 = 0.0
            epoch_total_loss = 0.0
            num_batches = 0
            
            if (epoch + 1) % 30 == 0:
                self.optim_sae.param_groups[0]['lr'] = max(self.optim_sae.param_groups[0]['lr'] * 0.1, 1e-06)
                    
            for idx, (images, _) in enumerate(loop):
                x = images
                x = x.to(self.device)
                features = self.extract_features(self.models[0], x, layer_index=3)

                print('lr: ', self.optim_sae.param_groups[0]['lr'])
                print("Epoch: ", epoch)
                sparse_features, features_reconstruction = self.sae(features)
                
                # First flatten sparse features
                sparse_features_flat = torch.reshape(sparse_features, shape=[sparse_features.shape[0],
                                                                        sparse_features.shape[1] *
                                                                        sparse_features.shape[2] *
                                                                        sparse_features.shape[3]])
                
                # Then normalize
                sparse_features_flat = torch.norm(sparse_features_flat, p=1, dim=0) / sparse_features_flat.shape[0]
                sparse_features_flat[sparse_features_flat <= 0] = 0.0001
                sparse_features_flat[sparse_features_flat >= 1] = 0.9999
                
                # Move visualization here, after features are computed
                if epoch % 1 == 0 and idx == 0:  # Visualize first batch every 2 epochs
                    self.visualize_activations(features, sparse_features, epoch)
                        
                kl_div = - rho * torch.log(sparse_features_flat / rho) - (1 - rho) * torch.log(
                    (1 - sparse_features_flat) / (1 - rho))
                loss1 = loss_fn1(features, features_reconstruction)
                loss2 = torch.sum(kl_div) / kl_div.shape[0]

                loss = loss1 + lmbda * loss2
                loss.backward()
                self.optim_sae.step()
                self.optim_sae.zero_grad()

                # Update metrics
                epoch_loss1 += loss1.item()
                epoch_loss2 += loss2.item()
                epoch_total_loss += loss.item()
                num_batches += 1
                
                # Log batch metrics
                wandb.log({
                    "batch/reconstruction_loss": loss1.item(),
                    "batch/sparsity_loss": loss2.item(),
                    "batch/total_loss": loss.item(),
                    "batch/learning_rate": self.optim_sae.param_groups[0]['lr']
                })

                loop.set_postfix(loss1=loss1, loss2=loss2)
                spf = sparse_features / (torch.max(sparse_features_flat))
                spf_ind = spf >= 0.4
                spf = torch.where(spf_ind == True, spf, 0)

            # Log epoch metrics
            wandb.log({
                "epoch": epoch,
                "epoch/avg_reconstruction_loss": epoch_loss1 / num_batches,
                "epoch/avg_sparsity_loss": epoch_loss2 / num_batches,
                "epoch/avg_total_loss": epoch_total_loss / num_batches
            })

            # Save model checkpoint
            checkpoint = {
                "state_dict": self.sae.state_dict(),
                "optimizer": self.optim_sae.state_dict(),
            }
            
            # Save checkpoint both locally and to wandb
            save_checkpoint(checkpoint, filename=self.SAVE_SAE_MODEL_FILE + f".pth")
            if (epoch + 1) % 5 == 0:  # Save to wandb every 5 epochs
                wandb.save(self.SAVE_SAE_MODEL_FILE + f".pth")

            print("Avg loss: ", epoch_total_loss / num_batches)

        # Close wandb run
        wandb.finish()


    def extract_features(self, model, img, layer_index=3):
        global intermediate_features
        intermediate_features = []
        
        # Map layer_index to actual ResNet layers
        resnet_layers = {
            0: model.conv1,
            1: model.bn1,
            2: model.relu,
            3: model.maxpool,
            4: model.layer1,
            5: model.layer2,
            6: model.layer3,
            7: model.layer4,
        }
        
        if layer_index not in resnet_layers:
            raise ValueError(f"Invalid layer_index {layer_index}. Must be between 0 and 7.")
            
        # Register hook on the correct layer
        hook = resnet_layers[layer_index].register_forward_hook(self.hook_fn)
        
        # Forward pass
        x = model(img)
        hook.remove()
        return intermediate_features[0]

    # Define hook function
    def hook_fn(self, module, input, output):
        intermediate_features.append(output) # Apends output of a particular layer


