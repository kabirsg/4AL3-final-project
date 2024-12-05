import os
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torchvision.transforms import Compose, RandomHorizontalFlip, Grayscale, Resize, RandomCrop, ToTensor
from torch.utils.data import RandomSampler, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import random
import pickle
from Fundus_Final_project_good import UNet, FundusDataset

#import Fundus_Final_project_good

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

data_dir = "eyedata/test_data"
BATCH_SIZE = 8
PRINT_ALL = True
PRINT_EVERY_N = 10

loaded_model = pickle.load(open('model_file', 'rb'))

#mean, std = calculate_mean_std(data_dir)

mean = torch.tensor([0.6229, 0.4124, 0.2856])
std = torch.tensor([0.1346, 0.1063, 0.0950])

# Transformations for the dataset
transform = Compose([
    Resize((128, 128)),
    ToTensor(),
    torchvision.transforms.Normalize(mean=mean.tolist(), std=std.tolist())  # Use calculated mean and std
])

# Load the test data
test_dataset = FundusDataset(data_dir, transform=transform)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

loaded_model.eval()
total_images = 0

with torch.no_grad():
        for i, (images, masks, original_images) in enumerate(test_loader):
            images, masks = images.to(device), masks.to(device)
            outputs = loaded_model(images)
            outputs = torch.sigmoid(outputs)  # Apply sigmoid to convert logits to probabilities
            outputs = (outputs > 0.5).float()

            num_images = images.size(0)

            for idx in range(num_images):
                total_images += 1
                if not PRINT_ALL and total_images % PRINT_EVERY_N != 0:
                    continue
                
                fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                
                # Original Image
                axes[0].imshow(original_images[idx].cpu().permute(1, 2, 0))
                axes[0].set_title("Original Image")
                axes[0].axis('off')
                
                # Normalized Image
                axes[1].imshow(images[idx].cpu().permute(1, 2, 0))
                axes[1].set_title("Normalized Image")
                axes[1].axis('off')
                
                # Ground Truth Mask
                axes[2].imshow(masks[idx].cpu().squeeze(), cmap='gray')
                axes[2].set_title("Ground Truth Mask")
                axes[2].axis('off')
                
                # Predicted Mask
                axes[3].imshow(outputs[idx].cpu().squeeze(), cmap='gray')
                axes[3].set_title("Predicted Mask")
                axes[3].axis('off')
                
                plt.tight_layout()
                plt.show()

