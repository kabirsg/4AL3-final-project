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
from Fundus_Final_project_good import FundusDataset #CHANGE THIS to point to Training.py eventually
from Fundus_Final_project_good import *
import sys


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

data_dir = "eyedata/input_images_generated_prep"
BATCH_SIZE = 4 #should this be the same?
PRINT_ALL = True
PRINT_EVERY_N = 10
RES_Y = 400
RES_X = 640

# If model specified on commandline, use it
# Otherwise, use the default model

model_file = 'models\model_file_20241213-233150_final_torch'
if len(sys.argv) > 1:
    model_file = sys.argv[1]

# If file path doesn't include models/ prefix and file doesn't exist without it, add it
if not os.path.exists(model_file) and os.path.exists('models/' + model_file):
    model_file = 'models/' + model_file

# If model name ends with _torch, use torch model
if model_file.endswith('_torch'):
    loaded_model = torch.load(model_file, map_location=torch.device(device))
else:
    loaded_model = pickle.load(open(model_file, 'rb'))

if len(sys.argv) > 2:
    data_dir = sys.argv[2]  


# Load the test data
test_dataset = FundusDataset(data_dir, device, include_orig=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

loaded_model.eval()
total_images = 0
with torch.no_grad():
        # Graph the results
        for i, (images, masks, original_images) in enumerate(test_loader):
            images, masks = images.to(device), masks.to(device)
            outputs = loaded_model(images)
            outputs = torch.sigmoid(outputs)  # Apply sigmoid to convert logits to probabilities
        
            num_images = images.size(0)

            for idx in range(num_images):
                total_images += 1
                if not PRINT_ALL and total_images % PRINT_EVERY_N != 0:
                    continue
                
                fig, axes = plt.subplots(1, 5, figsize=(20, 5))
                
                """
                print("loss function")
                #loss = loss_function(outputs[idx], masks[idx])
                print("masks[idx]", masks[idx])
                #val_loss = loss.item()
                print("calc stats")
                val_iou = iou_score(outputs[idx], masks[idx]).item()
                val_dice = dice_score(outputs[idx], masks[idx]).item()
                val_pixel_accuracy = pixel_accuracy(outputs[idx], masks[idx]).item()
                val_fpr = false_positive_rate(outputs[idx], masks[idx])

                print(f'Val Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, Dice: {val_dice:.4f}, Pixel Accuracy: {val_pixel_accuracy:.4f}')
                """
        
                # Original Image
                axes[0].imshow(original_images[idx].cpu().permute(1, 2, 0))
                axes[0].set_title("Original Image")
                axes[0].axis('off')

                # Normalized Image
                #axes[1].imshow(images[idx].cpu().permute(1, 2, 0))
                #axes[1].set_title("Normalized Image")
                #axes[1].axis('off')

                # Sobel Filter Output
                axes[1].imshow(images[idx].squeeze().cpu(), cmap='gray')
                axes[1].set_title("Sobel Filter Output")
                axes[1].axis('off')
                
                # Ground Truth Mask
                axes[2].imshow(masks[idx].cpu().squeeze(), cmap='gray')
                axes[2].set_title("Ground Truth Mask")
                axes[2].axis('off')
                
                # Predicted Mask in Greyscale
                axes[3].imshow(outputs[idx].cpu().squeeze(), cmap='gray', vmin=0, vmax=1)
                axes[3].set_title("Predicted Mask (Greyscale)")
                axes[3].axis('off')

                # Predicted Mask in Binary
                outputs_b = (outputs > 0.5).float()
                axes[4].imshow(outputs_b[idx].cpu().squeeze(), cmap='gray', vmin=0, vmax=1)
                axes[4].set_title("Predicted Mask (Binary)")
                axes[4].axis('off')
                
                plt.tight_layout()
                plt.show()

