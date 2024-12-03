import os
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize
import torch
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

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

# Define hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 8
NUM_EPOCHS = 30
DATA_DIR = "eyedata/output_images"
PRINT_ALL = True
PRINT_EVERY_N = 10

class FundusDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # Find all images and masks
        self.image_files = {
            f: f for f in os.listdir(data_dir) if '_output_' in f and f.endswith('.jpg')
        }
        self.mask_files = {
            f: f for f in os.listdir(data_dir) if '_mask_' in f and f.endswith('.png')
        }

        # Match images with their corresponding masks based on the shared prefix and suffix
        self.common_keys = []
        for img_name in self.image_files.keys():
            prefix_suffix = img_name.split('_output_')
            if len(prefix_suffix) == 2:
                potential_mask_name = f"{prefix_suffix[0]}_mask_{prefix_suffix[1].replace('.jpg', '.png')}"
                if potential_mask_name in self.mask_files:
                    self.common_keys.append((img_name, potential_mask_name))


    def __len__(self):
        return len(self.common_keys)

    def __getitem__(self, idx):
        img_name, mask_name = self.common_keys[idx]
        img_path = os.path.join(self.data_dir, img_name)
        mask_path = os.path.join(self.data_dir, mask_name)

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            transformed_image = self.transform(image)
            mask = Resize((128, 128))(mask)  # Ensure mask is resized independently
            mask = ToTensor()(mask)
            mask = (mask > 0.5).float()

        return transformed_image, mask, ToTensor()(image)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        
        self.encoder1 = conv_block(in_channels, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.encoder4 = conv_block(256, 512)

        self.pool = nn.MaxPool2d(2, 2)

        self.bottleneck = conv_block(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = conv_block(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.upconv4(b)
        d4 = self.decoder4(torch.cat((d4, e4), dim=1))
        d3 = self.upconv3(d4)
        d3 = self.decoder3(torch.cat((d3, e3), dim=1))
        d2 = self.upconv2(d3)
        d2 = self.decoder2(torch.cat((d2, e2), dim=1))
        d1 = self.upconv1(d2)
        d1 = self.decoder1(torch.cat((d1, e1), dim=1))

        return self.final_conv(d1)
    
def train(model, train_loader, loss_function, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, masks, _ in train_loader:
        images, masks = images.to(device), masks.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = loss_function(outputs, masks)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(train_loader)

# Define a helper function for validation
def validate(model, val_loader, loss_function, device):
    model.eval()
    val_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    total_pixel_accuracy = 0.0
    with torch.no_grad():
        for images, masks, _ in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = loss_function(outputs, masks)
            val_loss += loss.item()
            total_iou += iou_score(outputs, masks).item()
            total_dice += dice_score(outputs, masks).item()
            total_pixel_accuracy += pixel_accuracy(outputs, masks).item()
    return val_loss / len(val_loader), total_iou / len(val_loader), total_dice / len(val_loader), total_pixel_accuracy / len(val_loader)

def iou_score(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    return intersection / union

def dice_score(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum()
    return (2. * intersection) / (preds.sum() + targets.sum())

def pixel_accuracy(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    correct = (preds == targets).sum()
    total = targets.numel()
    return correct / total

# Function to calculate dataset-specific mean and std
def calculate_mean_std(data_dir):
    transform = Compose([
        Resize((128, 128)),
        ToTensor()
    ])
    dataset = FundusDataset(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    print("Calculating mean and standard deviation...")
    for images, _, _ in loader:
        for i in range(3):
            mean[i] += images[:, i, :, :].mean()
            std[i] += images[:, i, :, :].std()
    mean /= len(loader)
    std /= len(loader)
    print(f"Mean: {mean}")
    print(f"Std: {std}")
    return mean, std

# Define the main function
def main(learning_rate, num_epochs, data_dir):

    # Calculate dataset-specific mean and std
    mean, std = calculate_mean_std(data_dir)

    print("Training has started...")
    # Transformations for the dataset
    transform = Compose([
        Resize((128, 128)),
        ToTensor(),
        torchvision.transforms.Normalize(mean=mean.tolist(), std=std.tolist())  # Use calculated mean and std
    ])

    # Load the dataset
    dataset = FundusDataset(data_dir, transform=transform)
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(RANDOM_SEED))

    # Print the sizes of the training and validation datasets
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    """
    # Log the filenames used for training and validation
    print("Training data:")
    for idx in train_dataset.indices:
        print(dataset.common_keys[idx])
    print("\nValidation data:")
    for idx in val_dataset.indices:
        print(dataset.common_keys[idx])
    """

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Initialize model, loss function, and optimizer
    model = UNet(in_channels=3, out_channels=1).to(device)
    #loss_function = nn.BCELoss()
    #loss_function = nn.BCEWithLogitsLoss()

    pos_weight = torch.ones([1]) * 10  # Increase the weight to balance classes
    loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training and validation
    train_losses, val_losses, iou_scores, dice_scores, pixel_accuracies = [], [], [], [], []
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, loss_function, optimizer, device)
        val_loss, val_iou, val_dice, val_pixel = validate(model, val_loader, loss_function, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        iou_scores.append(val_iou)
        dice_scores.append(val_dice)
        pixel_accuracies.append(val_pixel)


        # Print loss, IoU, and Dice score after every epoch
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, Dice: {val_dice:.4f}, Pixel Accuracy: {val_pixel:.4f}')


    # Plot training and validation loss
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 3, 2)
    plt.plot(range(1, num_epochs + 1), iou_scores, label='IoU Score')
    plt.plot(range(1, num_epochs + 1), dice_scores, label='Dice Score')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.title('IoU and Dice Scores')

    plt.subplot(1, 3, 3)
    plt.plot(range(1, num_epochs + 1), pixel_accuracies, label='Pixel Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Pixel Accuracy')
    
    plt.tight_layout()
    plt.show()

    # Visualize all results
    model.eval()
    total_images = 0
    with torch.no_grad():
        for i, (images, masks, original_images) in enumerate(val_loader):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
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
            

    # Testing the model
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    total_pixel_accuracy = 0.0

    with torch.no_grad():
        for images, masks, _ in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            outputs = torch.sigmoid(outputs)  # Apply sigmoid to convert logits to probabilities
            loss = loss_function(outputs, masks)
            val_loss += loss.item()
            total_iou += iou_score(outputs, masks).item()
            total_dice += dice_score(outputs, masks).item()
            total_pixel_accuracy += pixel_accuracy(outputs, masks).item()
            predictions = (outputs > 0.5).float()
            correct += (predictions == masks).sum().item()
            total += masks.numel()

    val_loss /= len(val_loader)
    total_iou /= len(val_loader)
    total_dice /= len(val_loader)
    total_pixel_accuracy /= len(val_loader)

    print(f'Test Loss: {val_loss:.4f}, IoU: {total_iou:.4f}, Dice: {total_dice:.4f}, Pixel Accuracy: {total_pixel_accuracy:.4f}')
    
if __name__ == "__main__":
    main(LEARNING_RATE, NUM_EPOCHS, DATA_DIR)