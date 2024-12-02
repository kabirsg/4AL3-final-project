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

class FundusDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        self.image_files = {
            f.split('_output_')[0]: f for f in os.listdir(data_dir) if f.endswith('.jpg')
        }
        self.mask_files = {
            f.split('_mask_')[0]: f for f in os.listdir(data_dir) if f.endswith('.png')
        }

        self.common_keys = sorted(set(self.image_files.keys()) & set(self.mask_files.keys()))
        assert len(self.common_keys) > 0, "No matching images and masks found!"

    def __len__(self):
        return len(self.common_keys)

    def __getitem__(self, idx):
        key = self.common_keys[idx]
        img_path = os.path.join(self.data_dir, self.image_files[key])
        mask_path = os.path.join(self.data_dir, self.mask_files[key])

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image = self.transform(image)
            mask = Resize((128, 128))(mask)  
            mask = ToTensor()(mask)        

        return image, mask


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4)  # Increased dropout rate
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

        return torch.sigmoid(self.final_conv(d1))
    
def train(model, train_loader, loss_function, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, masks in train_loader:
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
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = loss_function(outputs, masks)
            val_loss += loss.item()
    return val_loss / len(val_loader)

def iou_score(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    return intersection / union

def dice_score(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum()
    return (2. * intersection) / (preds.sum() + targets.sum())

# Define the main function
def main(learning_rate, num_epochs, data_dir):
    # Transformations for the dataset
    transform = Compose([
        Resize((128, 128)),
        ToTensor()
    ])

    # Load the dataset
    dataset = FundusDataset(data_dir, transform=transform)
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Initialize model, loss function, and optimizer
    model = UNet(in_channels=3, out_channels=1).to(device)
    loss_function = nn.BCELoss()
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=learning_rate, 
        momentum=0.9,  # Adding momentum to help stabilize the gradients
        weight_decay=1e-1  # Adding weight decay to regularize and prevent overfitting
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # Early stopping parameters
    patience = 5  # Number of epochs to wait for improvement
    min_val_loss = np.Inf  # Initially set to infinity
    patience_counter = 0

    # Training and validation
    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, loss_function, optimizer, device)
        val_loss = validate(model, val_loader, loss_function, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Print loss after every epoch
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # Step the scheduler
        scheduler.step(val_loss)

        # Early Stopping Check
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            patience_counter = 0  # Reset counter if validation loss improves
            # Save the model if the validation loss is the best we've seen so far
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1

        # If the validation loss hasn't improved for 'patience' epochs, stop training
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    # Load the best model
    model.load_state_dict(torch.load('best_model.pth', weights_only=True))


    # Plot training and validation loss
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Testing the model
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            predictions = (outputs > 0.5).float()
            correct += (predictions == masks).sum().item()
            total += masks.numel()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

main(0.02, 20, "eyedata/output_images")
    