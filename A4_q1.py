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
import os

torch.manual_seed(1)
device = torch.device('cpu')

print("----------Part 1: CNN on FashionMNIST dataset----------")
print()
# Part 1: CNN on FashionMNIST dataset
# Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self, input_dim=28, input_channels=1, num_classes=10, hidden_dims=[128, 64]):

        super(SimpleCNN, self).__init__()
        # Define variables for channels and kernel size
        INPUT_CHANNELS = input_channels
        CONV1_OUT_CHANNELS = 10  # Conv_10
        CONV2_OUT_CHANNELS = 5   # Conv_5
        CONV3_OUT_CHANNELS = 16  # Conv_16
        KERNEL_SIZE = 3  # Kernel size for all convolutional layers
        POOL_SIZE = 2    # Pool size for MaxPooling
        POOL_STRIDE = 2  # Stride for pooling
        PADDING = 0      # No padding for convolutional layers

        def calculate_output_dim(input_dim, kernel_size, stride, padding):
            return (input_dim + 2 * padding - kernel_size) // stride + 1
        
        # Calculate dimensions dynamically
        conv1_out_dim = calculate_output_dim(input_dim, KERNEL_SIZE, 1, PADDING)
        pool1_out_dim = calculate_output_dim(conv1_out_dim, POOL_SIZE, POOL_STRIDE, 0)
        
        conv2_out_dim = calculate_output_dim(pool1_out_dim, KERNEL_SIZE, 1, PADDING)
        pool2_out_dim = calculate_output_dim(conv2_out_dim, POOL_SIZE, POOL_STRIDE, 0)
        
        conv3_out_dim = calculate_output_dim(pool2_out_dim, KERNEL_SIZE, 1, PADDING)
        pool3_out_dim = calculate_output_dim(conv3_out_dim, POOL_SIZE, POOL_STRIDE, 0)

        # Flattened features after all convolutions
        flattened_features = CONV3_OUT_CHANNELS * pool3_out_dim * pool3_out_dim

        # Define the convolutional layers
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=INPUT_CHANNELS, out_channels=CONV1_OUT_CHANNELS, kernel_size=KERNEL_SIZE),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=POOL_SIZE, stride=POOL_STRIDE),

            nn.Conv2d(in_channels=CONV1_OUT_CHANNELS, out_channels=CONV2_OUT_CHANNELS, kernel_size=KERNEL_SIZE),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=POOL_SIZE, stride=POOL_STRIDE),

            nn.Conv2d(in_channels=CONV2_OUT_CHANNELS, out_channels=CONV3_OUT_CHANNELS, kernel_size=KERNEL_SIZE),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=POOL_SIZE, stride=POOL_STRIDE)
        )

        # Define the fully connected layers dynamically
        fc_layers = []
        prev_dim = flattened_features
        for hidden_dim in hidden_dims:  # Create hidden layers
            fc_layers.append(nn.Linear(prev_dim, hidden_dim))
            fc_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        fc_layers.append(nn.Linear(prev_dim, num_classes))  # Final output layer
        self.fc_layers = nn.Sequential(*fc_layers)

    # Forward pass through the network
    def forward(self, x):
        x = self.layers(x)  # Pass through convolutional layers
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)  # Pass through fully connected layers
        return x

# Hyperparameters
batch_size = 64
learning_rate = 0.01
num_epochs = 30

input_dim = 28

# Data transformation
transform = Compose([
    Grayscale(num_output_channels=1),
    Resize((input_dim, input_dim)),
    ToTensor()
])

# Load dataset (FashionMNIST)
train_dataset = torchvision.datasets.FashionMNIST(root='.', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.FashionMNIST(root='.', train=False, download=True, transform=transform)

# Split train dataset into training and validation
frac = 0.8
train_size = int(frac * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model, loss, and optimizer
model = SimpleCNN().to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training function
def train(model, train_loader, loss_function, optimizer):        
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        # Forward pass
        outputs = model(images)
        loss = loss_function(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(train_loader)


# Validation function
def validate(model, val_loader, loss_function):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = loss_function(outputs, labels)
            val_loss += loss.item()
    return val_loss / len(val_loader)

# Training and validation
train_losses, val_losses = [], []
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, loss_function, optimizer)
    val_loss = validate(model, val_loader, loss_function)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # Print loss after every epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

# Plot training and validation loss
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Testing the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')
