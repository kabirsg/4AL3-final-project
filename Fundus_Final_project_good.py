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
from datetime import datetime
import sys

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

SAVE_PICKLE = False

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))  # If CUDA is available

# Define hyperparameters
LEARNING_RATE = 0.004
BATCH_SIZE = 8
NUM_EPOCHS = 25
#DATA_DIR = "eyedata/entire_dataset"
DATA_DIR = "eyedata/output_images"
PRINT_ALL = True
PRINT_EVERY_N = 10
PRINT = False
RES_X = 640
RES_Y = 400
MASK_X = 32
MASK_Y = 20
DO_TEST = True
ONE_BIAS = 12

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

        image = Image.open(img_path).convert('RGBA')
        original_image = np.array(image)
        original_image_t = torch.tensor(original_image, dtype=torch.uint8).permute(2, 0, 1)

        mask = Image.open(mask_path).convert('L')

        if self.transform:
            transformed_image = self.transform(image)
        
            image_array = np.array(transformed_image, dtype=np.float32)  # Convert to NumPy array for processing
            image_min, image_max = image_array.min(), image_array.max()
            normalized_image = (image_array - image_min) / (image_max - image_min + 1e-8)
            normalized_image = torch.tensor(normalized_image, dtype=torch.float32).permute(2, 0, 1)  # Convert back to Tensor

            # Apply resizing and transforms to mask
            mask = Resize((MASK_Y, MASK_X))(mask)  # Match size to image
            mask = ToTensor()(mask)
            mask = (mask > 0.5).float()

            #print(f"Original Min: {image_min}, Max: {image_max}")
            #print(f"Normalized Min: {normalized_image.min()}, Max: {normalized_image.max()}")


            #print(f"Normalized image shape: {normalized_image.shape}")
            #print(f"Mask shape: {mask.shape}")
            

        return normalized_image, mask, original_image_t

class UNet_old(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        def conv_block(in_c, out_c, kernel_size=3):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=kernel_size//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=kernel_size, padding=kernel_size//2),
                nn.ReLU(inplace=True)
            )
        
        # encoders will be used with self.pool so the resolution will be halved
        self.pool = nn.MaxPool2d(2, 2) # Halves the resolution
        self.pool5 = nn.MaxPool2d(5, 5) # Fifths the resolution

        self.encoder1 = conv_block(in_channels, 64)  # w/ no pool Output: (64, 640, 400)
        self.encoder2 = conv_block(64, 128)          # w/ pool    Output: (128, 320, 200)
        self.encoder3 = conv_block(128, 256)         # w/ pool    Output: (256, 160, 100)
        self.encoder4 = conv_block(256, 512)         # w/ no pool Output: (512, 160, 100)
        self.encoder5 = conv_block(512, 1024)        # w/ no pool Output: (1024, 160, 100)
        self.encoder6 = conv_block(1024, 2048, 5)    # w/ pool5   Output: (2048, 32, 20)

        # stride of 2 will double the resolution, stride 1 will keep the resolution the same
        self.upconv5 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=1, padding=0, output_padding=0)
        self.decoder5 = conv_block(1024, 1024)       # Output: (1024, 32+1, 20+1)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=1, padding=1, output_padding=0)
        self.decoder4 = conv_block(512, 512)        # Output: (512, 32, 20)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=1, padding=0, output_padding=0)
        self.decoder3 = conv_block(256, 256)         # Output: (256, 32+1, 20+1)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=1, padding=1, output_padding=0)
        self.decoder2 = conv_block(128, 128)         # Output: (128, 32, 20)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=1, stride=1, padding=0, output_padding=0)
        self.decoder1 = conv_block(64, 64)          # Output: (64, 32, 20)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    

    def forward(self, x):

        def printDimention (str, x):
            print(str, " ", x.size())
            pass

        e1 = self.encoder1(x)
        printDimention("e1", e1)
        e2 = self.encoder2(self.pool(e1))
        printDimention("e2", e2)
        e3 = self.encoder3(self.pool(e2))
        printDimention("e3", e3)
        e4 = self.encoder4(e3)
        printDimention("e4", e4)
        e5 = self.encoder5(e4)
        printDimention("e5", e5)
        e6 = self.encoder6(self.pool5(e5))
        printDimention("e6", e6)

        # we are not doing skip connections as we are changing resolutions
        # and it leads to overfitting on the pixel level (or something like that, maybe)
        d5 = self.upconv5(e6)
        d5 = self.decoder5(d5)
        printDimention("d5", d5)
        d4 = self.upconv4(d5)
        d4 = self.decoder4(d4)
        printDimention("d4", d4)
        d3 = self.upconv3(d4)
        d3 = self.decoder3(d3)
        printDimention("d3", d3)
        d2 = self.upconv2(d3)
        d2 = self.decoder2(d2)
        printDimention("d2", d2)
        d1 = self.upconv1(d2)
        d1 = self.decoder1(d1)
        printDimention("d1", d1)

        r = self.final_conv(d1)
        printDimention("final", r)
        return r
    

class SobelFilter(nn.Module):
    def __init__(self):
        super(SobelFilter, self).__init__()
        # Define Sobel kernels
        sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.sobel_kernel_x = nn.Parameter(sobel_kernel_x, requires_grad=False)
        self.sobel_kernel_y = nn.Parameter(sobel_kernel_y, requires_grad=False)

    def forward(self, x):
        # Apply Sobel filter in x and y directions
        print("x ", x.size())
        # print(x);
        print("sobel_kernel_x ", self.sobel_kernel_x.size())
        grad_x = F.conv2d(x, self.sobel_kernel_x, padding=1)
        print("grad_x ", grad_x.size())
        grad_y = F.conv2d(x, self.sobel_kernel_y, padding=1)
        # Combine gradients
        grad = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        # print some of the raw pixel data:
        #print("grad ", grad)
        sobel_output = grad * 8;
        # change any values greater than 1 to 1
        clamped_sobel_output = torch.clamp(sobel_output, 0, 1)
        
        return clamped_sobel_output

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, dropout_rate=0.5):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.skip = nn.Conv2d(in_c, out_c, kernel_size=1) if in_c != out_c else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout2(out)
        out += identity
        out = self.relu(out)
        return out
    
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        def conv_block_without_residual(in_c, out_c, kernel_size=3):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=kernel_size//2),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Conv2d(out_c, out_c, kernel_size=kernel_size, padding=kernel_size//2),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
            )
        
        def conv_block(in_c, out_c, kernel_size=3):
            return ResidualBlock(in_c, out_c, kernel_size)
        
        self.sobel_filter = SobelFilter()
        in_channels = 1  # Converted RGB to grayscale
        
        self.downsample1 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)      # Keeps the resolution
        self.downsample2 = nn.Conv2d(128, 128, kernel_size=2, stride=2, padding=0)    # Halves the resolution
        self.downsample3 = nn.Conv2d(256, 256, kernel_size=2, stride=2, padding=0)    # Halves the resolution
        self.downsample4 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)    # Keeps the resolution
        self.downsample5 = nn.Conv2d(512, 512, kernel_size=5, stride=5, padding=0)  # Fifths the resolution

        self.encoder1 = conv_block(in_channels, 64)  # Output: (64, 640, 400)
        self.encoder2 = conv_block(64, 128)          # Output: (128, 320, 200)
        self.encoder3 = conv_block(128, 256)         # Output: (256, 160, 100)
        self.encoder4 = conv_block(256, 512)         # Output: (512, 160, 100)
        self.encoder5 = conv_block(512, 512, 5)      # Output: (512, 32, 20)

        # stride 1 will keep the resolution the same
        self.upconv1 = nn.ConvTranspose2d(512, 64, kernel_size=1, stride=1, padding=0, output_padding=0)
        self.decoder1 = conv_block(64, 64)           # Output: (64, 32, 20)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)  # Output: (1, 32, 20)
    
    def to_grayscale(self, x):
        # Convert RGB image to grayscale
        return x.mean(dim=1, keepdim=True)

    def forward(self, x):

        def printDimention (str, x):
            print(str, " ", x.size())

        # Convert to grayscale
        x_grey = self.to_grayscale(x)
        printDimention("x", x_grey)

        # Apply Sobel filter to enhance texture details before downsampling
        x_filtered = self.sobel_filter(x_grey)
        printDimention("x_filtered", x_filtered)

        e1 = self.encoder1(x_filtered)
        printDimention("e1", e1)
        e2 = self.encoder2(self.downsample1(e1))
        printDimention("e2", e2)
        e3 = self.encoder3(self.downsample2(e2))
        printDimention("e3", e3)
        e4 = self.encoder4(self.downsample3(e3))
        printDimention("e4", e4)
        e5 = self.encoder5(self.downsample4(e4))
        printDimention("e5", e5)
        e6 = self.downsample5(e5)
        printDimention("e6", e6)

        # we are not doing skip connections as we are changing resolutions
        # and it leads to overfitting on the pixel level (or something like that, maybe)
        d1 = self.upconv1(e6)
        d1 = self.decoder1(d1)
        printDimention("d1", d1)

        r = self.final_conv(d1)
        printDimention("final", r)
        return r


def train(model, train_loader, loss_function, optimizer, device):
    log("Training...")
    count = 0
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
        count += 1
        log(f"Batch {count}, Loss: {loss.item():.4f}")

    return running_loss / len(train_loader)

# Define a helper function for validation
def validate(model, val_loader, loss_function, device):
    log("Validating...")
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
    #print("preds", preds)
    #print("targets", targets)
    correct = (preds == targets).sum()
    total = targets.numel()
    #print("correct", correct)
    #print("total", total)
    return correct / total

# Function to calculate dataset-specific mean and std
def calculate_mean_std(data_dir):
    transform = Compose([
        Resize((RES_Y, RES_X))
        #ToTensor()
    ])
    dataset = FundusDataset(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    log("Calculating mean and standard deviation...")
    for images, _, _ in loader:
        for i in range(3):
            mean[i] += images[:, i, :, :].mean()
            std[i] += images[:, i, :, :].std()
    mean /= len(loader)
    std /= len(loader)
    log(f"Mean: {mean}")
    log(f"Std: {std}")
    return mean, std

datetimestr = datetime.now().strftime("%Y%m%d-%H%M%S")

def log(*args):
    print(*args)
    with open('logs/log_' + datetimestr + '.txt', 'a') as f:
        print(*args, file=f)

# Define the main function
def main(learning_rate, num_epochs, data_dir, continue_from):

    # Calculate dataset-specific mean and std
    #mean, std = calculate_mean_std(data_dir)

    # Use the mean and std calculated from the dataset for faster computation
    mean = torch.tensor([0.6229, 0.4124, 0.2856])
    std = torch.tensor([0.1346, 0.1063, 0.0950])

    log("Current Timestamp =", datetimestr)

    log("Running ", sys.argv[0])
    log("  LEARNING_RATE", LEARNING_RATE)
    log("  NUM_EPOCHS", NUM_EPOCHS)
    log("  DATA_DIR", DATA_DIR)
    log("  PRINT_ALL", PRINT_ALL)
    log("  PRINT_EVERY_N", PRINT_EVERY_N)
    log("  PRINT", PRINT)
    log("  RES_X", RES_X)
    log("  RES_Y", RES_Y)
    log("  MASK_X", MASK_X)
    log("  MASK_Y", MASK_Y)
    log("  DO_TEST", DO_TEST)
    log("  RANDOM_SEED", RANDOM_SEED)
    log("  mean", mean)
    log("  std", std)
    log("  BATCH_SIZE", BATCH_SIZE)
    log("  continue_from", continue_from)
    log("  CUDA available", torch.cuda.is_available())
    log("  ONE_BIAS", ONE_BIAS)

    # make logs and models directories if not there
    if not os.path.exists('logs'):
        os.makedirs('logs')
    if not os.path.exists('models'):
        os.makedirs('models')

    # Transformations for the dataset
    transform = Compose([
        Resize((RES_Y, RES_X))
        #ToTensor()
        #torchvision.transforms.Normalize(mean=mean.tolist(), std=std.tolist())  # Use calculated mean and std
    ])

    # Load the dataset
    dataset = FundusDataset(data_dir, transform=transform)
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(RANDOM_SEED))

    # Print the sizes of the training and validation datasets
    log(f"Training dataset size: {len(train_dataset)}")
    log(f"Validation dataset size: {len(val_dataset)}")

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
    log(f'Using device: {device}')

    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # Initialize model, loss function, and optimizer
    base_model_name = "models/model_file"
    if (continue_from):
        log("Continuing training...")
        base_model_name = "models/" + continue_from
        # if name ends with _torch, load with torch
        if continue_from.endswith('_torch'):
            model = torch.load(base_model_name, map_location=torch.device(device))
        else:
            model = pickle.load(open(base_model_name, 'rb')).to(device)
        log (f"Continuing training from {continue_from}")
    else:
        model = UNet(in_channels=4, out_channels=1).to(device)
        model.apply(init_weights)
        log (f"Starting new training")
    
    #loss_function = nn.BCELoss()
    #loss_function = nn.BCEWithLogitsLoss()

    pos_weight = torch.ones([1]) * ONE_BIAS  # Increase the weight to balance classes
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

        if(SAVE_PICKLE):
            pickle.dump(model, open(base_model_name + '_' + datetimestr + "_" + str(epoch+1), 'wb')) # Save the model 
        torch.save(model, base_model_name + '_' + datetimestr + "_" + str(epoch+1) + '_torch') # Save the model

        # Print loss, IoU, and Dice score after every epoch
        log(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, Dice: {val_dice:.4f}, Pixel Accuracy: {val_pixel:.4f}')
        # show current time
        log(f"Current Time = {datetime.now().strftime('%H:%M:%S')}")

    if(SAVE_PICKLE):
        pickle.dump(model, open(base_model_name + '_' + datetimestr + '_final', 'wb')) # Save the model 
    torch.save(model, base_model_name + '_' + datetimestr + "_final" + '_torch') # Save the model

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
    if PRINT:
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
    if (DOTEST):
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

        log(f'Test Loss: {val_loss:.4f}, IoU: {total_iou:.4f}, Dice: {total_dice:.4f}, Pixel Accuracy: {total_pixel_accuracy:.4f}')
    
if __name__ == "__main__":
    # get first command line argument
    if len(sys.argv) < 2:
        print("Usage: python3 ", sys.argv[0], " <continue_from> [DATA_DIR]")
        print("continue_from: 'restart' or <model_file> (a file in models/)")
        print("DATA_DIR: directory containing images and masks, default is " + DATA_DIR)
        sys.exit(1)
    continue_from = sys.argv[1]
    if continue_from == "restart":
        continue_from = False
    else:   
        # check if file exists in models/
        # if continue_from includes 'models/' prefix, remove it
        if continue_from.startswith("models/"):
            continue_from = continue_from[7:]
        if not os.path.isfile("models/" + continue_from):
            print("Model file does not exist: ", "models/" + continue_from)
            sys.exit(1)

    if len(sys.argv) > 2:
        DATA_DIR = sys.argv[2]

    main(LEARNING_RATE, NUM_EPOCHS, DATA_DIR, continue_from)