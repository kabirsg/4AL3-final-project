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
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 15
#DATA_DIR = "eyedata/entire_dataset"
DATA_DIR = "eyedata/output_images"
RES_X = 640
RES_Y = 400
MASK_X = 32
MASK_Y = 20
DO_TEST = True
ONE_BIAS = 1

# Class that preprocesses the dataset
# Loads images and masks, applies transformations, and returns them as tensors
class FundusDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, device, include_orig=False):
        self.data_dir = data_dir
        self.device = device
        self.include_orig = include_orig

        # Find all images and masks
        self.image_files = {
            f: f for f in os.listdir(data_dir) if '_output_' in f and 'sobel_' in f and f.endswith('.png')
        }
        self.mask_files = {
            f: f for f in os.listdir(data_dir) if '_mask_' in f and f.endswith('.png')
        }
        if include_orig:
            self.orig_image_files = {
                f: f for f in os.listdir(data_dir) if '_output_' in f and 'sobel_' not in f and f.endswith('.png')
            }

        # Match images with their corresponding masks based on the shared prefix and suffix
        self.common_keys = []
        for img_name in self.image_files.keys():
            prefix_suffix = img_name.split('_output_')
            if len(prefix_suffix) == 2:
                mask_name = f"{prefix_suffix[0][6:]}_mask_{prefix_suffix[1]}"
                if mask_name not in self.mask_files:
                    print("Cannot find mask ",mask_name, " for ", img_name)
                    exit(1)
                if include_orig:
                    orig_name = img_name[6:]
                    if orig_name in self.orig_image_files:
                        self.common_keys.append((img_name, mask_name, orig_name))
                    else:
                        print("Cannot find original image ", orig_name, " for ", img_name)
                        exit(1)
                else:
                    self.common_keys.append((img_name, mask_name))

    def __len__(self):
        return len(self.common_keys)

    def __getitem__(self, idx):
        if (self.include_orig):
            img_name, mask_name, orig_name = self.common_keys[idx]
            orig_path = os.path.join(self.data_dir, orig_name)
        else:
            img_name, mask_name = self.common_keys[idx]

        img_path = os.path.join(self.data_dir, img_name)
        mask_path = os.path.join(self.data_dir, mask_name)

        print("opening image file ", img_path)
        image = Image.open(img_path).convert('L')
        np_image = np.array(image)
        image_t = torch.tensor(np_image, dtype=torch.uint8).float() / 256.0
        # show shape of image_t
        print("image_t ", image_t)

        print("opening mask file ", mask_path)
        mask = Image.open(mask_path).convert('L')
        mask = ToTensor()(mask)
        mask = (mask > 0.5).float()

        if (self.include_orig):
            orig_image = Image.open(orig_path).convert('RGBA')
            np_orig_image = np.array(orig_image)
            orig_image_t = torch.tensor(np_orig_image, dtype=torch.uint8).permute(2, 0, 1)
            return image_t, mask, orig_image_t

        return image_t, mask


class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, dropout_rate=0.5, do_norm=True):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=kernel_size//2)
        self.do_norm = do_norm
        if do_norm:
            self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=kernel_size, padding=kernel_size//2)
        if do_norm:
            self.bn2 = nn.BatchNorm2d(out_c)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.skip = nn.Conv2d(in_c, out_c, kernel_size=1) if in_c != out_c else nn.Identity()

    def forward(self, x): # resnet approach
        identity = self.skip(x)
        out = self.conv1(x)
        if self.do_norm:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        if self.do_norm:
            out = self.bn2(out)
        out = self.dropout2(out)
        out += identity
        out = self.relu2(out)
        return out
    
class ResNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(ResNet, self).__init__()
        
        def conv_block(in_c, out_c, kernel_size=3, do_norm=True):
            return ResidualBlock(in_c, out_c, kernel_size, do_norm)
        
        in_channels = 1  # Converted RGB to grayscale
        
        self.downsample1 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)      # Keeps the resolution
        self.downsample2 = nn.Conv2d(128, 128, kernel_size=2, stride=2, padding=0)    # Halves the resolution
        self.downsample3 = nn.Conv2d(256, 256, kernel_size=2, stride=2, padding=0)    # Halves the resolution
        self.downsample4 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)    # Keeps the resolution
        self.downsample5 = nn.Conv2d(512, 512, kernel_size=5, stride=5, padding=0)  # Fifths the resolution

        #self.encoder1 = conv_block(in_channels, 64, do_norm = False)  # Output: (64, 640, 400)
        self.encoder1 = conv_block(in_channels, 64)  # Output: (64, 640, 400)
        self.encoder2 = conv_block(64, 128)          # Output: (128, 320, 200)
        self.encoder3 = conv_block(128, 256)         # Output: (256, 160, 100)
        self.encoder4 = conv_block(256, 512)         # Output: (512, 160, 100)
        self.encoder5 = conv_block(512, 512, 5)      # Output: (512, 32, 20)

        # stride 1 will keep the resolution the same
        self.upconv1 = nn.ConvTranspose2d(512, 64, kernel_size=1, stride=1, padding=0, output_padding=0)
        self.decoder1 = conv_block(64, 64)           # Output: (64, 32, 20)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)  # Output: (1, 32, 20)
    

    def forward(self, x):

        def printDimension (str, x):
            #print(str, " ", x.size())
            pass

        e1 = self.encoder1(x)
        printDimension("e1", e1)
        e2 = self.encoder2(self.downsample1(e1))
        printDimension("e2", e2)
        e3 = self.encoder3(self.downsample2(e2))
        printDimension("e3", e3)
        e4 = self.encoder4(self.downsample3(e3))
        printDimension("e4", e4)
        e5 = self.encoder5(self.downsample4(e4))
        printDimension("e5", e5)
        e6 = self.downsample5(e5)
        printDimension("e6", e6)

        # we are not doing skip connections as we are changing resolutions
        # and it leads to overfitting on the pixel level (or something like that, maybe)
        d1 = self.upconv1(e6)
        d1 = self.decoder1(d1)
        printDimension("d1", d1)

        r = self.final_conv(d1)
        printDimension("final", r)
        return r


def train(model, train_loader, loss_function, optimizer, device):
    log("Training...")
    count = 0
    model.train()
    running_loss = 0.0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        
        # Forward pass
        starttime = datetime.now()
        print("starting model at ", starttime.strftime("%Y%m%d-%H%M%S"))
        outputs = model(images)
        endtime = datetime.now()
        print("model done at ", endtime.strftime("%Y%m%d-%H%M%S") + " took ", endtime - starttime)
        loss = loss_function(outputs, masks)

        # Backward pass and optimization
        optimizer.zero_grad()
        starttime = datetime.now()
        print("starting backward at ", starttime.strftime("%Y%m%d-%H%M%S"))
        loss.backward()
        endtime = datetime.now()
        print("backward done at ", endtime.strftime("%Y%m%d-%H%M%S") + " took ", endtime - starttime)
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
    total_fpr = 0.0
    count = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            print("starting model")
            outputs = model(images)
            print("model done")
            loss = loss_function(outputs, masks)
            val_loss += loss.item()
            total_iou += iou_score(outputs, masks).item()
            total_dice += dice_score(outputs, masks).item()
            total_pixel_accuracy += pixel_accuracy(outputs, masks).item()
            total_fpr += false_positive_rate(outputs, masks)
            count += 1
            log(f"Batch {count}, Loss: {loss.item():.4f}")
        
    num_batches = len(val_loader)
    return (
        val_loss / num_batches,
        total_iou / num_batches,
        total_dice / num_batches,
        total_pixel_accuracy / num_batches,
        total_fpr / num_batches, 
    )

def iou_score(preds_, targets, threshold=0.5):
    preds = (preds_ > threshold).float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    return intersection / union

def dice_score(preds_, targets, threshold=0.5):
    preds = (preds_ > threshold).float()
    intersection = (preds * targets).sum()
    return (2. * intersection) / (preds.sum() + targets.sum())

def pixel_accuracy(preds_, targets, threshold=0.5):
    preds = (preds_ > threshold).float()
    correct = (preds == targets).sum()
    total = targets.numel()
    return correct / total

def false_positive_rate(predictions_, targets):
    # Binarize predictions at a threshold of 0.5
    predictions = (predictions_ > 0.5).float()
    
    # Flatten tensors for computation
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    
    # Calculate False Positives (FP) and True Negatives (TN)
    fp = ((predictions == 1) & (targets == 0)).sum().item()
    tn = ((predictions == 0) & (targets == 0)).sum().item()
    
    # Avoid division by zero
    if fp + tn == 0:
        return 0.0
    
    return fp / (fp + tn)


datetimestr = datetime.now().strftime("%Y%m%d-%H%M%S")

def log(*args):
    print(*args)
    with open('logs/log_' + datetimestr + '.txt', 'a') as f:
        print(*args, file=f)

# Define the main function
def main(learning_rate, num_epochs, data_dir, continue_from):

    log("Current Timestamp =", datetimestr)

    log("Running ", sys.argv[0])
    log("  LEARNING_RATE", LEARNING_RATE)
    log("  NUM_EPOCHS", NUM_EPOCHS)
    log("  DATA_DIR", DATA_DIR)
    log("  RES_X", RES_X)
    log("  RES_Y", RES_Y)
    log("  MASK_X", MASK_X)
    log("  MASK_Y", MASK_Y)
    log("  DO_TEST", DO_TEST)
    log("  RANDOM_SEED", RANDOM_SEED)
    log("  BATCH_SIZE", BATCH_SIZE)
    log("  continue_from", continue_from)
    log("  CUDA available", torch.cuda.is_available())
    log("  ONE_BIAS", ONE_BIAS)

    # make logs and models directories if not there
    if not os.path.exists('logs'):
        os.makedirs('logs')
    if not os.path.exists('models'):
        os.makedirs('models')

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f'Using device: {device}')

    # Load the dataset
    dataset = FundusDataset(data_dir, device)
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(RANDOM_SEED))

    # Print the sizes of the training and validation datasets
    log(f"Training dataset size: {len(train_dataset)}")
    log(f"Validation dataset size: {len(val_dataset)}")

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

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
        model = ResNet(in_channels=4, out_channels=1).to(device)
        model.apply(init_weights)
        log (f"Starting new training")

    pos_weight = torch.ones([1]) * ONE_BIAS  # Increase the weight to balance classes
    loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training and validation
    train_losses, val_losses, iou_scores, dice_scores, pixel_accuracies, fprs = [], [], [], [], [], []
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, loss_function, optimizer, device)
        val_loss, val_iou, val_dice, val_pixel, fpr = validate(model, val_loader, loss_function, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        iou_scores.append(val_iou)
        dice_scores.append(val_dice)
        pixel_accuracies.append(val_pixel)
        fprs.append(fpr)

        if(SAVE_PICKLE):
            pickle.dump(model, open(base_model_name + '_' + datetimestr + "_" + str(epoch+1), 'wb')) # Save the model 
        torch.save(model, base_model_name + '_' + datetimestr + "_" + str(epoch+1) + '_torch') # Save the model

        # Print loss, IoU, and Dice score after every epoch
        log(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, Dice: {val_dice:.4f}, Pixel Accuracy: {val_pixel:.4f}, FPR: {fpr:.4f}')
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


    # Testing the model
    if (DO_TEST):
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        total_iou = 0.0 
        total_dice = 0.0
        total_pixel_accuracy = 0.0

        with torch.no_grad():
            for images, masks, _, _, _ in val_loader:
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

    # create models and logs directories if they don't already exist
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('logs'):
        os.makedirs('logs')

    main(LEARNING_RATE, NUM_EPOCHS, DATA_DIR, continue_from)