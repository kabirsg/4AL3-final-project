import os
import sys
print(sys.executable)
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

RES_X = 640
RES_Y = 400
MASK_X = 32
MASK_Y = 20

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Class that applies the Sobel filter to an image
class SobelFilter(nn.Module):
    def __init__(self):
        super(SobelFilter, self).__init__()
        # Define Sobel kernels
        sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.register_buffer('sobel_kernel_x', sobel_kernel_x)
        self.register_buffer('sobel_kernel_y', sobel_kernel_y)

    def forward(self, x):
        sobel_kernel_x = self.sobel_kernel_x.to(x.device)
        sobel_kernel_y = self.sobel_kernel_y.to(x.device)

        # Apply reflect padding to the input to avoid border effects
        x_padded = F.pad(x, (1, 1, 1, 1), mode='reflect')

        # Apply Sobel filter in x and y directions
        #print("x ", x.size())
        #print("sobel_kernel_x ", self.sobel_kernel_x.size())
        grad_x = F.conv2d(x_padded, sobel_kernel_x, padding=0)
        #print("grad_x ", grad_x.size())
        grad_y = F.conv2d(x_padded, sobel_kernel_y, padding=0)
        # Combine gradients
        grad = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        sobel_output = grad * 8;
        # Change any values greater than 1 to 1
        clamped_sobel_output = torch.clamp(sobel_output, 0, 1)
        
        return clamped_sobel_output


# Class that preprocesses the dataset
# Loads images and masks, applies transformations, and returns them as tensors
class FundusDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, device, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.sobel_filter = SobelFilter()
        self.device = device

        # Find all images and masks
        self.image_files = {
            f: f for f in os.listdir(data_dir) if '_output_' in f and f.endswith('.png')
        }
        self.mask_files = {
            f: f for f in os.listdir(data_dir) if '_mask_' in f and f.endswith('.png')
        }

        # Match images with their corresponding masks based on the shared prefix and suffix
        self.common_keys = []
        for img_name in self.image_files.keys():
            prefix_suffix = img_name.split('_output_')
            if len(prefix_suffix) == 2:
                mask_name = f"{prefix_suffix[0]}_mask_{prefix_suffix[1]}"
                if mask_name in self.mask_files:
                    self.common_keys.append((img_name, mask_name))


    def __len__(self):
        return len(self.common_keys)

    def __getitem__(self, idx):
        img_name, mask_name = self.common_keys[idx]
        img_path = os.path.join(self.data_dir, img_name)
        mask_path = os.path.join(self.data_dir, mask_name)

        image = Image.open(img_path).convert('RGBA')
        original_image = np.array(image)

        mask = Image.open(mask_path).convert('L')

        # Resize the image
        transformed_image = self.transform(image)
        print("transformed_image ", transformed_image.size)

        # Normalize the image
        image_array = np.array(transformed_image, dtype=np.float32)  # Convert to NumPy array for processing
        image_min, image_max = image_array.min(), image_array.max()
        normalized_image = (image_array - image_min) / (image_max - image_min + 1e-8)
        normalized_image = torch.tensor(normalized_image, dtype=torch.float32).permute(2, 0, 1)  # Convert back to Tensor

        normalized_image_device = normalized_image.to(self.device)

        # Convert Sobel image to greyscale
        greyscale_image_device = normalized_image_device.mean(dim=0, keepdim=True)

        # Apply Sobel filter
        sobel_image_device = self.sobel_filter(greyscale_image_device.unsqueeze(0)).squeeze(0)  # Add batch dimension for Sobel filter

        sobel_image = sobel_image_device.cpu()
        print("sobel_image ", sobel_image.size())
        
        # Apply resizing and transforms to mask
        mask = Resize((MASK_Y, MASK_X))(mask) 
        mask = ToTensor()(mask)
        mask = (mask > 0.5).float()

        # dimension of sobel image
        print("sobel_image ", sobel_image.size())

        return sobel_image, mask, self.common_keys[idx]
    

def main(in_data_dir, out_data_dir):
    print("Preprocessing images in", in_data_dir)
    print("Outputting to", out_data_dir)

    # Create output directory if it doesn't exist
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    # Transformations for the dataset
    transform = Compose([
        Resize((RES_Y, RES_X))
    ])
    dataset = FundusDataset(in_data_dir, device, transform)

    for i in range(len(dataset)):
        sobel_image, mask, names = dataset[i]
        print("sobel_image ", sobel_image.size())
        sobel_image = sobel_image.cpu().numpy()
        print("sobel_image numpy", sobel_image.shape)
        mask = mask.cpu().numpy()

        # Transpose the array to (640, 400, 1)
        #sobel_image = np.transpose(sobel_image, (2, 1, 0))
        #mask = np.transpose(mask, (2, 1, 0))
        print("sobel_image transposed", sobel_image.shape)

        sobel_image = (sobel_image * 255).astype(np.uint8).squeeze()
        mask = (mask * 255).astype(np.uint8).squeeze()
        print("sobel_image numpy * 255", sobel_image.shape)
        
        sobel_image = Image.fromarray(sobel_image)
        mask = Image.fromarray(mask)

        

        sobel_image.save(os.path.join(out_data_dir, f"sobel_{names[0]}"))
        mask.save(os.path.join(out_data_dir, f"{names[1]}"))


if __name__ == "__main__":
    # Get first command line argument
    if len(sys.argv) < 2:
        print("Usage: python3 ", sys.argv[0], " [SRC_DATA_DIR]")
        print("DATA_DIR: directory containing images and masks, default is " + DATA_DIR)
        sys.exit(1)
    DATA_DIR = sys.argv[1]
    output_dir = DATA_DIR + "_prep"
    # If dir doesn't exist, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    main(DATA_DIR, output_dir)