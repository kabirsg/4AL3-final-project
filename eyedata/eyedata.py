import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import random
import matplotlib.pyplot as plt

# Constants for image sizes
ARTIFACT_MIN_ALPHA = 50
ARTIFACT_MAX_ALPHA = 150
ARTIFACT_MIN_COUNT = 15
ARTIFACT_MAX_COUNT = 30
OUTPUT_WIDTH = 640
OUTPUT_HEIGHT = 400
MASK_WIDTH = 32
MASK_HEIGHT = 20
ZOOM_MIN = 1.0
ZOOM_MAX = 3.0
ARTIFACT_MIN_DIAMETER = 128
ARTIFACT_MAX_DIAMETER = 400

DEBUG = False

def display_debug(image, mask, stage):
    if DEBUG:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1, frame_on=True)
        plt.imshow(image)
        plt.gca().spines['top'].set_linewidth(2)
        plt.gca().spines['bottom'].set_linewidth(2)
        plt.gca().spines['left'].set_linewidth(2)
        plt.gca().spines['right'].set_linewidth(2)
        plt.title(f"Image - {stage}")
        plt.axis('off')

        plt.subplot(1, 2, 2, frame_on=True)
        plt.imshow(mask, cmap='gray')
        plt.gca().spines['top'].set_linewidth(2)
        plt.gca().spines['bottom'].set_linewidth(2)
        plt.gca().spines['left'].set_linewidth(2)
        plt.gca().spines['right'].set_linewidth(2)
        plt.title(f"Mask - {stage}")
        plt.axis('off')

        plt.show()

def generate_artifact_map(width, height):
    mask = np.ones((height, width), dtype=np.uint8) * 0
    circle_radius = height // 2
    circle_center = (width // 2, height // 2)
    y, x = np.ogrid[:height, :width]
    distance = (x - circle_center[0])**2 + (y - circle_center[1])**2
    mask[distance > circle_radius**2] = 255
    return mask

def process_image(image_path, output_image_path, output_mask_path):
    # Load and resize the image
    image = Image.open(image_path).convert("RGBA")
    image = image.resize((OUTPUT_WIDTH, OUTPUT_HEIGHT), Image.Resampling.LANCZOS)
    
    # Create the initial artifact mask
    mask = generate_artifact_map(OUTPUT_WIDTH, OUTPUT_HEIGHT)
    display_debug(image, mask, "Initial Artifact Mask")

    # Rotate the image and mask randomly (keeping original size)
    angle = random.uniform(0, 360)
    image = image.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=(0, 0, 0, 255))
    mask = Image.fromarray(mask).rotate(angle, resample=Image.Resampling.NEAREST, fillcolor=255)
    mask = np.array(mask)
    display_debug(image, mask, "After Rotation")

    # Random zoom on the image and mask
    zoom_factor = random.uniform(ZOOM_MIN, ZOOM_MAX)
    width, height = image.size
    new_width, new_height = int(width / zoom_factor), int(height / zoom_factor)
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    image = image.crop((left, top, right, bottom)).resize((OUTPUT_WIDTH, OUTPUT_HEIGHT), Image.Resampling.LANCZOS)
    mask = Image.fromarray(mask).crop((left, top, right, bottom)).resize((OUTPUT_WIDTH, OUTPUT_HEIGHT), Image.Resampling.NEAREST)
    mask = np.array(mask)
    display_debug(image, mask, "After Zoom")

    # Apply random circular mask to black out regions outside of the circle
    circle_diameter = random.randint(100, OUTPUT_HEIGHT)
    circle_radius = circle_diameter // 2
    circle_x = random.randint(circle_radius // 2, OUTPUT_WIDTH - circle_radius // 2)
    circle_y = random.randint(circle_radius // 2, OUTPUT_HEIGHT - circle_radius // 2)
    y, x = np.ogrid[:OUTPUT_HEIGHT, :OUTPUT_WIDTH]
    distance = (x - circle_x)**2 + (y - circle_y)**2
    circular_mask = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH), dtype=np.uint8)
    circular_mask[distance > circle_radius**2] = 255

    # Black out parts of the image outside the circular mask
    # Sample color from the non-masked part of the image for the background
    non_masked_pixels = np.where(mask == 0)
    if len(non_masked_pixels[0]) > 0:
        idx = random.randint(0, len(non_masked_pixels[0]) - 1)
        color_sample_x, color_sample_y = non_masked_pixels[1][idx], non_masked_pixels[0][idx]
        sampled_color = image.getpixel((color_sample_x, color_sample_y))
        background_color = (sampled_color[0], sampled_color[1], sampled_color[2], 255)
    else:
        # Fallback color if no non-masked pixels are found
        background_color = (0, 0, 0, 255)
    black_background = Image.new("RGBA", (OUTPUT_WIDTH, OUTPUT_HEIGHT), background_color)
    inverse_circular_mask_image = Image.fromarray(255 - circular_mask).convert("L")
    image = Image.composite(image, black_background, inverse_circular_mask_image)
    

    # Update the mask with the circular mask
    mask = np.maximum(mask, circular_mask)
    display_debug(image, mask, "After Circular Mask")

    # Add random artifacts (ellipses and blobs sampled from the good part of the image)
    draw = ImageDraw.Draw(image, "RGBA")
    for _ in range(random.randint(ARTIFACT_MIN_COUNT, ARTIFACT_MAX_COUNT)):
        artifact_type = 'ellipse'
        while True:
            x0, y0 = random.randint(0, OUTPUT_WIDTH), random.randint(0, OUTPUT_HEIGHT)
            x1, y1 = x0 + random.randint(ARTIFACT_MIN_DIAMETER, ARTIFACT_MAX_DIAMETER), y0 + random.randint(ARTIFACT_MIN_DIAMETER, ARTIFACT_MAX_DIAMETER)
            artifact_radius = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 / 2
            if (x0 - circle_x) ** 2 + (y0 - circle_y) ** 2 >= (circle_radius + artifact_radius / 2) ** 2 and (x1 - circle_x) ** 2 + (y1 - circle_y) ** 2 >= (circle_radius + artifact_radius / 2) ** 2:
               break
        # Sample color from the non-masked part of the image
        non_masked_pixels = np.where(mask == 0)
        if len(non_masked_pixels[0]) > 0:
            idx = random.randint(0, len(non_masked_pixels[0]) - 1)
            color_sample_x, color_sample_y = non_masked_pixels[1][idx], non_masked_pixels[0][idx]
            sampled_color = image.getpixel((color_sample_x, color_sample_y))
            r, g, b, _ = sampled_color
            a = random.randint(ARTIFACT_MIN_ALPHA, ARTIFACT_MAX_ALPHA)
            # Occasionally make the color completely random
            if random.random() < 0.1:
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), a)
            # Occasionally make the color close to black
            elif random.random() < 0.1:
                color = (random.randint(0, 50), random.randint(0, 50), random.randint(0, 50), a)
            else:
                color = (r, g, b, a)   
            a = random.randint(ARTIFACT_MIN_ALPHA, ARTIFACT_MAX_ALPHA)
            color = (r, g, b, a)
        else:
            # Fallback color if no non-masked pixels are found
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), random.randint(50, 150))
        (r, g, b, a) = color
          
        for scale in range(6):  # Create multiple smaller ellipses within the same area
            scale_factor = 1.0 - scale * 0.1  # Reduce size gradually
            offset_x = random.randint(-5, 5)  # Randomize position slightly for non-concentric ellipses
            offset_y = random.randint(-5, 5)
            scaled_x0 = int(x0 + (1 - scale_factor) * (x1 - x0) / 2) + offset_x
            scaled_y0 = int(y0 + (1 - scale_factor) * (y1 - y0) / 2) + offset_y
            scaled_x1 = int(x1 - (1 - scale_factor) * (x1 - x0) / 2) + offset_x
            scaled_y1 = int(y1 - (1 - scale_factor) * (y1 - y0) / 2) + offset_y
            r_variation = random.randint(-10, 10)
            g_variation = random.randint(-10, 10)
            b_variation = random.randint(-10, 10)
            new_color = (
                min(max(r + r_variation, 0), 255),
                min(max(g + g_variation, 0), 255),
                min(max(b + b_variation, 0), 255),
                a
            )
            draw.ellipse([scaled_x0, scaled_y0, scaled_x1, scaled_y1], fill=new_color)
            blurred_artifact = Image.new("RGBA", image.size, (0, 0, 0, 0))
            artifact_draw = ImageDraw.Draw(blurred_artifact)
            artifact_draw.ellipse([scaled_x0, scaled_y0, scaled_x1, scaled_y1], fill=new_color)
            blurred_artifact = blurred_artifact.filter(ImageFilter.GaussianBlur(5))
            image = Image.alpha_composite(image, blurred_artifact)

        # Update the mask to indicate artifact areas
        artifact_mask = Image.fromarray(mask)
        mask_draw = ImageDraw.Draw(artifact_mask)
        mask_draw.ellipse([x0, y0, x1, y1], fill=255)
        mask = np.maximum(mask, np.array(artifact_mask))

        # Update the mask to indicate artifact areas
        artifact_mask = Image.fromarray(mask)
        mask_draw = ImageDraw.Draw(artifact_mask)
        mask_draw.ellipse([x0, y0, x1, y1], fill=255)
        mask = np.maximum(mask, np.array(artifact_mask))
    display_debug(image, mask, "After Adding Artifacts")

    # Blur only the artifacts with feathered edges
    feathered_mask = Image.fromarray(mask).filter(ImageFilter.GaussianBlur(40)).convert("L")
    blurred_image = image.filter(ImageFilter.GaussianBlur(random.uniform(3, 10)))
    image = Image.composite(blurred_image, image, feathered_mask)
    display_debug(image, mask, "After Blurring Artifacts")

    # Save the final image and mask
    image.convert("RGB").save(output_image_path, format="JPEG")
    mask_image = Image.fromarray(mask).resize((MASK_WIDTH, MASK_HEIGHT), Image.Resampling.BOX)
    mask_image_array = np.array(mask_image)
    # Remove all white pixels if the count is below a threshold
    white_pixel_count = np.sum(mask_image_array == 255)
    if white_pixel_count < 6:
        mask_image_array[mask_image_array == 255] = 0
    mask_resized = (mask_image_array == 0).astype(np.uint8) * 255
    mask_image = Image.fromarray(mask_resized)
    display_debug(image, mask_image, "Resized Mask Image")
    mask_image.save(output_mask_path)
    # Save the mask data as a text file
    mask_text_path = output_mask_path.replace('.png', '.txt')
    mask_binary = (mask_image_array == 0).astype(np.uint8)
    np.savetxt(mask_text_path, mask_binary, fmt='%d')

def main(input_dir, output_dir, num_outputs_per_image):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        # Remove any existing images in the output directory
        for file in os.listdir(output_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.txt')):
                os.remove(os.path.join(output_dir, file))

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Processing image: {filename}")
            input_path = os.path.join(input_dir, filename)
            base_name = os.path.splitext(filename)[0]
            for i in range(num_outputs_per_image):
                output_image_path = os.path.join(output_dir, f"{base_name}_output_{i + 1}.jpg")
                output_mask_path = os.path.join(output_dir, f"{base_name}_mask_{i + 1}.png")
                process_image(input_path, output_image_path, output_mask_path)
                print(f"   Saved output image: {output_image_path}")

if __name__ == "__main__":
    main("input_images", "output_images", 50)
