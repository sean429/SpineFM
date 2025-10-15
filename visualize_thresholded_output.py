
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch

def binary_mask_from_logits(mask_logits, threshold=0.98):
    """Converts a logit mask tensor to a 2D binary numpy array."""
    if not isinstance(mask_logits, torch.Tensor):
        mask_logits = torch.as_tensor(mask_logits)
    
    binary_mask = (torch.sigmoid(mask_logits) > threshold).to(torch.uint8)
    
    if binary_mask.dim() == 3:
        binary_mask = binary_mask.squeeze(0)
        
    return binary_mask.numpy()

def visualize_thresholded_overlay(image_id, output_dir, gts_dir):
    """Visualizes all thresholded masks overlaid on the original image."""
    try:
        # --- Define Paths ---
        mask_path = output_dir / "masks" / f"{image_id}.npz"
        gt_image_path = gts_dir / f"{image_id}.npz"

        print(f"Loading original image from: {gt_image_path}")
        image_data = np.load(gt_image_path)
        base_image = image_data["image"]

        print(f"Loading mask logits from: {mask_path}")
        masks_data = np.load(mask_path)
        masks_logits = masks_data["masks"]

        # --- Create Visualization ---
        print("Creating visualization with threshold=0.98...")
        # Convert grayscale base image to a color image to allow for color overlays
        h, w = base_image.shape
        display_img = np.stack([base_image/base_image.max()]*3, axis=-1) # Normalize and convert to 3-channel

        colors = [
            (0, 0, 1.0), (0, 1.0, 0), (1.0, 0, 0), 
            (0, 1.0, 1.0), (1.0, 0, 1.0), (1.0, 1.0, 0)
        ]

        # --- Process and Overlay Each Mask ---
        for i, logit_mask in enumerate(masks_logits):
            binary_mask = binary_mask_from_logits(logit_mask, threshold=0.98)
            
            # Only apply overlay if the mask is not empty
            if binary_mask.sum() > 0:
                print(f"Overlaying mask #{i} with color {colors[i % len(colors)]}")
                # Apply color with transparency (alpha blending)
                color_overlay = np.array(colors[i % len(colors)])
                # Find pixels where the mask is 1
                mask_pixels = binary_mask == 1
                # Blend color
                display_img[mask_pixels] = (display_img[mask_pixels] * 0.5 + color_overlay * 0.5)

        # --- Display and Save ---
        plt.figure(figsize=(10, 12))
        plt.imshow(display_img)
        plt.title(f"Thresholded (0.98) Mask Overlay for {image_id}")
        plt.axis("off")
        
        save_path = f"threshold_overlay_{image_id}.png"
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        print(f"\nVisualization saved to {save_path}")

    except FileNotFoundError as e:
        print(f"Error: Could not find a required file. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    IMAGE_ID = "0195127"
    OUTPUT_DIR = Path("data/csxa/new_output")
    GTS_DIR = Path("data/csxa/gts")
    
    visualize_thresholded_overlay(IMAGE_ID, OUTPUT_DIR, GTS_DIR)
