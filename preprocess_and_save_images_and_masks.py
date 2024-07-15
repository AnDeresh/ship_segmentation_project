import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm

from config import *
from rle_decode import rle_decode

# Reload the CSV file
df = pd.read_csv(train_csv_path)

# Create directories if they don't exist
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(train_mask_dir, exist_ok=True)
os.makedirs(val_mask_dir, exist_ok=True)

# Split data into train and validation sets
image_ids = df['ImageId'].unique()
train_ids, val_ids = train_test_split(image_ids, test_size=0.2, random_state=42)

# Function to save images and masks
def save_images_and_masks(image_ids, image_dir, mask_dir):
    existing_images = []
    missing_images = []
    
    for image_id in tqdm(image_ids, desc=f"Processing images"):
        image_path = os.path.join(train_dir, image_id)
        
        if not os.path.exists(image_path):
            missing_images.append(image_id)
            continue
        
        try:
            img = Image.open(image_path)
            img.verify()  # Verify that the file is not corrupted
            img = Image.open(image_path)  # Reopen the file after verification
            
            existing_images.append(image_id)
            mask = np.zeros((768, 768), dtype=np.uint8)
            image_masks = df[df['ImageId'] == image_id]['EncodedPixels'].tolist()
            
            for mask_rle in image_masks:
                if mask_rle is not np.nan:
                    mask += rle_decode(mask_rle)
            
            mask = np.clip(mask, 0, 1)  # Ensure mask is binary

            # Save image
            img.save(os.path.join(image_dir, image_id))
            
            # Save mask
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
            mask_img.save(os.path.join(mask_dir, image_id))
        except (OSError, Image.DecompressionBombError) as e:
            print(f"Skipping corrupted image: {image_id}, error: {e}")
            missing_images.append(image_id)
        except Exception as e:
            print(f"Error processing image: {image_id}, error: {e}")
            missing_images.append(image_id)
    
    return existing_images, missing_images

# Save train and validation images and masks
train_existing, train_missing = save_images_and_masks(train_ids, train_image_dir, train_mask_dir)
val_existing, val_missing = save_images_and_masks(val_ids, val_image_dir, val_mask_dir)

print(
    {
    "train_image_dir": train_image_dir,
    "train_mask_dir": train_mask_dir,
    "val_image_dir": val_image_dir,
    "val_mask_dir": val_mask_dir,
    "train_existing": len(train_existing),
    "train_missing": len(train_missing),
    "val_existing": len(val_existing),
    "val_missing": len(val_missing)
}
)