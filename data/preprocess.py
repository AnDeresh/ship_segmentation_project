import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm
from typing import List, Tuple

from configs.config import *
from utils.rle import rle_decode

def load_data(csv_path: str) -> pd.DataFrame:
    """Load data from CSV file."""
    return pd.read_csv(csv_path)

def create_directories(directories: List[str]):
    """Create directories if they don't exist."""
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[List[str], List[str]]:
    """Split data into train and validation sets."""
    image_ids = df['ImageId'].unique()
    return train_test_split(image_ids, test_size=test_size, random_state=random_state)

def save_images_and_masks(image_ids: List[str], image_dir: str, mask_dir: str, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Save images and masks to specified directories."""
    existing_images = []
    missing_images = []
    
    for image_id in tqdm(image_ids, desc="Processing images"):
        image_path = os.path.join(TRAIN_DIR, image_id)
        
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

def main():
    # Load data
    df = load_data(TRAIN_CSV_PATH)
    
    # Create necessary directories
    create_directories([TRAIN_IMAGE_DIR, VAL_IMAGE_DIR, TRAIN_MASK_DIR, VAL_MASK_DIR])
    
    # Split data
    train_ids, val_ids = split_data(df)
    
    # Save train and validation images and masks
    train_existing, train_missing = save_images_and_masks(train_ids, TRAIN_IMAGE_DIR, TRAIN_MASK_DIR, df)
    val_existing, val_missing = save_images_and_masks(val_ids, VAL_IMAGE_DIR, VAL_MASK_DIR, df)
    
    # Print summary
    print({
        "train_image_dir": TRAIN_IMAGE_DIR,
        "train_mask_dir": TRAIN_MASK_DIR,
        "val_image_dir": VAL_IMAGE_DIR,
        "val_mask_dir": VAL_MASK_DIR,
        "train_existing": len(train_existing),
        "train_missing": len(train_missing),
        "val_existing": len(val_existing),
        "val_missing": len(val_missing)
    })

if __name__ == "__main__":
    main()