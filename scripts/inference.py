import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
from PIL import Image
import os
import cv2
import pandas as pd
from typing import List, Tuple

from models.dice_coefficient import custom_objects
from configs.config import model_path, input_image_dir, output_dir, output_csv_path

# Load the pre-trained model
model = load_model(model_path, custom_objects=custom_objects)

def load_and_preprocess_image(image_path: str, target_size: Tuple[int, int] = (128, 128)) -> np.ndarray:
    """
    Load and preprocess an image.

    Args:
        image_path (str): Path to the image file.
        target_size (Tuple[int, int]): Target size for the image.

    Returns:
        np.ndarray: Preprocessed image ready for prediction.
    """
    image = load_img(image_path, color_mode='rgb', target_size=target_size)  # Load image in RGB mode
    image = img_to_array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Expand dimensions to fit model input
    return image

def rle_encode(mask: np.ndarray) -> str:
    """
    Encode mask to RLE (Run-Length Encoding).

    Args:
        mask (np.ndarray): Binary mask.

    Returns:
        str: RLE encoded string.
    """
    pixels = mask.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def find_min_area_rect(mask: np.ndarray, threshold: float = 0.5) -> List[np.ndarray]:
    """
    Find minimum area rectangles from the mask.

    Args:
        mask (np.ndarray): Binary mask.
        threshold (float): Threshold value for binarizing the mask.

    Returns:
        List[np.ndarray]: List of rectangle points.
    """
    contours, _ = cv2.findContours((mask > threshold).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.minAreaRect(contour) for contour in contours]
    boxes = [cv2.boxPoints(rect) for rect in rects]
    boxes = [np.int0(box) for box in boxes]
    return boxes

def draw_rectangles(image_size: Tuple[int, int], boxes: List[np.ndarray]) -> np.ndarray:
    """
    Draw solid rectangles on the mask.

    Args:
        image_size (Tuple[int, int]): Size of the image/mask.
        boxes (List[np.ndarray]): List of rectangle points.

    Returns:
        np.ndarray: Mask with drawn rectangles.
    """
    mask = np.zeros(image_size, dtype=np.uint8)
    for box in boxes:
        cv2.drawContours(mask, [box], 0, (255), -1)  # Use -1 to fill the rectangle
    return mask

def predict_and_convert_to_rect(image_path: str) -> np.ndarray:
    """
    Predict and convert the mask to rectangular regions.

    Args:
        image_path (str): Path to the input image.

    Returns:
        np.ndarray: Rectangular mask.
    """
    image = load_and_preprocess_image(image_path)
    prediction = model.predict(image)[0, :, :, 0]
    prediction_resized = cv2.resize(prediction, (768, 768), interpolation=cv2.INTER_NEAREST)
    rects = find_min_area_rect(prediction_resized, threshold=0.5)
    rectangular_mask = draw_rectangles((768, 768), rects)
    return rectangular_mask

def main():
    results = []

    for image_filename in os.listdir(input_image_dir):
        if image_filename.endswith('.png') or image_filename.endswith('.jpg'):
            image_path = os.path.join(input_image_dir, image_filename)
            try:
                rectangular_mask = predict_and_convert_to_rect(image_path)
                predicted_mask_image = Image.fromarray(rectangular_mask)
                output_image_path = os.path.join(output_dir, f'mask_{image_filename}')
                predicted_mask_image.save(output_image_path)
                rle = rle_encode(rectangular_mask)
                results.append({'ImageId': image_filename, 'EncodedPixels': rle})
            except Exception as e:
                print(f"Error processing image {image_filename}: {e}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_csv_path, 'output_predictions.csv'), index=False)

if __name__ == "__main__":
    main()
