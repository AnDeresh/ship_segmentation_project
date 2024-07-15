import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
from PIL import Image
import os
import cv2
import pandas as pd

from dice_coefficient import *
from config import *

# Load the pre-trained model
model = load_model(model_path, custom_objects=custom_objects)

# Function to load and preprocess an image
def load_and_preprocess_image(image_path, target_size=(128, 128)):
    image = load_img(image_path, color_mode='rgb', target_size=target_size)  # Load image in RGB mode
    image = img_to_array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Expand dimensions to fit model input
    return image

# Function to encode mask to RLE (Run-Length Encoding)
def rle_encode(mask):
    pixels = mask.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# Function to find minimum area rectangles from the mask
def find_min_area_rect(mask, threshold=0.5):
    contours, _ = cv2.findContours((mask > threshold).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.minAreaRect(contour) for contour in contours]
    boxes = [cv2.boxPoints(rect) for rect in rects]
    boxes = [np.int0(box) for box in boxes]
    return boxes

# Function to draw solid rectangles on the mask
def draw_rectangles(image_size, boxes):
    mask = np.zeros(image_size, dtype=np.uint8)
    for box in boxes:
        cv2.drawContours(mask, [box], 0, (255), -1)  # Use -1 to fill the rectangle
    return mask

# Function to predict and convert the mask to rectangular regions
def predict_and_convert_to_rect(image_path):
    # Load and preprocess the image
    image = load_and_preprocess_image(image_path)

    # Predict the mask
    prediction = model.predict(image)[0, :, :, 0]

    # Resize the prediction to the original size
    prediction_resized = cv2.resize(prediction, (768, 768), interpolation=cv2.INTER_NEAREST)

    # Get the rectangular masks
    rects = find_min_area_rect(prediction_resized, threshold=0.5)
    rectangular_mask = draw_rectangles((768, 768), rects)

    return rectangular_mask

# Main code to process images and save masks and results to CSV
results = []

for image_filename in os.listdir(input_image_dir):
    if image_filename.endswith('.png') or image_filename.endswith('.jpg'):
        image_path = os.path.join(input_image_dir, image_filename)
        
        # Predict and convert to rectangular masks
        rectangular_mask = predict_and_convert_to_rect(image_path)
        
        # Save the predicted rectangular mask
        predicted_mask_image = Image.fromarray(rectangular_mask)
        output_image_path = os.path.join(output_dir, f'mask_{image_filename}')
        predicted_mask_image.save(output_image_path)
        
        # Convert the rectangular mask to RLE encoding and add to results
        rle = rle_encode(rectangular_mask)
        results.append({'ImageId': image_filename, 'EncodedPixels': rle})

# Save the results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(output_csv_path, 'output_predictions.csv'), index=False)