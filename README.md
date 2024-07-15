# Winstars Test Task

This repository contains the code for the Winstars test task. The project involves model training, inference, and evaluation for ship segmentation using U-Net. The main components of this project are structured into various Python scripts and notebooks.

## Project Structure

- `config.py`: Configuration file for the project settings.
- `data_generator.py`: Script to generate data for training and validation.
- `dice_coefficient.py`: Implementation of the Dice coefficient and loss functions.
- `EDA.ipynb`: Notebook for Exploratory Data Analysis.
- `inference.py`: Script for running inference and visualizing results.
- `model_training.py`: Script for training the U-Net model.
- `model.h5`: Pre-trained U-Net model file.
- `preprocess_and_save_images_and_masks.py`: Script to preprocess and save images and masks.
- `requirements.txt`: List of required Python packages.
- `rle_decode.py`: Script to decode Run-Length Encoding (RLE) masks.
- `save_predictions_to_a_csv_file.py`: Script to save predictions to a CSV file.
- `train_ship_segmentations_v2.csv`: CSV file with training data.
- `unet_model.py`: Script containing the U-Net model architecture.

## Setup Instructions

### Prerequisites

Ensure you have the following installed:
- Python 3.8+
- pip (Python package installer)

### Installation

```bash
# 1. Clone the repository:
git clone https://github.com/yourusername/winstars_test_task.git
cd winstars_test_task

# 2. Create and activate a virtual environment:
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# 3. Install the required packages:
pip install -r requirements.txt

## Usage

1. Configure model path in `config`.
2. Run the script to generate and save prediction masks and rectangles.
```

#### Data
- Data can be downloaded from the [Kaggle Bank Customer Churn Prediction competition](https://www.kaggle.com/competitions/bank-customer-churn-prediction-dlu/data).

### Training the Model

To train the U-Net model, run:

```bash
python model_training.py
```

#### Overview
Trains a U-Net model for image segmentation using specified data generators, with learning rate scheduling and model checkpointing.

#### Functions and Key Components
- `ReduceLROnPlateau`: Reduces learning rate when validation loss plateaus.
- `ModelCheckpoint`: Saves model checkpoints during training.
- `DataGenerator`: Prepares training and validation data.
- `unet(input_size)`: Defines the U-Net model architecture.
- `model.compile()`: Compiles the model with the Adam optimizer and custom loss/metrics.
- `model.fit()`: Trains the model.
- `model.save()`: Saves the trained model.
- `model.evaluate()`: Evaluates the model on the validation set.

#### Usage
1. Ensure paths and parameters are set in `config`.
2. Run the script to train the model, save checkpoints, and evaluate performance.

## Running Inference

To perform inference and visualize the results, run:

```bash
python inference_make_prediction_masks_and_csv.py
```

Generates prediction masks using a pre-trained model, converts them to rectangular regions, and saves masks and results in CSV format.

#### Functions
- `load_and_preprocess_image(image_path, target_size=(128, 128))`: Loads and preprocesses an image.
- `rle_encode(mask)`: Encodes a mask with Run-Length Encoding (RLE).
- `find_min_area_rect(mask, threshold=0.5)`: Finds minimum area rectangles in the mask.
- `draw_rectangles(image_size, boxes)`: Draws rectangles on the mask.
- `predict_and_convert_to_rect(image_path)`: Predicts and processes mask from an image.

## Preprocessing Data

To preprocess and save images and masks, run:

```bash
python preprocess_and_save_images_and_masks.py
```

#### Functions
- `rle_decode(mask_rle, shape=(768, 768))`: Decodes an RLE mask.
- `save_images_and_masks(image_ids, image_dir, mask_dir)`: Saves images and masks, handles corrupted files.


## Custom Scripts

- `rle_decode.py`: Contains functions to decode RLE masks.
- `dice_coefficient.py`: Contains implementations of Dice coefficient and loss functions.

## Model Architecture
The U-Net model architecture is defined in unet_model.py. The pre-trained model is saved as `model.h5`.