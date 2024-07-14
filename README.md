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

# 4. Install PyTorch with CUDA support:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Usage

### Training the Model

To train the U-Net model, run:

```bash
python model_training.py
```

## Running Inference

To perform inference and visualize the results, run:

```bash
python inference.py
```

The inference.py script processes images from the specified input directory, generates predicted masks, and displays the original images alongside their predicted masks.

## Preprocessing Data

To preprocess and save images and masks, run:

```bash
python preprocess_and_save_images_and_masks.py
```

## Saving Predictions to CSV

To save predicted masks to a CSV file, run:

```bash
python save_predictions_to_a_csv_file.py
```

## Custom Scripts

- `rle_decode.py`: Contains functions to decode RLE masks.
- `dice_coefficient.py`: Contains implementations of Dice coefficient and loss functions.

## Model Architecture
The U-Net model architecture is defined in unet_model.py. The pre-trained model is saved as `model.h5`.