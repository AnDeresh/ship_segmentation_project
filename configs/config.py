import os

def get_env_variable(var_name, description, default_value=None):
    """Get the environment variable or return the default value.
    If default_value is None, use the description as a guide to set the variable."""
    value = os.getenv(var_name)
    if value is None and default_value is not None:
        return default_value
    elif value is None:
        raise ValueError(f"The environment variable {var_name} is not set. It should be {description}.")
    return value

# Directories
TRAIN_DIR = get_env_variable('TRAIN_DIR', 'the directory where the training data is stored')
OUTPUT_DIR = get_env_variable('OUTPUT_DIR', 'the directory where the processed data will be stored')
TRAIN_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'train')
VAL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'val')
TRAIN_IMAGE_DIR = os.path.join(TRAIN_OUTPUT_DIR, 'images')
VAL_IMAGE_DIR = os.path.join(VAL_OUTPUT_DIR, 'images')
TRAIN_MASK_DIR = os.path.join(TRAIN_OUTPUT_DIR, 'masks')
VAL_MASK_DIR = os.path.join(VAL_OUTPUT_DIR, 'masks')

# Model paths
MODEL_SAVE_PATH = get_env_variable('MODEL_SAVE_PATH', 'the path where the model will be saved')
MODEL_PATH = get_env_variable('MODEL_PATH', 'the path where the model is stored')

# CSV paths
OUTPUT_CSV_PATH = get_env_variable('OUTPUT_CSV_PATH', 'the path where the output CSV will be stored')
TRAIN_CSV_PATH = get_env_variable('TRAIN_CSV_PATH', 'the path to the training CSV file', 'train_ship_segmentations_v2.csv')

# Input/Output directories for predictions
INPUT_IMAGE_DIR = get_env_variable('INPUT_IMAGE_DIR', 'the directory containing images for prediction')
PREDICTION_OUTPUT_DIR = get_env_variable('PREDICTION_OUTPUT_DIR', 'the directory where prediction results will be stored')

# Validate directories
for path in [TRAIN_DIR, OUTPUT_DIR, TRAIN_OUTPUT_DIR, VAL_OUTPUT_DIR, TRAIN_IMAGE_DIR, VAL_IMAGE_DIR, TRAIN_MASK_DIR, VAL_MASK_DIR, INPUT_IMAGE_DIR, PREDICTION_OUTPUT_DIR]:
    if not os.path.exists(path):
        os.makedirs(path)