import os

train_dir = 'E:/data2/train_v2' # Directory with images for model training
output_dir = 'E:/data2/processed_data' # Directory for processed images (train and val)
train_output_dir = os.path.join(output_dir, 'train')
val_output_dir = os.path.join(output_dir, 'val')
train_image_dir = os.path.join(train_output_dir, 'images')
val_image_dir = os.path.join(val_output_dir, 'images')
train_mask_dir = os.path.join(train_output_dir, 'masks')
val_mask_dir = os.path.join(val_output_dir, 'masks')

model_save = 'E:/data2/model.h5'

output_csv_path = 'E:/projects/winstars_test_task/submissions'

train_csv_path = r'train_ship_segmentations_v2.csv'

model_path = 'E:/projects/winstars_test_task/winstars_test_task/model.h5' # Path for model

input_image_dir = 'E:/data/test_v2_50'  # Directory with input images
output_dir = 'E:/data/test_v2_pred'  # Directory to save results