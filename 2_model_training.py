from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import os

from data_generator import *
from unet_model import *
from config import *
from dice_coefficient import *

# Prepare data generators
train_ids = os.listdir(train_image_dir)
val_ids = os.listdir(val_image_dir)

train_generator = DataGenerator(train_ids, train_image_dir, train_mask_dir)
val_generator = DataGenerator(val_ids, val_image_dir, val_mask_dir)

model = unet(input_size=(128, 128, 3))

# Define the learning rate scheduler callback
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',    # The metric to monitor
    factor=0.5,            # The factor by which the learning rate will be reduced
    patience=1,            # Number of epochs with no improvement before reducing learning rate
    min_lr=1e-6,           # The minimum learning rate
    verbose=1              # Print messages when reducing learning rate
)

# Define the model checkpoint callback
checkpoint_callback = ModelCheckpoint(
    filepath='model_checkpoint.h5',
    monitor='val_loss',
    save_best_only=False,  # Set to True to only save the best model
    save_weights_only=False,
    mode='auto',
    save_freq='epoch'
)

# Compile the model
model.compile(optimizer='adam', loss=combined_loss, metrics=[dice_coefficient])

# Train the model with the checkpoint callback
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5,
    callbacks=[checkpoint_callback]
)

# Save the model to a file
model.save(model_save)

# Evaluate the model
val_loss, val_dice = model.evaluate(val_generator)
print(f"Validation Loss: {val_loss}")
print(f"Validation Dice Coefficient: {val_dice}")