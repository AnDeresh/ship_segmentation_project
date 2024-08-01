from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint # type: ignore
import os
from typing import List, Tuple

from scripts.generate_data import *
from models.unet import *
from configs.config import *
from models.dice_coefficient import *

def prepare_data_generators(train_image_dir: str, train_mask_dir: str, val_image_dir: str, val_mask_dir: str, batch_size: int = 16, image_size: Tuple[int, int] = (128, 128)) -> Tuple[DataGenerator, DataGenerator]:
    train_ids = os.listdir(train_image_dir)
    val_ids = os.listdir(val_image_dir)
    
    train_generator = DataGenerator(train_ids, train_image_dir, train_mask_dir, batch_size=batch_size, image_size=image_size)
    val_generator = DataGenerator(val_ids, val_image_dir, val_mask_dir, batch_size=batch_size, image_size=image_size)
    
    return train_generator, val_generator

def setup_callbacks(model_checkpoint_path: str) -> List:
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',    
        factor=0.5,            
        patience=1,            
        min_lr=1e-6,           
        verbose=1              
    )

    checkpoint_callback = ModelCheckpoint(
        filepath=model_checkpoint_path,
        monitor='val_loss',
        save_best_only=True,  
        save_weights_only=False,
        mode='auto',
        save_freq='epoch'
    )
    
    return [reduce_lr, checkpoint_callback]

def compile_model(input_size: Tuple[int, int, int] = (128, 128, 3)) -> tf.keras.Model:
    model = unet(input_size=input_size)
    model.compile(optimizer='adam', loss=combined_loss, metrics=[dice_coefficient])
    return model

def train_model(model: tf.keras.Model, train_generator: DataGenerator, val_generator: DataGenerator, callbacks: List, epochs: int = 5) -> tf.keras.callbacks.History:
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks
    )
    return history

def evaluate_model(model: tf.keras.Model, val_generator: DataGenerator):
    val_loss, val_dice = model.evaluate(val_generator)
    print(f"Validation Loss: {val_loss}")
    print(f"Validation Dice Coefficient: {val_dice}")

def main():
    train_generator, val_generator = prepare_data_generators(TRAIN_IMAGE_DIR, TRAIN_MASK_DIR, VAL_IMAGE_DIR, TRAIN_MASK_DIR)
    callbacks = setup_callbacks('model_checkpoint.h5')
    model = compile_model()
    history = train_model(model, train_generator, val_generator, callbacks)
    model.save(MODEL_PATH)
    evaluate_model(model, val_generator)

if __name__ == "__main__":
    main()
