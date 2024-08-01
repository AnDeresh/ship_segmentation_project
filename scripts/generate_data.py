from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
from tensorflow.keras.utils import Sequence # type: ignore
import numpy as np
import os
from typing import List, Tuple, Any

class DataGenerator(Sequence):
    def __init__(self, image_ids: List[str], image_dir: str, mask_dir: str, batch_size: int = 16, 
                 image_size: Tuple[int, int] = (128, 128), n_channels: int = 3, shuffle: bool = True) -> None:
        """
        Initialization of DataGenerator
        """
        self.image_ids = image_ids
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self) -> int:
        """
        Denotes the number of batches per epoch
        """
        return int(np.floor(len(self.image_ids) / self.batch_size))

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate one batch of data
        """
        batch_ids = self.image_ids[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__data_generation(batch_ids)
        return X, y

    def on_epoch_end(self) -> None:
        """
        Updates indexes after each epoch
        """
        if self.shuffle:
            np.random.shuffle(self.image_ids)

    def __data_generation(self, batch_ids: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates data containing batch_size samples
        """
        X = np.empty((self.batch_size, *self.image_size, self.n_channels))
        y = np.empty((self.batch_size, *self.image_size, 1))

        for i, id in enumerate(batch_ids):
            try:
                image = self.__load_image(id)
                mask = self.__load_mask(id)
                X[i,] = image
                y[i,] = mask
            except Exception as e:
                print(f"Error loading {id}: {e}")
                continue

        return X, y

    def __load_image(self, image_id: str) -> np.ndarray:
        """
        Load and preprocess image
        """
        image_path = os.path.join(self.image_dir, image_id)
        image = load_img(image_path, target_size=self.image_size)
        return img_to_array(image) / 255.0

    def __load_mask(self, image_id: str) -> np.ndarray:
        """
        Load and preprocess mask
        """
        mask_path = os.path.join(self.mask_dir, image_id)
        mask = load_img(mask_path, target_size=self.image_size, color_mode="grayscale")
        return img_to_array(mask) / 255.0