from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import Sequence
import numpy as np
import os

class DataGenerator(Sequence):
    def __init__(self, image_ids, image_dir, mask_dir, batch_size=32, image_size=(128, 128), n_channels=3, shuffle=True):
        self.image_ids = image_ids
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.image_ids) / self.batch_size))

    def __getitem__(self, index):
        batch_ids = self.image_ids[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__data_generation(batch_ids)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.image_ids)

    def __data_generation(self, batch_ids):
        X = np.empty((self.batch_size, *self.image_size, self.n_channels))
        y = np.empty((self.batch_size, *self.image_size, 1))

        for i, id in enumerate(batch_ids):
            image = load_img(os.path.join(self.image_dir, id), target_size=self.image_size)
            image = img_to_array(image) / 255.0
            mask = load_img(os.path.join(self.mask_dir, id), target_size=self.image_size, color_mode="grayscale")
            mask = img_to_array(mask) / 255.0

            X[i,] = image
            y[i,] = mask

        return X, y