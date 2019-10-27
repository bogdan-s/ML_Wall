from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
# import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
print(tf.__version__)

from keras.preprocessing.image import ImageDataGenerator

# from keras.applications.resnet50 import ResNet50
# from keras.preprocessing import image
# from keras.applications.resnet50 import preprocess_input, decode_predictions

# From KERAS.IO:
# Example of transforming images and masks together.
# we create two instances with the same arguments
data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=10,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1
image_datagen.fit(images, augment=False, seed=seed)
mask_datagen.fit(masks, augment=False, seed=seed)

image_generator = image_datagen.flow_from_directory(
    'D:/Python/DataSets/ADE20K_Filtered/Train/Images',
    class_mode=None,
    seed=seed)

mask_generator = mask_datagen.flow_from_directory(
    'D:/Python/DataSets/ADE20K_Filtered/Train/Masks',
    class_mode=None,
    seed=seed)

# combine generators into one which yields image and masks
train_generator = zip(image_generator, mask_generator)

model.fit_generator(
    train_generator,
    steps_per_epoch=2000,
    epochs=50)


