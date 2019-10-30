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



# From KERAS.IO:
# Example of transforming images and masks together.
# we create two instances with the same arguments
data_gen_args = dict(
                    rescale=1./255,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True)

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1

# Train & Validation images and masks
train_image_generator = image_datagen.flow_from_directory(
    '../DataSets/ADE20K_Filtered/Train/Images',
    target_size=(150, 150),
    batch_size=32,
    class_mode=None,
    shuffle=True,
    seed=seed)
val_image_generator = image_datagen.flow_from_directory(
    '../DataSets/ADE20K_Filtered/Validation/Images',
    target_size=(150, 150),
    batch_size=32,
    class_mode=None,
    shuffle=True,
    seed=seed)
train_mask_generator = mask_datagen.flow_from_directory(
    '../DataSets/ADE20K_Filtered/Train/Masks',
    target_size=(150, 150),
    batch_size=32,
    class_mode=None,
    shuffle=True,
    seed=seed)
val_mask_generator = mask_datagen.flow_from_directory(
    '../DataSets/ADE20K_Filtered/Validation/Masks',
    target_size=(150, 150),
    batch_size=32,
    class_mode=None,
    shuffle=True,
    seed=seed)

# combine generators into one which yields image and masks
train_generator = zip(train_image_generator, train_mask_generator)
val_generator = zip(val_image_generator, val_mask_generator)











# model.fit_generator(
#     train_generator,
#     steps_per_epoch=2000,
#     epochs=50)


