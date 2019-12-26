from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

import tensorflow_datasets as tfds
print(tf.__version__)



# From KERAS.IO:
# Example of transforming images and masks together.
# we create two instances with the same arguments
# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1
# data_gen_args = dict(
#                     rescale=1./255,
#                     shear_range=0.2,
#                     zoom_range=0.2,
#                     horizontal_flip=True)

data_gen_args = dict(
                    rescale=1./255,
                    )

image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)

img_size = 512                          # resize al images to this size

# Train & Validation images and masks
train_image_generator = image_datagen.flow_from_directory(
    '../DataSets/ADE20K_Filtered/Train/Images',
    target_size=(img_size, img_size),
    batch_size=32,
    class_mode=None,
    shuffle=True,
    seed=seed)
val_image_generator = image_datagen.flow_from_directory(
    '../DataSets/ADE20K_Filtered/Validation/Images',
    target_size=(img_size, img_size),
    batch_size=32,
    class_mode=None,
    shuffle=True,
    seed=seed)
train_mask_generator = mask_datagen.flow_from_directory(
    '../DataSets/ADE20K_Filtered/Train/Masks',
    target_size=(img_size, img_size),
    batch_size=32,
    class_mode=None,
    shuffle=True,
    seed=seed)
val_mask_generator = mask_datagen.flow_from_directory(
    '../DataSets/ADE20K_Filtered/Validation/Masks',
    target_size=(img_size, img_size),
    batch_size=32,
    class_mode=None,
    shuffle=True,
    seed=seed)

# combine generators into one which yields image and masks
train_generator = zip(train_image_generator, train_mask_generator)
val_generator = zip(val_image_generator, val_mask_generator)

print(train_image_generator.__getitem__(1).shape)

# train = dataset['train'].map(train_generator, num_parallel_calls=tf.data.experimental.AUTOTUNE)
# test = dataset['test'].map(val_generator)



def display(display_list):
  for i in range(len(display_list)):
    plt.subplot((int(len(preview_list) / 6) + (len(preview_list) % 6 > 0)), 6, i+1)
    plt.imshow(display_list[i])
    plt.axis('off')
  plt.show()

# display(train_image_generator.__getitem__(1))
input_image_arr = train_image_generator.__getitem__(2)
input_mask_arr = train_mask_generator.__getitem__(2)

print(input_image_arr[1].shape)
print(input_mask_arr[1].shape)

preview_list = []

for i in range(15):                              #how many groups of image - mask to preview
    preview_list.append(input_image_arr[i])
    preview_list.append(input_mask_arr[i])

display(preview_list)


# Create DatasetFromGenerator

ds = tf.data.Dataset.from_generator(
    train_image_generator, args=None, 
    output_types=(tf.float32, tf.float32), 
    output_shapes=([32,img_size,img_size,3], [32,img_size,img_size,3])
)

ds




'''
OUTPUT_CHANNELS = 2

base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
layers = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

down_stack.trainable = False

# base_model.summary()


def upsample(filters, size, norm_type='batchnorm', apply_dropout=False):
  """Upsamples an input.
  Conv2DTranspose => Batchnorm => Dropout => Relu
  Args:
    filters: number of filters
    size: filter size
    norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
    apply_dropout: If True, adds the dropout layer
  Returns:
    Upsample Sequential Model
  """

  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))

  if norm_type.lower() == 'batchnorm':
    result.add(tf.keras.layers.BatchNormalization())
  elif norm_type.lower() == 'instancenorm':
    result.add(InstanceNormalization())

  if apply_dropout:
    result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result


up_stack = [
    upsample(512, 3),  # 4x4 -> 8x8
    upsample(256, 3),  # 8x8 -> 16x16
    upsample(128, 3),  # 16x16 -> 32x32
    upsample(64, 3),   # 32x32 -> 64x64
]


def unet_model(output_channels):

  # This is the last layer of the model
  last = tf.keras.layers.Conv2DTranspose(
      output_channels, 3, strides=2,
      padding='same', activation='softmax')  #64x64 -> 128x128

  inputs = tf.keras.layers.Input(shape=[128, 128, 3])
  x = inputs

  # Downsampling through the model
  skips = down_stack(x)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

model = unet_model(OUTPUT_CHANNELS)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# model.summary()                

def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]


def show_predictions(dataset=None, num=1):
  if dataset:
    for image, mask in dataset.take(num):
      pred_mask = model.predict(image)
      display([image[0], mask[0], create_mask(pred_mask)])
  else:
    display([sample_image, sample_mask,
             create_mask(model.predict(sample_image[tf.newaxis, ...]))])

show_predictions()
'''




