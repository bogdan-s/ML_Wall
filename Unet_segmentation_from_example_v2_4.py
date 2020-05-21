#tf.data - fast but less realtime augmentation

import os
from pathlib import Path
import datetime
# from IPython.display import clear_output
# import IPython.display as display
from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, Dropout, concatenate, UpSampling2D
import pix2pix
import tensorflow_addons as tfa

tf.keras.backend.clear_session()

from tensorflow.keras.mixed_precision import experimental as mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

#tf.logging.set_verbosity(tf.logging.ERROR)  #hide info
# from tensorflow_examples.models.pix2pix import pix2pix

AUTOTUNE = tf.data.experimental.AUTOTUNE

print(tf.__version__, end='\n\n')
print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))
# param
IMG_SIZE = 128
BATCH_SIZE = 64
OUTPUT_CHANNELS = 2
EPOCHS = 10
away_from_computer = True  # to show or not predictions between batches
save_model_for_inference = False # to save or not the model for inference
SEED = 2

# dataset location
Train_Images_Path = "D:/Python/DataSets/ADE20K_Filtered/Train/Images/0/"
Val_Images_Path =  "D:/Python/DataSets/ADE20K_Filtered/Validation/Images/0/"

# similar to glob but with tensorflow
train_imgs = tf.data.Dataset.list_files(Train_Images_Path + "*.jpg", shuffle=False)
val_imgs = tf.data.Dataset.list_files(Val_Images_Path + "*.jpg", shuffle=False)

@tf.function
def parse_image(img_path):
    """Load an image and its annotation (mask) and returning
    a dictionary.

    Parameters
    ----------
    img_path : str
        Image (not the mask) location.

    Returns
    -------
    dict
        Dictionary mapping an image and its annotation.
    """
    image = tf.io.read_file(img_path)
    image = tf.io.decode_jpeg(image, channels=3)
    # image = tf.io.convert_image_dtype(image, tf.uint8)
    
    mask_path = tf.strings.regex_replace(img_path, "Images", "New_Masks")
    mask_path = tf.strings.regex_replace(mask_path, ".jpg", "_seg.png")
    mask = tf.io.read_file(mask_path)
    mask = tf.io.decode_png(mask, channels=0, dtype=tf.dtypes.uint8)
    print(mask.shape)
    
    return {'image': image, 'segmentation_mask' : mask}

train_set = train_imgs.map(parse_image, num_parallel_calls=AUTOTUNE)

test_set = val_imgs.map(parse_image, num_parallel_calls=AUTOTUNE)
dataset = {"train": train_set, "test": test_set}

print(dataset.keys())

# first I create the function to normalize, resize and apply some data augmentation on my dataset:

@tf.function
def normalize(input_image: tf.Tensor, input_mask: tf.Tensor) -> tuple:
    """Rescale the pixel values of the images between 0.0 and 1.0
    compared to [0,255] originally.

    Parameters
    ----------
    input_image : tf.Tensor
        Tensorflow tensor containing an image of IMG_SIZE [IMG_SIZE,IMG_SIZE,3].
    input_mask : tf.Tensor
        Tensorflow tensor containing an annotation of IMG_SIZE [IMG_SIZE,IMG_SIZE,1].

    Returns
    -------
    tuple
        Normalized image and its annotation.
    """
    input_image = tf.cast(input_image, tf.float16) / 255.0
    input_mask = tf.cast(input_mask, tf.uint8) / 255
    return input_image, input_mask

@tf.function
def load_image_train(datapoint: dict) -> tuple:
    """Apply some transformations to an input dictionary
    containing a train image and its annotation.

    Notes
    -----
    An annotation is a regular  channel image.
    If a transformation such as rotation is applied to the image,
    the same transformation has to be applied on the annotation also.

    Parameters
    ----------
    datapoint : dict
        A dict containing an image and its annotation.

    Returns
    -------
    tuple
        A modified image and its annotation.
    """
    input_image = tf.image.resize(datapoint['image'], (150, 150), method='area')
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (150, 150), method='area') #, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)
    
    # input_image, input_mask = normalize(input_image, input_mask)

    input_image = input_image / 255
    input_mask = tf.image.rgb_to_grayscale(input_mask)
    input_mask = input_mask / 255
    # print("Shape: {}".format(input_image.shape))
    # print("Shape: {}".format(input_mask.shape))
    return input_image, input_mask

@tf.function
def train_random_crop(crop_image, crop_mask) -> tuple:
    print("Shape image before crop: {}".format(crop_image.shape))
    print("Shape mask before crop: {}".format(crop_mask.shape))
    crop_image = tf.image.random_crop(crop_image, size = [IMG_SIZE, IMG_SIZE, 3])
    crop_mask = tf.image.random_crop(crop_mask, size = [IMG_SIZE, IMG_SIZE, 1])
    print("Shape image after crop: {}".format(crop_image.shape))
    print("Shape mask after crop: {}".format(crop_mask.shape))    
    return crop_image, crop_mask


@tf.function
def load_image_test(datapoint: dict) -> tuple:
    """Normalize and resize a test image and its annotation.

    Notes
    -----
    Since this is for the test set, we don't need to apply
    any data augmentation technique.

    Parameters
    ----------
    datapoint : dict
        A dict containing an image and its annotation.

    Returns
    -------
    tuple
        A modified image and its annotation.
    """
    input_image = tf.image.resize(datapoint['image'], (IMG_SIZE, IMG_SIZE), method='area')
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (IMG_SIZE, IMG_SIZE), method='area')

    # input_image, input_mask = normalize(input_image, input_mask)
    input_image = input_image / 255
    input_mask = tf.image.rgb_to_grayscale(input_mask)
    input_mask = input_mask / 255

    return input_image, input_mask

# Then I set some parameters related to my dataset:

train_imgs = glob(Train_Images_Path + "*.jpg")
val_imgs = glob(Val_Images_Path + "*.jpg")
TRAIN_LENGTH = len(train_imgs)
VAL_LENGTH = len(val_imgs)
print('train lenght: ', TRAIN_LENGTH)
print('val lenght: ', VAL_LENGTH)


BUFFER_SIZE = BATCH_SIZE
STEPS_PER_EPOCH = (TRAIN_LENGTH // BATCH_SIZE)
print('steps per epoch: ', STEPS_PER_EPOCH)

train = dataset['train'].map(load_image_train, num_parallel_calls=AUTOTUNE)
# test = dataset['test'].map(load_image_test)
# train = train.concatenate(test)

# print(tf.data.Dataset.cardinality(train).numpy())        # how big is the dataset
# print(tf.data.Dataset.cardinality(test).numpy())


train_dataset = train.cache().shuffle(buffer_size=TRAIN_LENGTH, seed=SEED, reshuffle_each_iteration=True)
train_dataset = train_dataset.map(train_random_crop, num_parallel_calls=AUTOTUNE)                                                       #  apply random_crop | if disabled adjust image resize in load_image_train
train_dataset = train_dataset.batch(BATCH_SIZE).repeat() #
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
# train_dataset = train_dataset.cache().shuffle(BATCH_SIZE, reshuffle_each_iteration=True).repeat().prefetch(buffer_size=AUTOTUNE)  #buffer_size=AUTOTUNE



test = dataset['test'].map(load_image_test)
test_dataset = test.batch(BATCH_SIZE)
test_dataset = test_dataset.cache().repeat()


# Visualizing the Loaded Dataset

def display_sample(display_list):
    """Show side-by-side an input image,
    the ground truth and the prediction.
    """
    plt.figure(figsize=(18, 18))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        print("Shape: {}".format(display_list[i].shape))
        print("Type: {}".format(display_list[i].dtype))
        print("Mean: {}".format(tf.math.reduce_mean(display_list[i])))
        
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

# Test dataset images and masks

for image, mask in train_dataset.take(1):
    sample_image, sample_mask = image[0], mask[0]
    print("Shape image[0] dataset.take(1): {}".format(sample_image[0].shape))
    print("Shape mask[0] dataset.take(1): {}".format(sample_mask[0].shape))     
#     # print(tf.reduce_min(sample_mask), tf.reduce_mean(sample_mask), tf.reduce_max(sample_mask))
#     # print(sample_image.shape)
#     print(sample_mask.shape)
#     t1d = tf.reshape(sample_mask, shape=(-1,)) # create a 1D tensor
#     print(t1d.shape)
#     uniques, _ = tf.unique(t1d)   # check for unique values to see how the mask was resized
#     print(uniques)
#     display_sample([sample_image, sample_mask])

# print('train daset: ', train)



SIZE = IMG_SIZE

OUTPUT_CHANNELS = 2

# Model from https://yann-leguilly.gitlab.io/post/2019-12-14-tensorflow-tfdata-segmentation/

# -- Keras Functional API -- #
# -- UNet Implementation -- #
# Everything here is from tensorflow.keras.layers
# I imported tensorflow.keras.layers * to make it easier to read
dropout_rate = 0.5
input_size = (IMG_SIZE, IMG_SIZE, 3)

# If you want to know more about why we are using `he_normal`: 
# https://stats.stackexchange.com/questions/319323/whats-the-difference-between-variance-scaling-initializer-and-xavier-initialize/319849#319849  
# Or the excelent fastai course: 
# https://github.com/fastai/course-v3/blob/master/nbs/dl2/02b_initializing.ipynb
# initializer = 'he_normal'
initializer = tf.keras.initializers.he_normal(seed=SEED)

# -- Encoder -- #
# Block encoder 1
inputs = Input(shape=input_size)
conv_enc_1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=initializer)(inputs)
conv_enc_1 = Conv2D(32, 3, activation = 'relu', padding='same', kernel_initializer=initializer)(conv_enc_1)

# Block encoder 2
max_pool_enc_2 = MaxPooling2D(pool_size=(2, 2))(conv_enc_1)
conv_enc_2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(max_pool_enc_2)
conv_enc_2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_enc_2)

# Block  encoder 3
max_pool_enc_3 = MaxPooling2D(pool_size=(2, 2))(conv_enc_2)
conv_enc_3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(max_pool_enc_3)
conv_enc_3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_enc_3)

# Block  encoder 4
max_pool_enc_4 = MaxPooling2D(pool_size=(2, 2))(conv_enc_3)
conv_enc_4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(max_pool_enc_4)
conv_enc_4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_enc_4)
# -- Encoder -- #

# ----------- #
maxpool = MaxPooling2D(pool_size=(2, 2))(conv_enc_4)
conv = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(maxpool)
conv = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv)
# ----------- #

# -- Dencoder -- #
# Block decoder 1
up_dec_1 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2,2))(conv))
merge_dec_1 = concatenate([conv_enc_4, up_dec_1], axis = 3)
conv_dec_1 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge_dec_1)
conv_dec_1 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_dec_1)

# Block decoder 2
up_dec_2 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2,2))(conv_dec_1))
merge_dec_2 = concatenate([conv_enc_3, up_dec_2], axis = 3)
conv_dec_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge_dec_2)
conv_dec_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_dec_2)

# Block decoder 3
up_dec_3 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2,2))(conv_dec_2))
merge_dec_3 = concatenate([conv_enc_2, up_dec_3], axis = 3)
conv_dec_3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge_dec_3)
conv_dec_3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_dec_3)

# Block decoder 4
up_dec_4 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2,2))(conv_dec_3))
merge_dec_4 = concatenate([conv_enc_1, up_dec_4], axis = 3)
conv_dec_4 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge_dec_4)
conv_dec_4 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_dec_4)
conv_dec_4 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_dec_4)
# -- Dencoder -- #

output = Conv2D(OUTPUT_CHANNELS, 1, activation = 'sigmoid')(conv_dec_4)


model = tf.keras.Model(inputs = inputs, outputs = output)



class MaskMeanIoU(tf.keras.metrics.MeanIoU):
    #                                                                                                    Mean Intersection over Union
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight=sample_weight)



optimizer_Adam = tf.keras.optimizers.Adam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
    name='Adam')

# model = unet_model(OUTPUT_CHANNELS)
model.compile(optimizer=optimizer_Adam,
              # loss=tfa.losses.GIoULoss(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), #from_logits=True
              # metrics=[tf.keras.metrics.Accuracy()])
              metrics=['accuracy', MaskMeanIoU(name='iou', num_classes=OUTPUT_CHANNELS)])

# model.summary()


# tf.keras.utils.plot_model(model, show_shapes=True)                                                                                                #plot model

def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]


def show_predictions(dataset=None, num=1):
  if dataset:
    for image, mask in dataset.take(num):
      pred_mask = model.predict(image)
      display_sample([image[0], mask[0], create_mask(pred_mask)])
  else:
    display_sample([sample_image, sample_mask,
             create_mask(model.predict(sample_image[tf.newaxis, ...]))])

#                                                                                                                                          load weights from last save
# if os.path.exists("./Weights/U-net_128_16bit_model_initializer.h5"): 
#     model.load_weights("./Weights/U-net_128_16bit_model_initializer.h5")
#     print("Model loded - OK")

# show_predictions()

# This function keeps the learning rate at 0.001 for the first ten epochs
# and decreases it exponentially after that.
def scheduler(epoch):
  if epoch < 5:
    return 0.001
  else:
    return 0.001 * tf.math.exp(0.1 * (10 - epoch))

LRS = tf.keras.callbacks.LearningRateScheduler(scheduler)

#  - TensorBoard
data_folder = Path("c:/TFlogs/fit/")
log_dir=data_folder / datetime.datetime.now().strftime("%m%d-%H%M%S")  #folder for tensorboard
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_images=True, profile_batch='10,50', write_graph=True) #, profile_batch=2, histogram_freq=1, write_graph=True

class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    # clear_output(wait=True)
    # show_predictions()
    show_predictions(train_dataset, 1)
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))
    # model.save_weights("./Weights/U-net_128_16bit_model.h5")



VALIDATION_STEPS = VAL_LENGTH // BATCH_SIZE

model_history = model.fit(train_dataset, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_dataset,
                          callbacks=[DisplayCallback(), tensorboard_callback])  #LRS,


loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

# epochs = range(EPOCHS)

# plt.figure()
# plt.plot(epochs, loss, 'r', label='Training loss')
# plt.plot(epochs, val_loss, 'bo', label='Validation loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss Value')
# plt.ylim([0, 1])
# plt.legend()
# plt.show()

# show_predictions(train_dataset, 3)
# show_predictions(test_dataset, 3)

