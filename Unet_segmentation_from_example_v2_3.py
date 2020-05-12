#image data generator - slow, but more augmentation

import os
from pathlib import Path
import datetime
from IPython.display import clear_output
import IPython.display as display
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
EPOCHS = 200
away_from_computer = True  # to show or not predictions between batches
save_model_for_inference = False # to save or not the model for inference
SEED = 3

# dataset location
Train_Images_Path = "D:/Python/DataSets/ADE20K_Filtered/Train/Images/0/"
Val_Images_Path =  "D:/Python/DataSets/ADE20K_Filtered/Validation/Images/0/"

Train_Images_Path_v2 = "D:/Python/DataSets/ADE20K_Filtered/Train/Images/"
Train_Masks_Path_v2 = "D:/Python/DataSets/ADE20K_Filtered/Train/New_Masks/"
Val_Images_Path_v2 =  "D:/Python/DataSets/ADE20K_Filtered/Validation/Images/"
Val_Masks_Path_v2 =  "D:/Python/DataSets/ADE20K_Filtered/Validation/New_Masks/"

data_gen_args = dict(
                    rotation_range=2, 
                    width_shift_range=15,
                    height_shift_range=15, 
                    # brightness_range=[-0.05, 0.05], 
                    zoom_range=0.2,
                    fill_mode='reflect', 
                    horizontal_flip=True,
                    vertical_flip=False, 
                    rescale=1./255, 
                    data_format="channels_last", validation_split=0.0, dtype=None)
Train_ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
Train_MaskDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)

train_image_dataset = Train_ImageDataGenerator.flow_from_directory(
    Train_Images_Path_v2,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='rgb',
    class_mode=None,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)
train_mask_dataset = Train_MaskDataGenerator.flow_from_directory(
    Train_Masks_Path_v2,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',
    class_mode=None,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

test_image_dataset = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, data_format="channels_last").flow_from_directory(
    Val_Images_Path_v2,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='rgb',
    class_mode=None,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)
test_mask_dataset = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, data_format="channels_last").flow_from_directory(
    Val_Masks_Path_v2,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',
    class_mode=None,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

train_dataset = zip(train_image_dataset, train_mask_dataset)
test_dataset = zip(test_image_dataset, test_mask_dataset)
# print(train_image_dataset)


# Visualizing the Loaded Dataset

def display_sample(display_list):
    # Show side-by-side an input image,
    # the ground truth and the prediction.
    
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


# for i in next(iter(train_image_dataset)):
#     for m in next(iter(train_mask_dataset)):
#         print (i.shape, m.shape)
#         display_sample([i, m])
#         break
#     break
# for i in next(iter(test_image_dataset)):
#     for m in next(iter(test_mask_dataset)):
#         print (i.shape, m.shape)
#         display_sample([i, m])
#         break
#     break
# display_sample(next(iter(train_dataset)))




train_imgs = glob(Train_Images_Path + "*.jpg")
val_imgs = glob(Val_Images_Path + "*.jpg")
TRAIN_LENGTH = len(train_imgs)
VAL_LENGTH = len(val_imgs)
print('train lenght: ', TRAIN_LENGTH)
print('val lenght: ', VAL_LENGTH)


BUFFER_SIZE = BATCH_SIZE
STEPS_PER_EPOCH = (TRAIN_LENGTH // BATCH_SIZE)
print('steps per epoch: ', STEPS_PER_EPOCH)



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
if os.path.exists("./Weights/U-net_128_16bit_model_initializer.h5"): 
    model.load_weights("./Weights/U-net_128_16bit_model_initializer.h5")
    print("Model loded - OK")

# show_predictions()

# This function keeps the learning rate at 0.001 for the first ten epochs
# and decreases it exponentially after that.
def scheduler(epoch):
  if epoch < 5:
    return 0.002
  else:
    return 0.001 * tf.math.exp(0.1 * (10 - epoch))

LRS = tf.keras.callbacks.LearningRateScheduler(scheduler)

#  - TensorBoard
data_folder = Path("c:/TFlogs/fit/")
log_dir=data_folder / datetime.datetime.now().strftime("%m%d-%H%M%S")  #folder for tensorboard
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_images=False,) #, profile_batch=2, histogram_freq=1, write_graph=True

class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    clear_output(wait=True)
    # show_predictions()
    # show_predictions(train_dataset, 1)
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))
    model.save_weights("./Weights/U-net_128_16bit_model.h5")



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

