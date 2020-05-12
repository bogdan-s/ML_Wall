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


from tensorflow.keras.mixed_precision import experimental as mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

#tf.logging.set_verbosity(tf.logging.ERROR)  #hide info
# from tensorflow_examples.models.pix2pix import pix2pix

AUTOTUNE = tf.data.experimental.AUTOTUNE

print(tf.__version__, end='\n\n')

# param
IMG_SIZE = 512
BATCH_SIZE = 64
OUTPUT_CHANNELS = 2
EPOCHS = 20
away_from_computer = True  # to show or not predictions between batches
save_model_for_inference = False # to save or not the model for inference


# dataset location
Train_Images_Path = "D:/Python/DataSets/ADE20K_Filtered/Train/Images/0/"
Val_Images_Path =  "D:/Python/DataSets/ADE20K_Filtered/Validation/Images/0/"

# similar to glob but with tensorflow
train_imgs = tf.data.Dataset.list_files(Train_Images_Path + "*.jpg")
val_imgs = tf.data.Dataset.list_files(Val_Images_Path + "*.jpg")

@tf.function
def parse_image(img_path):
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.uint8)
    
    mask_path = tf.strings.regex_replace(img_path, "Images", "New_Masks")
    mask_path = tf.strings.regex_replace(mask_path, ".jpg", "_seg.png")
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    
    return {'image': image, 'segmentation_mask' : mask}

train_set = train_imgs.map(parse_image)
test_set = val_imgs.map(parse_image)
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
    input_image = tf.image.resize(datapoint['image'], (IMG_SIZE, IMG_SIZE))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (IMG_SIZE, IMG_SIZE)) #, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

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
    input_image = tf.image.resize(datapoint['image'], (IMG_SIZE, IMG_SIZE))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (IMG_SIZE, IMG_SIZE))

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

# Then I set some parameters related to my dataset:

train_imgs = glob(Train_Images_Path + "*.jpg")
val_imgs = glob(Val_Images_Path + "*.jpg")
TRAIN_LENGTH = len(train_imgs)
VAL_LENGTH = len(val_imgs)
print('train lenght: ', TRAIN_LENGTH)
print('val lenght: ', VAL_LENGTH)


BUFFER_SIZE = BATCH_SIZE
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
print('steps per epoch: ', STEPS_PER_EPOCH)

train = dataset['train'].map(load_image_train, num_parallel_calls=AUTOTUNE)


train_dataset = train.batch(BATCH_SIZE).cache().repeat().prefetch(buffer_size=AUTOTUNE)  #.shuffle(BUFFER_SIZE)
# train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

test = dataset['test'].map(load_image_test)
test_dataset = test.batch(BATCH_SIZE)


#                                                                                                               Visualizing the Loaded Dataset

def display_sample(display_list):
    """Show side-by-side an input image,
    the ground truth and the prediction.
    """
    plt.figure(figsize=(18, 18))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

#                                                                                                               Test dataset images and masks

# for image, mask in train.take(1):
    # sample_image, sample_mask = image, mask
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

# Model from https://github.com/rudolfsteiner/fast_scnn/blob/master/Fast_SCNN.py

def down_sample(input_layer):
    
    ds_layer = tf.keras.layers.Conv2D(32, (3,3), padding='same', strides = (2,2))(input_layer)
    ds_layer = tf.keras.layers.BatchNormalization()(ds_layer)
    ds_layer = tf.keras.activations.relu(ds_layer)
    
    ds_layer = tf.keras.layers.SeparableConv2D(48, (3,3), padding='same', strides = (2,2))(ds_layer)
    ds_layer = tf.keras.layers.BatchNormalization()(ds_layer)
    ds_layer = tf.keras.activations.relu(ds_layer)
    
    ds_layer = tf.keras.layers.SeparableConv2D(64, (3,3), padding='same', strides = (2,2))(ds_layer)
    ds_layer = tf.keras.layers.BatchNormalization()(ds_layer)
    ds_layer = tf.keras.activations.relu(ds_layer)
    
    return ds_layer
    

def _res_bottleneck(inputs, filters, kernel, t, s, r=False):
    
    
    tchannel = tf.keras.backend.int_shape(inputs)[-1] * t

    #x = conv_block(inputs, 'conv', tchannel, (1, 1), strides=(1, 1))
    x = tf.keras.layers.Conv2D(tchannel, (1,1), padding='same', strides = (1,1))(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)

    x = tf.keras.layers.DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)

    #x = #conv_block(x, 'conv', filters, (1, 1), strides=(1, 1), padding='same', relu=False)
    
    x = tf.keras.layers.Conv2D(filters, (1,1), padding='same', strides = (1,1))(x)
    x = tf.keras.layers.BatchNormalization()(x)


    if r:
        x = tf.keras.layers.add([x, inputs])
    return x

"""#### Bottleneck custom method"""

def bottleneck_block(inputs, filters, kernel, t, strides, n):
    x = _res_bottleneck(inputs, filters, kernel, t, strides)

    for i in range(1, n):
        x = _res_bottleneck(x, filters, kernel, t, 1, True)

        return x

def global_feature_extractor(lds_layer):
    gfe_layer = bottleneck_block(lds_layer, 64, (3, 3), t=6, strides=2, n=3)
    print("gfe_layer.shape:", gfe_layer.shape)
    gfe_layer = bottleneck_block(gfe_layer, 96, (3, 3), t=6, strides=2, n=3)
    print("gfe_layer.shape:", gfe_layer.shape)
    gfe_layer = bottleneck_block(gfe_layer, 128, (3, 3), t=6, strides=1, n=3)
    print("gfe_layer.shape:", gfe_layer.shape)
    gfe_layer = pyramid_pooling_block(gfe_layer, [2,4,6,8], gfe_layer.shape[1], gfe_layer.shape[2])
    print("gfe_layer.shape:", gfe_layer.shape)
    
    return gfe_layer


def pyramid_pooling_block(input_tensor, bin_sizes, w, h):
    print(w, h)
    concat_list = [input_tensor]
    #w = 16 # 64
    #h = 16 #32

    for bin_size in bin_sizes:
        x = tf.keras.layers.AveragePooling2D(pool_size=(w//bin_size, h//bin_size), 
                                             strides=(w//bin_size, h//bin_size))(input_tensor)
        x = tf.keras.layers.Conv2D(128, 3, 2, padding='same')(x)
        x = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, (w,h)))(x)
        print("x in paramid.shape", x.shape)

    concat_list.append(x)

    return tf.keras.layers.concatenate(concat_list)



def feature_fusion(lds_layer, gfe_layer):
    ff_layer1 = tf.keras.layers.Conv2D(128, (1,1), padding='same', strides = (1,1))(lds_layer)
    ff_layer1 = tf.keras.layers.BatchNormalization()(ff_layer1)
    #ff_layer1 = tf.keras.activations.relu(ff_layer1)
    print("ff_layer1.shape", ff_layer1.shape)
    
    #ss = conv_block(gfe_layer, 'conv', 128, (1,1), padding='same', strides= (1,1), relu=False)
    #print(ss.shape, ff_layer1.shape)
    

    ff_layer2 = tf.keras.layers.UpSampling2D((4, 4))(gfe_layer)
    print("ff_layer2.shape", ff_layer2.shape)
    ff_layer2 = tf.keras.layers.DepthwiseConv2D(128, strides=(1, 1), depth_multiplier=1, padding='same')(ff_layer2)
    
    print("ff_layer2.shape", ff_layer2.shape)
    ff_layer2 = tf.keras.layers.BatchNormalization()(ff_layer2)
    ff_layer2 = tf.keras.activations.relu(ff_layer2)
    ff_layer2 = tf.keras.layers.Conv2D(128, 1, 1, padding='same', activation=None)(ff_layer2)
    
    print("ff_layer2.shape", ff_layer2.shape)

    ff_final = tf.keras.layers.add([ff_layer1, ff_layer2])
    ff_final = tf.keras.layers.BatchNormalization()(ff_final)
    ff_final = tf.keras.activations.relu(ff_final)
    
    print("ff_final.shape", ff_final.shape)
    
    return ff_final

def classifier_layer(ff_final, num_classes):
    classifier = tf.keras.layers.SeparableConv2D(128, (3, 3), padding='same', 
                                                 strides = (1, 1), name = 'DSConv1_classifier')(ff_final)
    classifier = tf.keras.layers.BatchNormalization()(classifier)
    classifier = tf.keras.activations.relu(classifier)
    print("classifier.shape", classifier.shape)

    classifier = tf.keras.layers.SeparableConv2D(128, (3, 3), padding='same', 
                                                 strides = (1, 1), name = 'DSConv2_classifier')(classifier)
    classifier = tf.keras.layers.BatchNormalization()(classifier)
    classifier = tf.keras.activations.relu(classifier)
    print("classifier.shape", classifier.shape)
    #change 19 to 20
    #classifier = conv_block(classifier, 'conv', 20, (1, 1), strides=(1, 1), padding='same', relu=True)

    classifier = tf.keras.layers.Conv2D(num_classes, (1,1), padding='same', strides = (1,1))(classifier)
    classifier = tf.keras.layers.BatchNormalization()(classifier)
    classifier = tf.keras.activations.relu(classifier)
    print("classifier.shape", classifier.shape)
    
    classifier = tf.keras.layers.Dropout(0.3)(classifier)
    print("classifier before upsampling:", classifier.shape)

    classifier = tf.keras.activations.sigmoid(classifier)
    
    return classifier


inputs = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name = 'input_layer')
ds_layer = down_sample(inputs)
gfe_layer = global_feature_extractor(ds_layer)
ff_final = feature_fusion(ds_layer, gfe_layer)
classifier = classifier_layer(ff_final, 2)

# fast_scnn = tf.keras.Model(inputs = input_layer , outputs = classifier, name = 'Fast_SCNN')
# optimizer = tf.keras.optimizers.SGD(momentum=0.9, lr=0.045)
# fast_scnn.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])





# output = Conv2D(OUTPUT_CHANNELS, 1, activation = 'sigmoid')(conv_dec_4)

model = tf.keras.Model(inputs = inputs, outputs = classifier, name = 'Fast_SCNN')



class MaskMeanIoU(tf.keras.metrics.MeanIoU):
    """Mean Intersection over Union """
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight=sample_weight)



# model = unet_model(OUTPUT_CHANNELS)
model.compile(optimizer='adam',
              # loss=tfa.losses.GIoULoss(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(), #from_logits=True
              # metrics=[tf.keras.metrics.Accuracy()])
              metrics=['accuracy', MaskMeanIoU(name='iou', num_classes=OUTPUT_CHANNELS)])

model.summary()


tf.keras.utils.plot_model(model, show_shapes=True)    #plot model

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


# show_predictions()

# This function keeps the learning rate at 0.001 for the first ten epochs
# and decreases it exponentially after that.
def scheduler(epoch):
  if epoch < 10:
    return 0.001
  else:
    return 0.001 * tf.math.exp(0.1 * (10 - epoch))

LRS = tf.keras.callbacks.LearningRateScheduler(scheduler)

#  - TensorBoard
data_folder = Path("c:/TFlogs/fit/")
log_dir=data_folder / datetime.datetime.now().strftime("%m%d-%H%M%S")  #folder for tensorboard
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=True, profile_batch = 0, histogram_freq=1, write_images=False,) #, profile_batch=2, histogram_freq=1, write_graph=True

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
                          callbacks=[DisplayCallback(), LRS, tensorboard_callback])


loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

epochs = range(EPOCHS)

plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.show()

show_predictions(train_dataset, 3)
show_predictions(test_dataset, 3)

