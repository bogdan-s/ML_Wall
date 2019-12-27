import os
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
# from tensorflow_examples.models.pix2pix import pix2pix

AUTOTUNE = tf.data.experimental.AUTOTUNE

print(tf.__version__, end='\n\n')

# param
IMG_SIZE = 256
BATCH_SIZE = 8
OUTPUT_CHANNELS = 2
EPOCHS = 20
away_from_computer = True  # to show or not predictions between batches

# dataset location
Test_Images_Path = "D:/Python/ML_Wall/Test_Images/"

test_imgs = tf.data.Dataset.list_files(Test_Images_Path + "*.jpg")


@tf.function
def parse_image(img_path):
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.uint8)

    return {'image': image}

# train_set = train_imgs.map(parse_image)
test_set = test_imgs.map(parse_image)

dataset = {"test": test_set}

print(dataset.keys())

# first I create the function to normalize, resize and apply some data augmentation on my dataset:

@tf.function
def normalize(input_image: tf.Tensor, ) -> tuple:
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image, 



@tf.function
def load_image_test(datapoint: dict) -> tuple:

    input_image = tf.image.resize(datapoint['image'], (IMG_SIZE, IMG_SIZE))


    input_image = normalize(input_image)

    return input_image, 




test = dataset['test'].map(load_image_test)

test_dataset = test.batch(BATCH_SIZE)


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
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

for image in test.take(1):
    sample_image = image
    print(sample_image[0][0].shape)

display_sample([sample_image[0][0]])

# print('train daset: ', train)

SIZE = IMG_SIZE

# Unet model without pretrained weights

def Unet(num_class, image_size):

    inputs = Input(shape=[image_size, image_size, 3])
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same')(conv9)
    conv10 = Conv2D(num_class, 1, activation = 'sigmoid')(conv9)
    model = Model(inputs = inputs, outputs = conv10)
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

    return model

model = Unet(OUTPUT_CHANNELS, IMG_SIZE)



def create_mask(pred_mask: tf.Tensor) -> tf.Tensor:

    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def show_predictions(dataset=None, num=1):

    if dataset:
        for image in dataset.take(num):
            image = image[0][0]
            pred_mask = model.predict(image)
            display_sample([image, create_mask(pred_mask)])
    else:
        for i in range(0, 2):
            for image in test.take(i):
                sample_image = image[0][0]
                display_sample([sample_image,
                                create_mask(model.predict(sample_image[tf.newaxis, ...]))])


# load weights from last save
if os.path.exists("./U-net_model.h5"): model.load_weights("U-net_model.h5")

# for i in range(0,9):
show_predictions()
