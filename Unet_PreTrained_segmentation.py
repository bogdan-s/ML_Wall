import tensorflow as tf
from IPython.display import clear_output
import IPython.display as display
from glob import glob
import matplotlib.pyplot as plt
from tensorflow import keras
import pix2pix
# from tensorflow_examples.models.pix2pix import pix2pix

AUTOTUNE = tf.data.experimental.AUTOTUNE

print(tf.__version__, end='\n\n')

# the dataset is downloaded in data/raw/ade20k/
IMG_SIZE = 224

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
    
    mask_path = tf.strings.regex_replace(img_path, "Images", "Masks")
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
    input_image = tf.cast(input_image, tf.float32) / 255.0
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
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (IMG_SIZE, IMG_SIZE))

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

BATCH_SIZE = 128
BUFFER_SIZE = BATCH_SIZE
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
print('steps per epoch: ', STEPS_PER_EPOCH)

train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test = dataset['test'].map(load_image_test)

train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
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

for image, mask in train.take(1):
    sample_image, sample_mask = image, mask
    print(sample_mask)

# display_sample([sample_image, sample_mask])

# print('train daset: ', train)

SIZE = IMG_SIZE

# Developing the Model (UNet-ish)

OUTPUT_CHANNELS = 2

base_model = tf.keras.applications.MobileNetV2(input_shape=[SIZE, SIZE, 3], include_top=False)

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

up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]

def unet_model(output_channels):

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
        output_channels, 3, strides=2,
        padding='same', activation='softmax')  #64x64 -> 128x128

    inputs = tf.keras.layers.Input(shape=[SIZE, SIZE, 3])
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

tf.keras.utils.plot_model(model, show_shapes=True)



def create_mask(pred_mask: tf.Tensor) -> tf.Tensor:
    """Return a filter mask with the top 1 predicitons
    only.

    Parameters
    ----------
    pred_mask : tf.Tensor
        A [SIZE, SIZE, N_CLASS] tensor. For each pixel we have
        N_CLASS values (vector) which represents the probability of the pixel
        being these classes. Example: A pixel with the vector [0.0, 0.0, 1.0]
        has been predicted class 2 with a probability of 100%.

    Returns
    -------
    tf.Tensor
        A [SIZE, SIZE, 1] mask with top 1 predictions
        for each pixels.
    """
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def show_predictions(dataset=None, num=1):
    """
    Show a sample prediction.

    Parameters
    ----------
    dataset : [type], optional
        [Input dataset, by default None
    num : int, optional
        Number of sample to show, by default 1
    """
    if dataset:
        for image, mask in dataset.take(num):
            # print(type(image))
            # print(image.shape)
            # print(image)
            # print('/n/n')
            pred_mask = model.predict(image)
            display_sample([image[0], mask[0], create_mask(pred_mask)])
    else:
        # The model is expecting a tensor of the size
        # [BATCH_SIZE, SIZE, SIZE, 3] for the images
        # and
        # [BATCH_SIZE, SIZE, SIZE, 1] for the annotations
        # -> sample_image is [SIZE, SIZE, 1]
        # -> sample_image[tf.newaxis, ...] is [BATCH_SIZE, SIZE, SIZE, 1]
        for image, mask in test.take(1):
            sample_image, sample_mask = image, mask
        display_sample([sample_image, sample_mask,
                        create_mask(model.predict(sample_image[tf.newaxis, ...]))])



# show_predictions(train)
# for image, mask in train.take(2):
#     sample_image, sample_mask = image, mask
# show_predictions()


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        # show_predictions()
        print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

EPOCHS = 20
VAL_SUBSPLITS = 2
VALIDATION_STEPS = VAL_LENGTH // BATCH_SIZE

model_history = model.fit(train_dataset, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_dataset,
                          callbacks=[DisplayCallback()])


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


show_predictions(test_dataset, 1)