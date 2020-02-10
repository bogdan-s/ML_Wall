import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np


file = tf.keras.utils.get_file(
    "c.jpg",
    "file:///D:/Python/ML_Wall/Test_images/c.jpg")
img = tf.keras.preprocessing.image.load_img(file, target_size=[224, 224])

IMG_SIZE = 256

# Show image
# plt.imshow(img)
# plt.axis('off')
# plt.show()

x = tf.keras.preprocessing.image.img_to_array(img)
x = tf.image.resize(x,(IMG_SIZE, IMG_SIZE))
x = tf.keras.applications.mobilenet.preprocess_input(
    x[tf.newaxis,...])

# print(type(x))

def display_sample(display_list):
    """Show side-by-side an input image,
    the ground truth and the prediction.
    """
    plt.figure(figsize=(18, 18))

    title = ['Input Image', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

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
    print("Prediction's shape is: ")
    print(pred_mask.shape)
    pred_mask = tf.argmax(pred_mask, axis=-1)
    print("Prediction's shape is: ")
    print(pred_mask.shape)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


model = keras.models.load_model("./Weights/U-net_for_inference.h5")

y = model.predict(x)

print(y.shape)

y = create_mask(y)

print(y.shape)


display_sample([tf.squeeze(x), y])


# Show image
# plt.imshow(y,  cmap="gray")
# plt.axis('off')
# plt.show()

# display_sample([x, create_mask(model.predict(x))])
