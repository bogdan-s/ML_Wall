import tensorflow as tf
import matplotlib.pyplot as plt

a_img_ds = tf.data.Dataset.list_files('D:/Python/DataSets/ADE20K_Filtered/compare/A/*.png', shuffle=False)
b_img_ds = tf.data.Dataset.list_files('D:/Python/DataSets/ADE20K_Filtered/compare/B/*.png', shuffle=False)

a_img_ds = a_img_ds.map(lambda x: tf.io.read_file(x))
a_img_ds = a_img_ds.map(lambda x: tf.io.decode_png(x, channels=0, dtype=tf.dtypes.uint8))

b_img_ds = b_img_ds.map(lambda x: tf.io.read_file(x))
b_img_ds = b_img_ds.map(lambda x: tf.io.decode_png(x, channels=0, dtype=tf.dtypes.uint8))

for x in a_img_ds:
    print(x.dtype, x.shape)
for x in b_img_ds:
    print(x.dtype, x.shape)


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

a_img_ds = a_img_ds.map(lambda x: tf.image.rgb_to_grayscale(x))
b_img_ds = b_img_ds.map(lambda x: tf.image.rgb_to_grayscale(x))
for x in a_img_ds:
    print("a dataset grayscale: {}".format(x))
for x in b_img_ds:
    print("b dataset grayscale: {}".format(x))
a_img_ds = a_img_ds.map(lambda x: x/255)
b_img_ds = b_img_ds.map(lambda x: x/255)

for x in a_img_ds:
    print("a dataset normalized: {}".format(x))
for x in b_img_ds:
    print("b dataset normalized: {}".format(x))
# display_sample(list(a_img_ds))
# display_sample(list(b_img_ds))