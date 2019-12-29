import os
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib

from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, tarball_path):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()

    graph_def = None
    # Extract frozen graph from tar archive.
    tar_file = tarfile.open(tarball_path)
    for tar_info in tar_file.getmembers():
      if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
        file_handle = tar_file.extractfile(tar_info)
        graph_def = tf.GraphDef.FromString(file_handle.read())
        break

    tar_file.close()

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(graph=self.graph)

  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    width, height = image.size

    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    return resized_image, seg_map


def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=int)
  ind = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

  return colormap


def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_pascal_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]


def vis_segmentation(image, seg_map):
  """Visualizes input image, segmentation map and overlay view."""
  plt.figure(figsize=(15, 5))
  grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

  plt.subplot(grid_spec[0])
  plt.imshow(image)
  plt.axis('off')
  plt.title('input image')

  plt.subplot(grid_spec[1])
  seg_image = label_to_color_image(seg_map).astype(np.uint8)
  plt.imshow(seg_image)
  plt.axis('off')
  plt.title('segmentation map')

  plt.subplot(grid_spec[2])
  plt.imshow(image)
  plt.imshow(seg_image, alpha=0.7)
  plt.axis('off')
  plt.title('segmentation overlay')

  unique_labels = np.unique(seg_map)
  ax = plt.subplot(grid_spec[3])
  plt.imshow(
      FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
  ax.yaxis.tick_right()
  plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
  plt.xticks([], [])
  ax.tick_params(width=0.0)
  plt.grid('off')
  plt.show()


LABEL_NAMES = np.asarray(['not labeled', 'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road',
                     'bed', 'windowpane', 'grass', 'cabinet', 'sidewalk', 'person',
                     'earth', 'door', 'table', 'mountain', 'plant', 'curtain', 'chair',
                     'car', 'water', 'painting', 'sofa', 'shelf', 'house', 'sea',
                     'mirror', 'rug', 'field', 'armchair', 'seat', 'fence', 'desk',
                     'rock', 'wardrobe', 'lamp', 'bathtub', 'railing', 'cushion',
                     'base', 'box', 'column', 'signboard', 'chest of drawers',
                     'counter', 'sand', 'sink', 'skyscraper', 'fireplace',
                     'refrigerator', 'grandstand', 'path', 'stairs', 'runway', 'case',
                     'pool table', 'pillow', 'screen door', 'stairway', 'river',
                     'bridge', 'bookcase', 'blind', 'coffee table', 'toilet', 'flower',
                     'book', 'hill', 'bench', 'countertop', 'stove', 'palm',
                     'kitchen island', 'computer', 'swivel chair', 'boat', 'bar',
                     'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck',
                     'tower', 'chandelier', 'awning', 'streetlight', 'booth',
                     'television receiver', 'airplane', 'dirt track', 'apparel', 'pole',
                     'land', 'bannister', 'escalator', 'ottoman', 'bottle', 'buffet',
                     'poster', 'stage', 'van', 'ship', 'fountain', 'conveyer belt',
                     'canopy', 'washer', 'plaything', 'swimming pool', 'stool',
                     'barrel', 'basket', 'waterfall', 'tent', 'bag', 'minibike',
                     'cradle', 'oven', 'ball', 'food', 'step', 'tank', 'trade name',
                     'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher',
                     'screen', 'blanket', 'sculpture', 'hood', 'sconce', 'vase',
                     'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen',
                     'plate', 'monitor', 'bulletin board', 'shower', 'radiator',
                     'glass', 'clock', 'flag'])

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)


MODEL = DeepLabModel('Weights/deeplabv3_xception_ade20k_train_2018_05_29.tar.gz')




def run_visualization(url):
  """Inferences DeepLab model and visualizes result."""
  try:
    # f = urllib.request.urlopen(url)
    # jpeg_str = f.read()
    original_im = Image.open(url)
  except IOError:
    print('Cannot retrieve image. Please check url: ' + url)
    return

  print('running deeplab on image %s...' % url)
  resized_im, seg_map = MODEL.run(original_im)
  plt.imshow(seg_map)
  plt.show()
  vis_segmentation(resized_im, seg_map)


# image_url = 'D:/Python/DataSets/ADE20K_Filtered/Train/Images/0/' + 'ADE_train_00000054.jpg'
image_url = 'D:/Python/ML_Wall/Test_images/' + '20191227_184754.jpg'
run_visualization(image_url)








