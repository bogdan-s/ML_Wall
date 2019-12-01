from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

tf.__version__


label_colours = [(0,0,0)
                # 0=background
                ,(255,255,255)]
                #wall mask

import pathlib
data_dir = 'D:/Python/DataSets/ADE20K_Filtered'
data_dir = pathlib.Path(data_dir)
print(data_dir)

#create a dataset of the file paths

list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*/*/*'))
for f in list_ds.take(5):
  print(f.numpy())