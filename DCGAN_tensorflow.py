from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
from IPython import display


(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
print(type(train_images), type(train_labels))
print(train_images.shape, train_labels.shape)

BUFFER_SIZE = 60000
BATCH_SIZE = 256

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
print(train_dataset)