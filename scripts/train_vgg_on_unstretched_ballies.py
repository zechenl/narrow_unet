import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import os, sys
import random
import numpy as np
import argparse

from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt

from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard

import time

sys.path.insert(1, '.')
from . import deep_learning_techniques

parser = argparser.parser_args()
parser.add_argument("-g", "--gpu", required=True, type=str)
args = parser.parser_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

# Initialise the input image dimensions and the number of colour channels
# X-ray images are greyscale so they we require only one colour channel
IMG_WIDTH = 1024
IMG_HEIGHT = 1024
IMG_CHANNELS = 1

# Initialise numpy arrays for X_data and Y_data to store the input images
X_data = np.zeros((180, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
Y_data = np.zeros((180, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

# Here the 180 images are loaded into numpy arrays with shape (180, 1024, 1024, 1)
# For each 1024 by 1024 matrix, each element in the matrix corresponds to the pixel value in the .tif image
# Load 1mm images into X_data
for i in range(10):
    img = imread(('1 mm/32p1r00' + str(i) + '.tif'), as_gray=True)[:,:]
    img = img[:,:,np.newaxis]
    X_data[i] = img

for i in range(10,100):
    img = imread(('1 mm/32p1r0' + str(i) + '.tif'), as_gray=True)[:,:]
    img = img[:,:,np.newaxis]
    X_data[i] = img

for i in range(10):
    img = imread(('1 mm/32p1r10' + str(i) + '.tif'), as_gray=True)[:,:]
    img = img[:,:,np.newaxis]
    X_data[100+i] = img

for i in range (10,80):
    img = imread(('1 mm/32p1r1' + str(i) + '.tif'), as_gray=True)[:,:]
    img = img[:,:,np.newaxis]
    X_data[100+i] = img


# Load 0mm data into Y_data
for i in range(10):
    img = imread(('0 mm/32p0r00' + str(i) + '.tif'), as_gray=True)[:,:]
    img = img[:,:,np.newaxis]
    Y_data[i] = img

for i in range(10,100):
    img = imread(('0 mm/32p0r0' + str(i) + '.tif'), as_gray=True)[:,:]
    img = img[:,:,np.newaxis]
    Y_data[i] = img

for i in range(10):
    img = imread(('0 mm/32p0r10' + str(i) + '.tif'), as_gray=True)[:,:]
    img = img[:,:,np.newaxis]
    Y_data[100+i] = img

for i in range (10,80):
    img = imread(('0 mm/32p0r1' + str(i) + '.tif'), as_gray=True)[:,:]
    img = img[:,:,np.newaxis]
    Y_data[100+i] = img

# Assign the first 170 images in X_data and Y_data to X_train and Y_train respectively
# and the last 10 images to X_test and Y_test
TRAIN_NUM = 170
X_train = X_data[0:TRAIN_NUM]
X_test = X_data[TRAIN_NUM:]
Y_train = Y_data[0:TRAIN_NUM]
Y_test = Y_data[TRAIN_NUM:]

# Stretch X_train and Y_train and assign to X_train_stretch and Y_train_stretch respectively
X_train_temp = np.subtract(X_train,0.9)
X_train_stretch = np.multiply(X_train_temp,10)

Y_train_temp = np.subtract(Y_train,0.9)
Y_train_stretch = np.multiply(Y_train_temp,10)

X_test_temp = np.subtract(X_test,0.9)
X_test_stretch = np.multiply(X_test_temp,10)

random.seed(a=8, version=2)

model = deep_learning_techniques.networks.NarrowUNet()
model.compile(
    optimizer='sgd',
    loss='mse',
    metrics=[tf.keras.metrics.RootMeanSquaredError()])
model.summary()

NAME = "narrow-unet-unstretched{}".format(int(time.time()))

#Callbacks
callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss'),
        tensorboard]

tf.test.is_gpu_available(
    cuda_only=False, min_cuda_compute_capability=None
)

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

with tf.device("/device:GPU:0"):
    results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=1, epochs=25, callbacks=callbacks)
model.save(os.path.join("./models", NAME))
