import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import os, sys
import random
random.seed(a=8, version=2)

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
from deep_learning_techniques import networks

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu", required=True, type=str)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

IMG_WIDTH = 2048 - 512
IMG_HEIGHT = 2048 - 512
IMG_CHANNELS = 1

sys.path.insert(1, '.')

data_dir = "./datasets/Teflon/"

def normalise(data):
    return (data - data.min())/(data.max() - data.min())

X_data = np.zeros((220, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
Y_data = np.zeros((220, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

for i in range(220):
    i_str = "0000"[len(str(i)):] + str(i)
    img_x = imread((data_dir + 'Plastic_4_holes_1m/SAMPLE_T' + i_str + '.tif'), as_gray=True)[:,:]
    img_x = img_x[560:2096, 512:2048,np.newaxis]
    img_x = normalise(img_x)
    X_data[i] = img_x

    img_y = np.load(data_dir + f"Registered_plastic_4_holes/SAMPLE_T{i_str}.tif.npy")
    img_y = img_y[560:2096,512:2048,np.newaxis]
    img_y = normalise(img_y)
    Y_data[i] = img_y

TRAIN_NUM = 220
X_train = X_data[0:TRAIN_NUM]
X_test = X_data[TRAIN_NUM:]
Y_train = Y_data[0:TRAIN_NUM]
Y_test = Y_data[TRAIN_NUM:]

model = networks.UNet(w=IMG_WIDTH, h=IMG_HEIGHT, c=IMG_CHANNELS)
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=1e-2),
    loss='mse',
    metrics=[tf.keras.metrics.RootMeanSquaredError()])
model.summary()

NAME = "narrow-unet-teflon{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

#Callbacks
callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode = 'min', patience=5, verbose=1),
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
