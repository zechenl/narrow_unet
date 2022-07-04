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
import scipy
import skimage
from scipy.ndimage import shift

from PIL import Image

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

data_dir = "./datasets/Teflon/"

def normalise(data):
    return (data - data.min())/(data.max() - data.min())

for i in range(221):
    i_str = "0000"[len(str(i)):] + str(i)
    path_x = 'Plastic_4_holes_1m/SAMPLE_T' + i_str + '.tif'
    img_x = imread((data_dir + path_x), as_gray=True)[:,:]
    img_x = img_x[560:2080, 280:2280]
    img_x = normalise(img_x)
    path_y = 'PLastic_4_holes/SAMPLE_T' + i_str + '.tif'
    img_y = imread((data_dir + path_y), as_gray=True)[:,:]
    cache_raw_img_y = img_y
    img_y = img_y[560:2080, 280:2280]
    img_y = normalise(img_y)

    source_im = img_x
    moving_im = img_y

    sobel1 = scipy.ndimage.sobel(source_im)
    sobel2 = scipy.ndimage.sobel(moving_im)

    flow = skimage.registration.optical_flow_ilk(sobel1, sobel2)
    flow_x = flow[0, :, :]
    flow_y = flow[1, :, :]
    x_off = np.mean(flow_x)
    y_off = np.mean(flow_y)
    corrected_im = shift(cache_raw_img_y, shift = (-x_off, -y_off), mode = "constant")

    save_dir = data_dir + f"Registered_plastic_4_holes/SAMPLE_T{i_str}.tif"
    np.save(save_dir, corrected_im)
