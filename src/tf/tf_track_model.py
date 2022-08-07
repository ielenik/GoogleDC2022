from re import T
import pickle

from numpy.core.fromnumeric import reshape
from numpy.lib.function_base import append
from tensorflow.python.ops.variables import Variable
from src.laika.lib.coordinates import ecef2geodetic, geodetic2ecef
from src.laika import AstroDog
from src.laika.gps_time import GPSTime

import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.ops.array_ops import zeros
from tensorflow.python.training.tracking import base

import pymap3d as pm
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from skimage.io import imread, imsave
import time
import pandas as pd
import itertools
from matplotlib import pyplot
import datetime
from src.utils.coords_tools import getValuesAtTimeLiniar


class WeightsData(tf.keras.layers.Layer):
    def __init__(self, value, regularizer = None, trainable = True, **kwargs):
        super(WeightsData, self).__init__(**kwargs)
        self.value = value
        self.tr = trainable
        self.regularizer = regularizer

    def build(self, input_shape):
        super(WeightsData, self).build(input_shape)
        self.W = self.add_weight(name='W', shape=self.value.shape, 
                                dtype = tf.float32,
                                initializer=tf.keras.initializers.Constant(self.value),
                                regularizer=self.regularizer,
                                trainable=self.tr)

    def call(self, inputs):
        return tf.gather_nd(self.W, inputs)
    def compute_output_shape(self, input_shape):
        return [(input_shape[0], 3)]
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'initializer': self.initializer,
            'inshape': self.inshape,
        })
        return config


class BiasLayer(tf.keras.layers.Layer):
  def __init__(self):
    super(BiasLayer, self).__init__()
    self.variable = tf.Variable([[0,0,0]], trainable=True, dtype=tf.float32)

  def call(self, inputs, **kwargs):
    return inputs + self.variable

def createTrackModel(start_nanos, end_nanos, df_baseline):


    #initialize positions at 10hz from baseline
    tick = 500000000
    start_nanos = start_nanos - 10000000000
    end_nanos = end_nanos + 10000000000
    num_measures = (end_nanos - start_nanos)//tick # ten times a second
    positions = np.zeros((num_measures,3))
    for i in range(num_measures):
        positions[i] = getValuesAtTimeLiniar(df_baseline['times'], df_baseline['values'], (i*tick+start_nanos))

    rover_pos  = WeightsData(positions, None, True)
    #rover_dir  = WeightsData(np.ones((num_measures,3)))

    bias = BiasLayer()
    #model to get position at given times (for training phone models)
    model_input = tf.keras.layers.Input((1), dtype=tf.int64)
    rel_times = model_input - start_nanos
    prev_index   = rel_times//tick
    prev_weight  = ((prev_index+1)*tick - rel_times)/tick
    prev_weight = tf.cast(prev_weight,tf.float32)

    poses = bias((rover_pos(prev_index) * prev_weight + rover_pos(prev_index+1) * (1 - prev_weight)))
    #dires = ((rover_dir(prev_index) * prev_weight + rover_dir(prev_index+1) * (1 - prev_weight)))
    track_model = tf.keras.Model(model_input, poses)

    #model to get all position (for training speed/acs etc)
    dummy_input = tf.keras.layers.Input((1), dtype=tf.int64)
    poses = bias(rover_pos(dummy_input))
    #dires = rover_dir(dummy_input)
    track_model_error = tf.keras.Model(dummy_input, poses)

    return track_model, track_model_error, num_measures, start_nanos, tick
