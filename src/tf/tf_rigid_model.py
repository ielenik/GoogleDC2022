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
from .tf_numpy_tools import *


class RigidModel(tf.keras.layers.Layer):
    def __init__(self, pre_poses, pre_quats, **kwargs):
        super(RigidModel, self).__init__(**kwargs)
        self.epochs = len(pre_poses)
        self.pre_poses = pre_poses
        self.positions = tf.Variable(np.array(pre_poses), trainable=False, dtype = tf.float32, name = 'positions')
        self.positions_fourier = tf.Variable(np.zeros((self.epochs, 3)), trainable=False, dtype = tf.float32, name = 'positions_fourier')
        
        self.quat = tf.Variable(pre_quats, trainable=True, dtype = tf.float32, name = 'quat')
        self.shift_abs = tf.Variable([[0.,0.,0.]], trainable=True, dtype = tf.float32, name = 'shift_abs')
        self.shift_rel = tf.Variable([[0.,0.,0.]], trainable=True, dtype = tf.float32, name = 'shift_rel')

        indicies = tf.cast(tf.range(self.epochs), tf.float32)
        xval = indicies/self.epochs*math.pi
        xval = tf.reshape(xval,[-1,1])
        perd = tf.cast(tf.range(self.epochs), tf.float32)
        perd = tf.reshape(perd, [1,-1])
        self.fourier_args = tf.math.cos(xval*perd)
        print(self.fourier_args[self.epochs//2])
        print(self.fourier_args[0])
        print(tf.shape(self.fourier_args@self.positions_fourier))

    def build(self, input_shape):
        super(RigidModel, self).build(input_shape)

    def get_position_correction(self,index):
        indicies = tf.cast(tf.reshape(index,[-1]), tf.float32)
        xval = indicies/self.epochs*math.pi
        xval = tf.reshape(xval,[-1,1])
        perd = tf.cast(tf.range(self.epochs), tf.float32)
        perd = tf.reshape(perd, [1,-1])
        args = xval*perd
        return tf.reduce_sum(self.positions_fourier*tf.expand_dims(tf.math.cos(args), -1), axis = 1)
    def get_position(self,indicies):
        return tf.gather(self.positions, indicies, axis = 0) + self.get_position_correction(indicies)        
    
    def call(self, epoch_input):
        indicies = tf.reshape(epoch_input,[-1])
        qt, _ = tf.linalg.normalize(tf.gather(self.quat, indicies, axis = 0), axis = -1)

        pos = self.positions + self.fourier_args@self.positions_fourier
        pos = tf.gather(pos, indicies, axis = 0) + self.shift_abs + transform(self.shift_rel, qt)
        #pos = self.get_position(indicies) 

        return [pos, qt ]

    def compute_output_shape(self, _):
        return (1)
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'poses' : self.pre_poses,
        })
        return config        
def createTrackModel(start_nanos, end_nanos, df_baseline):


    #initialize positions at 10hz from baseline
    tick = 1000000000
    start_nanos = start_nanos - 10000000000
    end_nanos = end_nanos + 10000000000
    num_measures = (end_nanos - start_nanos)//tick # ten times a second
    positions = np.zeros((num_measures,3))
    for i in range(num_measures):
        positions[i] = getValuesAtTimeLiniar(df_baseline['times'], df_baseline['values'], (i*tick+start_nanos))


    course = np.zeros(positions.shape)
    first_pos = 0
    mark_pos = 0
    last_pos  = 1
    while last_pos < len(positions):
        mv = positions[last_pos] - positions[first_pos]
        if np.linalg.norm(mv) > 10:
            mvd = mv/np.linalg.norm(mv)
            while mark_pos < (last_pos + first_pos)/2:
                course[mark_pos ]= mvd
                mark_pos += 1
            first_pos += 1
        else:
            last_pos += 1
    while mark_pos < last_pos:
        course[mark_pos ]= mvd
        mark_pos += 1

    quats = np.zeros((course.shape[0],4))
    quats[:,3] = 1
    fwd = np.array([0.,1.,0])
    for i in range(course.shape[0]):
        quats[i] = get_quat(fwd, course[i])


    rover_pos  = RigidModel(positions, quats)

    #model to get position at given times (for training phone models)
    model_input = tf.keras.layers.Input((1), dtype=tf.int64)
    rel_times = model_input - start_nanos
    prev_index   = rel_times//tick
    prev_weight  = ((prev_index+1)*tick - rel_times)/tick
    prev_weight = tf.cast(prev_weight,tf.float32)


    poses1 = rover_pos(prev_index)
    poses2 = rover_pos(prev_index+1)

    poses = [poses1[0] * prev_weight + poses2[0] * (1 - prev_weight), poses1[1] * prev_weight + poses2[1] * (1 - prev_weight)]
    track_model = tf.keras.Model(model_input, poses)

    #model to get all position (for training speed/acs etc)
    dummy_input = tf.keras.layers.Input((1), dtype=tf.int64)
    poses = rover_pos(dummy_input)
    #dires = rover_dir(dummy_input)
    track_model_error = tf.keras.Model(dummy_input, poses)

    return track_model, track_model_error, num_measures, start_nanos, tick
