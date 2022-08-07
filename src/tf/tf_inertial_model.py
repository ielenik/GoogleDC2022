from re import T

from numpy.core.fromnumeric import reshape
from numpy.lib.function_base import append
from tensorflow.python.keras import callbacks
from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.python.keras.layers import BatchNormalization
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

import tensorflow_addons as tfa
import tensorflow_datasets as tfds
import pymap3d as pm
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from skimage.io import imread, imsave
import time
import pandas as pd
import itertools
from matplotlib import pyplot
import datetime

class InertialLayer(tf.keras.layers.Layer):
    def __init__(self, measures, valid, **kwargs):
        super(InertialLayer, self).__init__(**kwargs)
        measures = np.reshape(measures,(1,-1,2,3))
        self.measures = tf.Variable(measures, trainable=False, dtype=tf.float32)
        self.valid = tf.Variable(np.reshape(valid,(-1,1)), trainable=False, dtype=tf.float32)

        self.conv1 = tf.keras.layers.Conv2D(4,3,padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(4,3,padding='same', activation='relu')
        self.conv4 = tf.keras.layers.Conv2D(4,3,padding='same', activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(4,3,padding='same', activation='relu')
        self.pooling = tf.keras.layers.AveragePooling2D(pool_size=(2, 1))
        self.pooling2d = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))
        self.maxpool = tf.keras.layers.MaxPooling2D(pool_size=(2, 1))
        self.lastconv = tf.keras.layers.Conv2D(3,3,padding='same', activation=None, use_bias=False)
        self.batchnorm1 = BatchNormalization()
        self.batchnorm2 = BatchNormalization()


    def build(self, input_shape):
        super(InertialLayer, self).build(input_shape)

    def call(self, inputs):

        x = self.conv1(self.measures)
        x = self.batchnorm1(x)
        x = self.conv2(x)
        #x = self.pooling(x)
        #x = self.conv4(x)
        x = self.pooling2d(x)
        x = self.batchnorm2(x)
        x = self.conv3(x)
        x = self.lastconv(x)
        x = tf.reshape(x,(-1,3))*self.valid
        return tf.gather_nd(x, inputs)

        '''
        raw = self.measures
        raw = self.pooling(raw)
        raw = raw[:,:,0,:]
        raw = tf.reshape(raw,(-1,3))
        v = tf.relu(1 - self.valid)
        v = self.maxpool(v)
        v = self.maxpool(v)
        v = 1-v
        '''
        return tf.reduce_mean(tf.math.minimum(tf.math.abs(x-accsl), tf.math.abs(accsl))) + tf.reduce_mean(tf.math.abs(x-accsl))/100, accsl, x, raw

    def compute_output_shape(self, _):
        return (1)
    def get_config(self):
        config = super().get_config().copy()
        config.update({
        })
        return config        

def create_inertial_model(start_nanos, numepochs, tick, df_UncalAccel, df_UncalGyro,phone_shifts):
    multiplier = 2
    my_tick = tick//multiplier
    measures = np.zeros( ((numepochs-2)*multiplier,2,3)  )
    count = np.zeros(((numepochs-2)*multiplier)).astype(np.int32)

    df_UncalAccel['epoch'] = ((df_UncalAccel['utcTimeMillis']- 315964800000.0 + 18000)*1000000-start_nanos)//my_tick
    df_UncalAccel = df_UncalAccel.groupby(['epoch']).mean()
    df_UncalAccel['epoch'] = ((df_UncalAccel['utcTimeMillis']- 315964800000.0 + 18000)*1000000-start_nanos)//my_tick
    for _, r in df_UncalAccel.iterrows():
        epoch = int(r['epoch'])
        if epoch < 240 or numepochs*multiplier - epoch < 120: #drop 30 sec of noise
            continue
        count[epoch] += 1
        measures[epoch,0] = r[['UncalAccelXMps2','UncalAccelYMps2','UncalAccelZMps2']].to_numpy()

    
    df_UncalGyro['epoch'] = ((df_UncalGyro['utcTimeMillis']- 315964800000.0 + 18000)*1000000-start_nanos)//my_tick
    df_UncalGyro = df_UncalGyro.groupby(['epoch']).mean()
    df_UncalGyro['epoch'] = ((df_UncalGyro['utcTimeMillis']- 315964800000.0 + 18000)*1000000-start_nanos)//my_tick
    for _, r in df_UncalGyro.iterrows():
        epoch = int(r['epoch'])
        if epoch < 10 or numepochs*multiplier - epoch < 10: #drop 5 sec of noise
            continue
        measures[epoch,1] = r[['UncalGyroXRadPerSec','UncalGyroYRadPerSec','UncalGyroZRadPerSec']].to_numpy()

    measures[count == 0] = np.array([[0.,0.,0.], [0.,0.,0.]])

    @tf.custom_gradient
    def norm(x):
        y = tf.linalg.norm(x, 'euclidean', -1)
        def grad(dy):
            return tf.expand_dims(dy,-1) * (x / (tf.expand_dims(y + 1e-19,-1)))

        return y, grad

    def my_loss(t,p):
        return tf.reduce_mean(norm(t-p))

    valid = count[::2]
    il = InertialLayer(measures, valid)
    epoch_index = tf.keras.layers.Input((1), dtype=tf.int64)
    pred_acs_local = il(epoch_index)
    model_prebuild = tf.keras.models.Model(epoch_index,pred_acs_local)
    model_prebuild.compile(optimizer = 'Adam', loss = 'mse')

    valid_shifts = (phone_shifts[1:] + phone_shifts[:-1])**np.reshape(valid,(-1,1))


    mat_local = np.zeros((len(valid_shifts),3,3))
    mat_local[:,1] = valid_shifts/(np.linalg.norm(valid_shifts, axis = -1, keepdims=True)+1e-5)
    mat_local[:,2] = np.array([[0,0,1]])
    mat_local[:,2] = mat_local[:,2] - mat_local[:,1]*np.sum(mat_local[:,2]*mat_local[:,1], axis = -1, keepdims=True)
    mat_local[:,2] = mat_local[:,2]/(np.linalg.norm(mat_local[:,2], axis = -1, keepdims=True)+1e-5)
    mat_local[:,0] = np.cross(mat_local[:,1], mat_local[:,2])
    mat_local = np.transpose(mat_local, (0,2,1))
    mat_local_copy = mat_local.copy()


    valid_accel  = (phone_shifts[1:] - phone_shifts[:-1])*np.reshape(valid,(-1,1))
    valid_accel = np.matmul(np.reshape(valid_accel,(-1,1,3)),mat_local)
    valid_accel = np.reshape(valid_accel,(-1, 3))

    valid_count = np.arange(len(valid_shifts))

    valid_train = np.linalg.norm(valid_shifts,axis=-1)>0.3
    valid_accel_train = valid_accel[valid_train]
    valid_count_train = valid_count[valid_train]

    model_prebuild.fit(valid_count_train,valid_accel_train, epochs = 256, batch_size = 128, verbose=0, callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',patience=30,factor=0.5)])

    pred_acs = model_prebuild(valid_count).numpy()
    err = model_prebuild.evaluate(valid_count,valid_accel)


    mes = measures[0::2,0,:]
    mes = mes[:len(valid_shifts)]
    
    '''plt.clf()
    plt.plot( valid_count, valid_accel[:,1])
    plt.plot( valid_count, pred_acs[:,1])
    plt.plot( valid_count, mes[:,0]+3, alpha=0.5)
    plt.plot( valid_count, mes[:,1]+3, alpha=0.5)
    plt.plot( valid_count, mes[:,2]+3, alpha=0.5)

    plt.legend(['acs y', 'phn y', 'msr x','msr y', 'msr z'])
    plt.show()
    '''
    epoch_index = tf.keras.layers.Input((1), dtype=tf.int64)
    inp_dir = tf.keras.layers.Input((3))
    pred_acs_local = il(epoch_index)
    
    mat_local = tf.zeros((len(valid_shifts),3,3))
    vecy = inp_dir/(tf.linalg.norm(inp_dir, axis = -1, keepdims=True)+1e-5)
    vecz = tf.Variable([[0.,0.,1.]])
    vecz = vecz - vecy*tf.reduce_sum(vecz*vecy, axis = -1, keepdims=True)
    vecz = vecz/(tf.linalg.norm(vecz, axis = -1, keepdims=True)+1e-5)
    vecx = tf.linalg.cross(vecy, vecz)

    vecx = tf.reshape(vecx,(-1,1,3))
    vecy = tf.reshape(vecy,(-1,1,3))
    vecz = tf.reshape(vecz,(-1,1,3))

    mat_local = tf.concat([vecx,vecy,vecz], axis = 1)
    mat_local = tf.transpose(mat_local,(0,2,1))


    pred_acs_global = tf.linalg.matvec(mat_local,pred_acs_local)
  
    
    model_prebuild = tf.keras.models.Model([epoch_index,inp_dir],pred_acs_global)#[pred_acs_global,mat_local,pred_acs_local])

    mat_local_copy = np.transpose(mat_local_copy, (0,2,1))
    valid_accel = np.matmul(np.reshape(valid_accel,(-1,1,3)),mat_local_copy)
    valid_accel = np.reshape(valid_accel,(-1, 3))

    pred_acs = np.matmul(np.reshape(pred_acs,(-1,1,3)),mat_local_copy)
    pred_acs = np.reshape(pred_acs,(-1, 3))

    #pred_acs2, mat_local_copy2,pred_acs_local2 = model_prebuild([valid_count,valid_shifts])
    #pred_acs2, mat_local_copy2,pred_acs_local2 = pred_acs2.numpy(), mat_local_copy2.numpy(), pred_acs_local2.numpy()

    return model_prebuild, err, pred_acs, valid_accel, valid
    
