from collections import deque
from bisect import insort, bisect_left
from itertools import islice
import numpy as np
import tensorflow as tf
from matplotlib import pyplot
import math
import random
from ..utils.magnet_model import Magnetometer

from tensorflow.python.framework import function
from tensorflow.python.framework import function
from .tf_numpy_tools import *


class DoplerModel(tf.keras.layers.Layer):
    def __init__(self, dir, mes, weight, types, **kwargs):
        super(DoplerModel, self).__init__(**kwargs)
        weight[weight < 0.5] = 0
        NaN = float("NaN")
        num_sput = np.sum(weight > 0, axis = -1)
        weight[num_sput<10,:] = 0
        mes[weight == 0] = NaN
        for i in range(5):
            medians = np.zeros((8))
            for i in range(8):
                medians[i] = np.nanmedian(mes[:,types == i])

            for i in range(len(mes[0])):
                mes[:,i] -= medians[types[i]]

            mes -= np.nanmedian(mes,axis=-1,keepdims=True)
        weight[mes > 40] = 0
        mes[weight == 0] = 0

        dir = dir[1:-1]
        mes = mes[1:-1]
        weight = weight[1:-1]

        self.dir  = tf.Variable(dir, name = 'dir', trainable=False, dtype = tf.float32)
        self.mes  = tf.Variable(mes, name = 'mes', trainable=False, dtype = tf.float32)
        self.weight  = tf.Variable(weight, name = 'weight', trainable=False, dtype = tf.float32)
        self.types  = tf.Variable(types, name = 'types', trainable=False, dtype = tf.int32)
        self.bias      = tf.Variable(np.zeros((6,len(mes))), name = 'bias', trainable=True, dtype = tf.float32)
        self.bias_shift      = tf.Variable(np.zeros( (1,3)) , name = 'bias_shift', trainable=True, dtype = tf.float32)
        self.time_shift      = tf.Variable(np.ones(3) , name = 'time_shift', trainable=True, dtype = tf.float32)

    def build(self, input_shape):
        super(DoplerModel, self).build(input_shape)

    def calc_median(self, mes, weights):
        mes = mes + tf.cast(weights == 0,tf.float32)*1e7
        ind = tf.reduce_sum(tf.cast(weights > 0,tf.int32), axis = -1, keepdims=True) 
        srt = tf.sort(mes, axis = -1)
        res = tf.reshape(tf.gather_nd(batch_dims=1, params=srt, indices = ind//2),(-1,1))
        return res * tf.cast(ind > 0, tf.float32)

    def call(self, inputs):
        speed, quats, times_dif = inputs
        speed = speed*1000/times_dif
        speed = tf.concat([speed,[speed[-1]]], axis = 0)
        speed = (speed[2:]*self.time_shift[0]+speed[1:-1]*self.time_shift[1]+speed[:-2]*self.time_shift[2])/tf.reduce_sum(self.time_shift) + self.bias_shift/100
        bias = tf.gather(self.bias, self.types)
        speed = tf.reshape(speed,(-1,1,3))
        bias = tf.transpose(bias)
        mes_est = tf.reduce_sum(self.dir*speed, axis = -1)  - self.mes  + bias
        mes_est -= self.calc_median(mes_est,self.weight)
        bias_loss = tf.reduce_sum(tf.abs(self.bias[:,1:] - self.bias[:,:-1]), axis = 0)
        bias_loss = tf.concat([bias_loss, [0]],axis = 0)

        loss = tf.reduce_mean(tf.abs(mes_est)*self.weight, axis = -1) + bias_loss
        loss = tf.concat([[0],loss],axis = 0)
        return loss 

    def compute_output_shape(self, _):
        return (1)
    def get_config(self):
        config = super().get_config().copy()
        return config        

def createXiaomiDoplerModel(sat_directions, sat_deltaspeed, sat_deltaspeeduncert, sat_types):
    return DoplerModel(sat_directions, sat_deltaspeed, 1./sat_deltaspeeduncert, sat_types)
    
