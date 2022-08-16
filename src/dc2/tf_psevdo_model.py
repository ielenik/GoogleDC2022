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


class PsevdoModel(tf.keras.layers.Layer):
    def __init__(self, dir, mes, weight, types, **kwargs):
        super(PsevdoModel, self).__init__(**kwargs)
        weight[np.abs(mes) > 150] = 0
        mes[weight == 0] = 0
        self.dir    = tf.Variable(dir, name = 'dir', trainable=False, dtype = tf.float32)
        self.mes    = tf.Variable(mes, name = 'mes', trainable=False, dtype = tf.float32)
        self.weight = tf.Variable(weight, name = 'weight', trainable=False, dtype = tf.float32)
        self.types  = tf.Variable(types, name = 'types', trainable=False, dtype = tf.int32)

        NaN = float("NaN")
        mes[weight == 0] = NaN
        medians = np.zeros((len(mes), 6))
        for i in range(6):
            medians[:,i] = np.nanmedian(mes[:,types == i],axis=-1)
        medians[np.isnan(medians)] = 0
        weight[np.isnan(mes)] = 0
        mes[np.isnan(mes)] = 0

        self.bias   = tf.Variable(medians.T, name = 'bias', trainable=True, dtype = tf.float32)
        #self.shift_pp  = tf.Variable(np.zeros((len(mes),3)), name = 'shift_pp', trainable=True, dtype = tf.float32)
        #self.speedShift  = tf.Variable(np.zeros((3,3)), name = 'speedShift', trainable=True, dtype = tf.float32)
        #self.shift1  = tf.Variable(np.zeros((1,3)), name = 'shift1', trainable=True, dtype = tf.float32)
        #self.shift2  = tf.Variable(np.zeros((1,3)), name = 'shift2', trainable=True, dtype = tf.float32)
        #self.shift3  = tf.Variable(np.zeros((1,3)), name = 'shift3', trainable=True, dtype = tf.float32)
        #self.shift4  = tf.Variable(np.zeros((1,3)), name = 'shift4', trainable=True, dtype = tf.float32)

    def build(self, input_shape):
        super(PsevdoModel, self).build(input_shape)
    
    def calc_median(self, mes, weights):
        mes = mes + tf.cast(weights == 0,tf.float32)*1e7
        ind = tf.reduce_sum(tf.cast(weights > 0,tf.int32), axis = -1, keepdims=True) 
        srt = tf.sort(mes, axis = -1)
        res = tf.reshape(tf.gather_nd(batch_dims=1, params=srt, indices = ind//2),(-1,1))
        return res * tf.cast(ind > 0, tf.float32)

    def call(self, pos_shift):
        bias = tf.gather(self.bias, self.types)
        bias = tf.transpose(bias)
        pos_shift = tf.reshape(pos_shift,(-1,1,3))
        mes_est = tf.reduce_sum(self.dir*pos_shift, axis = -1) + self.mes + bias
        median = self.calc_median(mes_est,self.weight)
        mes_est -= median
        grad = tf.reduce_sum(self.dir*tf.expand_dims(mes_est*self.weight,-1), axis = 1)
        return tf.reduce_mean(tf.abs(mes_est)*self.weight) + tf.reduce_mean(tf.abs(self.bias[:,1:]-self.bias[:,:-1])), grad

    def compute_output_shape(self, _):
        return (1)
    def get_config(self):
        config = super().get_config().copy()
        return config        

def createPsevdoModel(sat_directions, sat_psevdoshifts, weights, types):
    return PsevdoModel(sat_directions, sat_psevdoshifts, weights, types)
    
