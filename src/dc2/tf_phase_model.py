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


class PhaseModel(tf.keras.layers.Layer):
    def __init__(self, dir, mes, weight, **kwargs):
        super(PhaseModel, self).__init__(**kwargs)
        NaN = float("NaN")
        num_sput = np.sum(weight > 0, axis = -1)
        weight[num_sput<8,:] = 0
        mes[weight == 0] = NaN
        mes -= np.nanmedian(mes,axis=-1,keepdims=True)
        mes[weight == 0] = 0
        weight[tf.abs(mes) > 30] = 0
        mes[weight == 0] = 0

        self.dir  = tf.Variable(dir, name = 'dir', trainable=False, dtype = tf.float32)
        self.mes  = tf.Variable(mes, name = 'mes', trainable=False, dtype = tf.float32)
        self.weight  = tf.Variable(weight, name = 'weight', trainable=False, dtype = tf.float32)
        #self.bias      = tf.Variable(np.zeros( (len(mes),1)) , name = 'bias', trainable=True, dtype = tf.float32)
        self.bias_sat      = tf.Variable(np.zeros( (1,len(dir[0]))) , name = 'bias_sat', trainable=True, dtype = tf.float32)
        self.bias_shift      = tf.Variable(np.zeros( (1,3)) , name = 'bias_shift', trainable=True, dtype = tf.float32)

    def build(self, input_shape):
        super(PhaseModel, self).build(input_shape)
    
    def calc_median(self, mes, weights):
        mes = mes + tf.cast(weights == 0,tf.float32)*1e7
        ind = tf.reduce_sum(tf.cast(weights > 0,tf.int32), axis = -1, keepdims=True) 
        srt = tf.sort(mes, axis = -1)
        res = tf.reshape(tf.gather_nd(batch_dims=1, params=srt, indices = ind//2),(-1,1))
        return res * tf.cast(ind > 0, tf.float32)

    def call(self, inputs):
        speed, quats = inputs
        hsp = tf.reshape(speed-self.bias_shift/100,(-1,1,3))
        mes_est = tf.reduce_sum(self.dir*hsp, axis = -1) - self.mes + self.bias_sat/100
        med = self.calc_median(mes_est, self.weight)
        mes_est -= med
        # 
        loss = tf.reduce_mean(tf.abs(mes_est)*self.weight, axis = -1)
        return loss

    def compute_output_shape(self, _):
        return (1)
    def get_config(self):
        config = super().get_config().copy()
        return config        

def createPhaseModel(sat_directions, sat_phases, phase_weights):
    phase_weights_valid = np.sum(phase_weights>1,axis=-1,keepdims=True)
    phase_weights = phase_weights*(phase_weights_valid>8)
    return PhaseModel(sat_directions, sat_phases, phase_weights)
    
