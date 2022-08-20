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
    def __init__(self, dir, mes, weight, types, baselines, **kwargs):
        super(PsevdoModel, self).__init__(**kwargs)
        weight[np.abs(mes) > 150] = 0
        mes[weight == 0] = 0
        self.dir    = tf.Variable(dir, name = 'dir', trainable=False, dtype = tf.float32)
        self.mes    = tf.Variable(mes, name = 'mes', trainable=False, dtype = tf.float32)
        self.weight = tf.Variable(weight, name = 'weight', trainable=False, dtype = tf.float32)
        self.types  = tf.Variable(types, name = 'types', trainable=False, dtype = tf.int32)
        self.gaus_kernel = gaussian_kernel(16)
        NaN = float("NaN")
        def calc_bias(pos):
            mes_loc = np.sum(dir*np.reshape(pos,(-1,1,3)), axis = -1)
            mes_loc = (mes_loc - mes)
            mes_loc[weight == 0] = NaN
            mes_loc -= np.nanmedian(mes_loc, axis = -1, keepdims=True)
            medians = np.zeros((6, len(pos)))
            for i in range(6):
                medians[i] = np.nanmedian(mes_loc[:,types == i], axis = -1)

            for i in range(len(mes[0])):
                mes_loc[:,i] -= medians[types[i]]

            mes_loc[weight == 0] = 0
            mes_loc = mes_loc*weight
            print("Initial psevdo error", np.mean(np.abs(mes_loc)))

            medians[np.isnan(medians)] = 0
            return medians



        self.bias   = tf.Variable(calc_bias(baselines), name = 'bias', trainable=True, dtype = tf.float32)
        self.epoch_bias   = tf.Variable(np.zeros((len(baselines),1)), name = 'epoch_bias', trainable=True, dtype = tf.float32)
        self.shift0  = tf.Variable(np.array([baselines[0]]), name = 'shift0', trainable=False, dtype = tf.float32)
        #self.shift_pp  = tf.Variable(np.zeros((len(mes),3)), name = 'shift_pp', trainable=True, dtype = tf.float32)
        #self.speedShift  = tf.Variable(np.zeros((3,3)), name = 'speedShift', trainable=True, dtype = tf.float32)
        #self.shift1  = tf.Variable(np.zeros((1,3)), name = 'shift1', trainable=True, dtype = tf.float32)
        #self.shift2  = tf.Variable(np.zeros((1,3)), name = 'shift2', trainable=True, dtype = tf.float32)
        #self.shift3  = tf.Variable(np.zeros((1,3)), name = 'shift3', trainable=True, dtype = tf.float32)
        #self.shift4  = tf.Variable(np.zeros((1,3)), name = 'shift4', trainable=True, dtype = tf.float32)

    def build(self, input_shape):
        super(PsevdoModel, self).build(input_shape)
    
    def get_poses(self, inpt, use_bias = True):
        poses, speeds = inpt
        pos_shift = poses + self.shift0 #+ tf.matmul(speeds,self.speedShift) #+ self.shift_pp#  #+ self.shift1*times
        return pos_shift

    def calc_median(self, mes, weights):
        mes = mes + tf.cast(weights == 0,tf.float32)*1e7
        ind = tf.reduce_sum(tf.cast(weights > 0,tf.int32), axis = -1, keepdims=True) 
        srt = tf.sort(mes, axis = -1)
        res = tf.reshape(tf.gather_nd(batch_dims=1, params=srt, indices = ind//2),(-1,1))
        return res * tf.cast(ind > 0, tf.float32)

    def call(self, inputs):
        #speeds, poses, times = inputs
        pos_shift = self.get_poses(inputs)

        bias = tf.gather(self.bias, self.types)
        bias = tf.transpose(bias)
        pos_shift = tf.reshape(pos_shift,(-1,1,3))
        mes_est = tf.reduce_sum(self.dir*pos_shift, axis = -1) + self.mes - bias - self.epoch_bias
        # median = self.calc_median(mes_est,self.weight)
        # mes_est -= median

        dir_error = mes_est*self.weight
        dir_error = tf.tanh(dir_error)
        loss = tf.reduce_mean(tf.abs(dir_error), axis = -1)


        grad = tf.reduce_sum(self.dir*dir_error[:,:,tf.newaxis], axis = 1)
        grad_mean = tf.reduce_mean(grad, axis = 0, keepdims=True)
        grad -= grad_mean
        self.shift0.assign_sub(grad_mean/10)

        #grad = callconv(grad,self.gaus_kernel)
        #grad = tf.reshape(tf.nn.avg_pool1d(tf.reshape(grad,(1,-1,3)),30,1,'SAME'),(-1,3))
        # grad = grad[::-1]
        # grad = tf.cumsum(grad,axis = 0)
        # grad = grad[::-1]/100
        #grad = grad/(tf.maximum(tf.linalg.norm(grad, axis = 0),1))
        grad = grad[1:] - grad[:-1]
        return tf.reduce_mean(loss)\
             + tf.reduce_mean(tf.reduce_sum(tf.square(self.bias[:,1:]-self.bias[:,:-1]), axis = 0))*1e-2\
            , grad
        #return loss[1:] + tf.reduce_sum(tf.abs(self.bias[:,1:]-self.bias[:,:-1]), axis = 0)/10, grad #+ tf.reduce_sum(tf.sqrt(tf.abs(self.shift_pp[1:]-self.shift_pp[:-1])), axis = -1)/10

    def compute_output_shape(self, _):
        return (1)
    def get_config(self):
        config = super().get_config().copy()
        return config        

def createPsevdoModel(sat_directions, sat_psevdoshifts, weights, types, baselines):
    return PsevdoModel(sat_directions, sat_psevdoshifts, weights, types, baselines)
    
