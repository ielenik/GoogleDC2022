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


@tf.custom_gradient
def my_norm(x):
    y = tf.linalg.norm(x, axis = -1)
    def grad(dy):
        return tf.expand_dims(dy,-1) * (x / (tf.expand_dims(y + 1e-19,-1)))

    return y, grad

def running_median_insort(seq, window_size):
    """Contributed by Peter Otten"""
    seq = iter(seq)
    d = deque()
    s = []
    result = []
    for item in islice(seq, window_size):
        d.append(item)
        insort(s, item)
        result.append(s[len(d)//2])
    m = window_size // 2
    for item in seq:
        old = d.popleft()
        d.append(item)
        del s[bisect_left(s, old)]
        insort(s, item)
        result.append(s[m])
    return result

def mult_np(a, b):
    return np.array([a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],  
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],  
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]])

def mult(a, b):
    return tf.stack([a[:,3] * b[:,0] + a[:,0] * b[:,3] + a[:,1] * b[:,2] - a[:,2] * b[:,1],  
        a[:,3] * b[:,1] - a[:,0] * b[:,2] + a[:,1] * b[:,3] + a[:,2] * b[:,0],  
        a[:,3] * b[:,2] + a[:,0] * b[:,1] - a[:,1] * b[:,0] + a[:,2] * b[:,3],
        a[:,3] * b[:,3] - a[:,0] * b[:,0] - a[:,1] * b[:,1] - a[:,2] * b[:,2]], axis = 1)

def inv(a):
    return tf.stack([-a[:,0], -a[:,1], -a[:,2], a[:,3]], axis = 1)

def inv_np(a):
    return [-a[0], -a[1], -a[2], a[3]]

def to_quat(omega, dt = 1):
    omegaMagnitude = np.linalg.norm(omega)
    if (omegaMagnitude < 0.00001):
        omegaMagnitude = 0.00001

    thetaOverTwo = omegaMagnitude * dt / 2.0
    sinThetaOverTwo = math.sin(thetaOverTwo) / omegaMagnitude
    cosThetaOverTwo = math.cos(thetaOverTwo) 

    return np.array([sinThetaOverTwo * omega[0], sinThetaOverTwo * omega[1], sinThetaOverTwo * omega[2], cosThetaOverTwo])
def get_quat(v1,v2):
    v1 = v1/np.linalg.norm(v1)
    v2 = v2/np.linalg.norm(v2)
    res = np.zeros(4)
    res[:3] = np.cross(v1,v2)
    if np.dot(v1, v2) < - 0.999:
        res[:3] = [0.,0.,1.]
    res[3] = 1 + np.dot(v1, v2)
    res = res/np.linalg.norm(res)
    return res

def transform(value, rotation):
    num12 = rotation[:,0] + rotation[:,0]
    num2 = rotation[:,1] + rotation[:,1]
    num = rotation[:,2] + rotation[:,2]
    num11 = rotation[:,3] * num12
    num10 = rotation[:,3] * num2
    num9 = rotation[:,3] * num
    num8 = rotation[:,0] * num12
    num7 = rotation[:,0] * num2
    num6 = rotation[:,0] * num
    num5 = rotation[:,1] * num2
    num4 = rotation[:,1] * num
    num3 = rotation[:,2] * num
    num15 = ((value[:,0] * ((1. - num5) - num3)) + (value[:,1] * (num7 - num9))) + (value[:,2] * (num6 + num10))
    num14 = ((value[:,0] * (num7 + num9)) + (value[:,1] * ((1. - num8) - num3))) + (value[:,2] * (num4 - num11))
    num13 = ((value[:,0] * (num6 - num10)) + (value[:,1] * (num4 + num11))) + (value[:,2] * ((1. - num8) - num5))
    return tf.stack([num15,num14,num13], axis = 1)


def transform_np(value, rotation):
    num12 =rotation[0] + rotation[0]
    num2 = rotation[1] + rotation[1]
    num =  rotation[2] + rotation[2]
    num11 =rotation[3] * num12
    num10 =rotation[3] * num2
    num9 = rotation[3] * num
    num8 = rotation[0] * num12
    num7 = rotation[0] * num2
    num6 = rotation[0] * num
    num5 = rotation[1] * num2
    num4 = rotation[1] * num
    num3 = rotation[2] * num
    num15 = ((value[0] * ((1. - num5) - num3)) + (value[1] * (num7 - num9))) + (value[2] * (num6 + num10))
    num14 = ((value[0] * (num7 + num9)) + (value[1] * ((1. - num8) - num3))) + (value[2] * (num4 - num11))
    num13 = ((value[0] * (num6 - num10)) + (value[1] * (num4 + num11))) + (value[2] * ((1. - num8) - num5))
    return np.array([num15,num14,num13])


class MagnetCallibration(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MagnetCallibration, self).__init__(**kwargs)
    def build(self, input_shape):
        super(MagnetCallibration, self).build(input_shape)

        self.hard_iron = self.add_weight(name='hard_iron', shape=(1, 3), trainable=True, initializer=tf.keras.initializers.Constant(value=[[1,1,1]]))
        self.soft_iron = self.add_weight(name='soft_iron', shape=(6), trainable=True, initializer=tf.keras.initializers.Constant(value=[1,1,1, 0,0,0]))
    def call(self, mag_input):
        mag_input = mag_input - self.hard_iron
        
        magx = mag_input[:,0]*mag_input[:,0]*self.soft_iron[0] + mag_input[:,0]*mag_input[:,1]*self.soft_iron[3] + mag_input[:,0]*mag_input[:,2]*self.soft_iron[4]
        magy = mag_input[:,1]*mag_input[:,0]*self.soft_iron[3] + mag_input[:,1]*mag_input[:,1]*self.soft_iron[1] + mag_input[:,0]*mag_input[:,2]*self.soft_iron[5]
        magz = mag_input[:,2]*mag_input[:,0]*self.soft_iron[4] + mag_input[:,2]*mag_input[:,1]*self.soft_iron[5] + mag_input[:,2]*mag_input[:,2]*self.soft_iron[2]

        self.add_loss(tf.abs(1-magx - magy - magz))
        return 1-magx - magy - magz
