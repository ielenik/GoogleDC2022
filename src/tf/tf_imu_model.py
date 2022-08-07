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


class RigidModelImu(tf.keras.layers.Layer):
    def __init__(self, acs, gyr, mag, **kwargs):
        super(RigidModelImu, self).__init__(**kwargs)
        self.acs  = tf.Variable(acs, name = 'acs', trainable=False, dtype = tf.float32)
        self.gyr  = tf.Variable(gyr, name = 'gyr', trainable=False, dtype = tf.float32)
        self.mag  = tf.Variable(mag, name = 'mag', trainable=False, dtype = tf.float32)

        self.down  = tf.Variable([[0.,0.,-1.]], trainable=False, dtype = tf.float32, name = 'down')
        self.fwd  = tf.Variable([[0.,1.,0.]], trainable=False, dtype = tf.float32, name = 'fwd')
        self.idquat = tf.Variable([[0.,0.,0.,1.]], trainable=False, dtype = tf.float32, name = 'idquat')
        self.north = self.fwd

        self.g      = tf.Variable([[9.8]], name = 'g', trainable=True, dtype = tf.float32)
        self.ort    = tf.Variable([get_quat([0,1,0], [0,0,1])], name = 'orientaiton', trainable=True, dtype = tf.float32)

    def build(self, input_shape):
        super(RigidModelImu, self).build(input_shape)

    def call(self, inputs):
        poses, quats = inputs
        poses = tf.reshape(poses,(-1,3))
        orientation, _ = tf.linalg.normalize(self.ort, axis = -1)

        speed = poses[1:] - poses[:-1]
        acsr = speed[1:] - speed[:-1]

        qt1 = quats[1:]
        qt2 = quats[:-1]

        north_pred = transform(self.mag,orientation)
        north_pred = transform(north_pred,quats)
        north, _ = tf.linalg.normalize(self.north,axis = -1)
        mag_loss = tf.reduce_mean(my_norm((north - north_pred)[:,:2]))

        
        acsi = transform(self.acs,orientation)
        acsi = transform(acsi, quats)
        acsi = acsi + self.down*self.g


        acs_loss = tf.reduce_mean(my_norm(acsi[1:-1] - acsr))
        #speed_loss = tf.reduce_mean((my_norm(speed) - tf.reduce_sum(speed*transform(self.fwd,qt1), axis = -1))/(my_norm(speed) + 1))
        speed_loss = tf.reduce_mean((my_norm(speed) - tf.reduce_sum(speed*transform(self.fwd,qt1), axis = -1)))

        iqt1 = inv(qt1)
        true_quat = mult(iqt1, qt2)
        local_gyr = mult(mult(inv(orientation), self.gyr), orientation)
        quat_loss = tf.reduce_mean(1 - tf.square(tf.reduce_sum(true_quat*local_gyr[:-1], axis = -1)))*10
        
        return acs_loss, quat_loss, mag_loss,  speed_loss, self.g

    def compute_output_shape(self, _):
        return (1)
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'poses' : self.poses,
            'dires' : self.dires,
        })
        return config        



def createRigidModel(epochtimes, logs):

    acs = logs['acs']
    gir = logs['gir']
    mag = logs['mag']

    gyrtimes = gir['utcTimeMillis'].to_numpy()
    gyrvalues = gir[['UncalGyroXRadPerSec','UncalGyroYRadPerSec','UncalGyroZRadPerSec']].to_numpy()
    acstimes = acs['utcTimeMillis'].to_numpy()
    acsvalues = acs[['UncalAccelXMps2','UncalAccelYMps2','UncalAccelZMps2']].to_numpy()
    magtimes = mag['utcTimeMillis'].to_numpy()
    magvalues = mag[['UncalMagXMicroT','UncalMagYMicroT','UncalMagZMicroT']].to_numpy()

    nummesures = len(epochtimes)
    
    quat_epochs = np.zeros((nummesures,4))
    gyr_epochs  = np.zeros((nummesures,3))
    acs_epochs  = np.zeros((nummesures,3))
    mag_epochs  = np.zeros((nummesures,3))
    mag_epochs[:,2] = 1
    quat_epochs[:,3] = 1


    acstimes += -315964800000 + 18000
    gyrtimes += -315964800000 + 18000
    magtimes += -315964800000 + 18000

    def accum_values(times,values,epochs):
        j = 0
        for i in range(1, len(times)):
            while j < len(epochtimes) and times[i]*1000000 > epochtimes[j]:
                j += 1
            if j >= len(epochtimes): 
                break
            dt = (times[i] - times[i-1])/1000
            epochs[j] += values[i]*dt

    #magvalues -= np.mean(magvalues, axis = 0, keepdims = True)
    #magvalues /= np.mean(np.linalg.norm(magvalues,axis=-1))

    accum_values(acstimes,acsvalues,acs_epochs)
    accum_values(magtimes,magvalues,mag_epochs)
    accum_values(gyrtimes,gyrvalues,gyr_epochs)

    for i in range(nummesures):
        quat_epochs[i] = to_quat(gyr_epochs[i], 1)

    #mag_epochs -= np.mean(mag_epochs, axis = 0, keepdims = True)
    #mag_epochs /= np.mean(np.linalg.norm(mag_epochs,axis=-1))


    #mag1 = Magnetometer()
    #mag1.calibrate(mag_epochs)
    mag = Magnetometer()
    mag.calibrate(magvalues)
    mag_epochs = (mag_epochs - mag.b.reshape((1,3))).dot(mag.A)
    mag_epochs = mag_epochs/np.linalg.norm(mag_epochs, axis = -1, keepdims=True)

    return RigidModelImu(acs_epochs, quat_epochs, mag_epochs)
    
