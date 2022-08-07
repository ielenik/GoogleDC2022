from collections import deque
from bisect import insort, bisect_left
from itertools import islice
from cv2 import rotate
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
    def __init__(self, acs, gyr, dn, **kwargs):
        super(RigidModelImu, self).__init__(**kwargs)
        self.angle_scale = 1e-1

        self.acs  = tf.Variable(acs, name = 'acs', trainable=False, dtype = tf.float32)
        self.gyr  = tf.Variable(gyr, name = 'gyr', trainable=False, dtype = tf.float32)

        self.down  = tf.Variable([[0.,0.,-1.]], trainable=False, dtype = tf.float32, name = 'down')
        self.fwd  = tf.Variable([[0.,1.,0.]], trainable=False, dtype = tf.float32, name = 'fwd')
        self.idquat = tf.Variable([[0.,0.,0.,1.]], trainable=False, dtype = tf.float32, name = 'idquat')
        self.north = self.fwd

        self.g      = tf.Variable([[9.81]], name = 'g', trainable=False, dtype = tf.float32)
        self.g_scale      = tf.Variable([[1]], name = 'g_scale', trainable=True, dtype = tf.float32)
        #self.start_dir = tf.Variable([[0,0,0]], name = 'start_dir', trainable=True, dtype = tf.float32)
        #self.ort   = tf.Variable(np.ones((len(acs),4))*np.array([get_quat([0,1,0], [0,0,1])]), name = 'orientaiton', trainable=True, dtype = tf.float32)
        self.ort    = tf.Variable(np.array([get_quat(dn, [0,0,1])]), name = 'orientaiton', trainable=True, dtype = tf.float32)
        #self.gyro_bias = tf.Variable([[0,0,0]], name = 'gyro_bias', trainable=True, dtype = tf.float32)

    def build(self, input_shape):
        super(RigidModelImu, self).build(input_shape)

    def to_quat_local1(self, omega):
        omegaMagnitude = tf.linalg.norm(omega, axis = -1, keepdims = True)
        thetaOverTwo = omegaMagnitude / 2.0
        sinThetaOverTwo = tf.math.sin(thetaOverTwo) / (omegaMagnitude + 0.00001)
        cosThetaOverTwo = tf.math.cos(thetaOverTwo) 
        return tf.concat([sinThetaOverTwo * omega, cosThetaOverTwo], axis = -1)

    def ToQuaternion(self, roll, pitch, yaw):# // yaw (Z), pitch (Y), roll (X)
        cy = tf.math.cos(yaw * 0.5)
        sy = tf.math.sin(yaw * 0.5)
        cp = tf.math.cos(pitch * 0.5)
        sp = tf.math.sin(pitch * 0.5)
        cr = tf.math.cos(roll * 0.5)
        sr = tf.math.sin(roll * 0.5)

        q = tf.stack([sr * cp * cy - cr * sp * sy,
                        cr * sp * cy + sr * cp * sy,
                        cr * cp * sy - sr * sp * cy,
                        cr * cp * cy + sr * sp * sy], axis = 1)

        return q

    def rotate_z(self, vec, angle):
        q = tf.stack([tf.math.cos(angle)*vec[:,0]-tf.math.sin(angle)*vec[:,1],tf.math.cos(angle)*vec[:,1]+tf.math.sin(angle)*vec[:,0] ,tf.ones_like(angle)*vec[:,2]], axis = 1)
        return q
    def get_fwd(self, omega):
        #q = tf.stack([-tf.math.sin(omega[:,2]),tf.math.cos(omega[:,2]),tf.zeros_like(omega[:,2])], axis = 1)
        return self.rotate_z(self.fwd, omega[:,2])
    def get_up(self, omega):
        q = tf.stack([-tf.math.sin(omega[:,1]),-tf.math.cos(omega[:,1])*tf.math.sin(omega[:,0]),-tf.math.cos(omega[:,1])*tf.math.cos(omega[:,0])], axis = 1)
        return q

    def to_quat_local(self, omega):
        return self.ToQuaternion(omega[:,0],omega[:,1],omega[:,2])

    def call(self, inputs):
        speed, angles, rotate_gyro = inputs
        angles = angles * self.angle_scale

        orientation = tf.linalg.l2_normalize(self.ort, axis = -1)
        angles_cum  = tf.cumsum(angles, axis = 0) + self.start_dir
        angles = angles[1:]

        angles_cum_mid = (angles_cum[1:]+angles_cum[:-1])/2
        acsl = transform(self.acs, orientation)
        gyrl = transform(self.gyr, orientation)
        #gyrl = transform(self.gyr + self.gyro_bias*1e-6, orientation)
        if rotate_gyro:
            acsl = self.rotate_z(acsl,angles_cum_mid[:,2])
            gyrl = self.rotate_z(gyrl,angles_cum_mid[:,2])
        acsi = acsl

        acsr = speed[1:] - speed[:-1]
        acsi = acsi*self.g_scale  + self.down*self.g
        acs_loss = tf.reduce_sum(tf.abs(acsi - acsr), axis = -1) * 1e-2
 
        speed_loss = ((my_norm(speed) - tf.reduce_sum(speed*self.get_fwd(angles_cum), axis = -1)))
        speed_loss = (speed_loss[1:]+speed_loss[:-1])*1e-5

        if not rotate_gyro:
            speed_loss = speed_loss * 1e-6

        quat_loss = tf.reduce_sum(tf.abs(angles-gyrl), axis = -1)*10
        quat_loss += (tf.square(angles_cum[1:,1])+tf.square(angles_cum[1:,0]))*1e-2

        stable_poses = tf.reduce_sum(tf.square(self.gyr),axis=-1)
        speed_norm = my_norm(speed)[:-1]
        stable_poses = tf.cast(stable_poses<1e-6,tf.float32)*tf.cast(speed_norm<0.1,tf.float32)
        
        return acs_loss, quat_loss, speed_loss, self.g_scale, stable_poses

    def get_angles(self, inputs):
        sp, angles = inputs
        angles = angles[1:]

        orientation = tf.linalg.l2_normalize(self.ort, axis = -1)
        gyrl = transform(self.gyr, orientation)

        return gyrl, angles*self.angle_scale


    def get_acses(self, inputs):
        speed, speedgt, angles = inputs

        orientation = tf.linalg.l2_normalize(self.ort, axis = -1)
        acsl = transform(self.acs, orientation)
        gyrl = transform(self.gyr, orientation)

        acsr = speed[1:] - speed[:-1]
        acst = speedgt[1:] - speedgt[:-1]

        angles_cum  = tf.cumsum(angles, axis = 0)
        angles_cum_mid = (angles_cum[1:]+angles_cum[:-1])/2
        acsl = self.rotate_z(acsl,angles_cum_mid[:,2])
        acsi = acsl + self.down*self.g


        acsi = self.rotate_z(acsi,angles_cum_mid[:,2])
        acsr = self.rotate_z(acsr,angles_cum_mid[:,2])
        acst = self.rotate_z(acst,angles_cum_mid[:,2])

        return acsi, acst, acsr

    def compute_output_shape(self, _):
        return (1)
    def get_config(self):
        config = super().get_config().copy()
        return config        

from scipy import ndimage, misc
def running_median_insort(seqin, window_size):
    """Contributed by Peter Otten"""
    return ndimage.median_filter(seqin,size=(window_size,1))

def createRigidModel(epochtimes, logs):

    acs = logs[logs['MessageType'] == 'UncalAccel']
    gir = logs[logs['MessageType'] == 'UncalGyro']

    gyrtimes = gir['utcTimeMillis'].to_numpy()
    gyrvalues = gir[['MeasurementX','MeasurementY','MeasurementZ']].to_numpy()
    acstimes = acs['utcTimeMillis'].to_numpy()
    acsvalues = acs[['MeasurementX','MeasurementY','MeasurementZ']].to_numpy()
    nummesures = len(epochtimes)
    
    quat_epochs = np.zeros((nummesures,4))
    gyr_epochs  = np.zeros((nummesures,3))
    acs_epochs  = np.zeros((nummesures,3))
    quat_epochs[:,3] = 1

    def accum_values(times,values,epochs):
        j = 0
        counts = np.zeros((len(epochs)))
        for i in range(1, len(times)):
            while j < len(epochtimes) and times[i] > epochtimes[j]:
                j += 1
            if j >= len(epochtimes): 
                break
            epochs[j] += values[i]
            counts[j] += 1
        counts[counts == 0] = 1
        epochs /= np.reshape(counts,(-1,1))

    print("Filtering acseleration...")
    acsvalues = running_median_insort(acsvalues,100)
    # print("Filtering gyro...")
    # running_median_insort(gyrvalues,30)
    # covmat = np.cov(acsvalues.T)
    # sc, vecs = np.linalg.eig(covmat)
    # ind = np.argmax(sc)
    # pr = np.sum(np.array([vecs[ind]])*acsvalues)
    # if pr < 0:
    #     vecs[ind] = -vecs[ind]
    mean = np.mean(acsvalues, axis=0)
    mean = np.reshape(mean/np.linalg.norm(mean),(1,3))
    acsvalues_nog = acsvalues - mean#*np.sum(mean*acsvalues, axis = -1, keepdims= True)

    covmat = np.cov(acsvalues_nog.T)
    sc, vecs = np.linalg.eig(covmat.T)
    ind = np.argmax(sc)
    pr = np.sum(np.array([vecs[ind]])*acsvalues)
    if pr < 0:
        vecs[ind] = -vecs[ind]


    accum_values(acstimes - 600 - 500,acsvalues,acs_epochs)
    accum_values(gyrtimes - 600 - 500,gyrvalues,gyr_epochs)

    # for i in range(nummesures):
    #     quat_epochs[i] = to_quat(gyr_epochs[i], 1)

    gyr_epochs = gyr_epochs[1:-1]
    acs_epochs = acs_epochs[1:-1]

    return RigidModelImu(acs_epochs, gyr_epochs, mean)
    
