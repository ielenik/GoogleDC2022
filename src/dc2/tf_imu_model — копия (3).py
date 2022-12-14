from collections import deque
from bisect import insort, bisect_left
from itertools import islice
from cv2 import norm, rotate
import numpy as np
import tensorflow as tf
import math
import random
import matplotlib.pyplot as plt
from ..utils.magnet_model import Magnetometer
from ..utils.tensorflow_geometry import*

from tensorflow.python.framework import function
from tensorflow.python.framework import function
from .tf_numpy_tools import *

class RigidModelImu(tf.keras.layers.Layer):
    def __init__(self, acs, gyr, mat, **kwargs):
        super(RigidModelImu, self).__init__(**kwargs)
        self.acs  = tf.Variable(acs, name = 'acs', trainable=False, dtype = tf.float32)
        self.gyr  = tf.Variable(gyr, name = 'gyr', trainable=False, dtype = tf.float32)

        self.down  = tf.Variable([[0.,0.,-1.]], trainable=False, dtype = tf.float32, name = 'down')
        self.fwd  = tf.Variable([[0.,1.,0.]], trainable=False, dtype = tf.float32, name = 'fwd')
        self.right= tf.Variable([[1.,0.,0.]], trainable=True, dtype = tf.float32, name = 'right')
        self.idquat = tf.Variable([[0.,0.,0.,1.]], trainable=False, dtype = tf.float32, name = 'idquat')
        self.north = self.fwd

        self.g      = tf.Variable([[9.81]], name = 'g', trainable=False, dtype = tf.float32)
        self.g_scale      = tf.Variable([[0.99]], name = 'g_scale', trainable=True, dtype = tf.float32)

        #self.ort   = tf.Variable(np.ones((len(acs),4))*np.array([get_quat([0,1,0], [0,0,1])]), name = 'orientaiton', trainable=True, dtype = tf.float32)
        self.ort    = tf.Variable(mat, name = 'orientaiton', trainable=False, dtype = tf.float32)
        conv_init = np.zeros((8,1,1))
        conv_init[3:5] = 1
        # self.conv_filters_gyro  = tf.Variable(conv_init, name = 'conv_filters_gyro', trainable=True, dtype = tf.float32)
        # self.conv_filters_acel  = tf.Variable(conv_init, name = 'conv_filters_acel', trainable=True, dtype = tf.float32)
        #self.conv_filters  = tf.Variable(conv_init, name = 'conv_filters', trainable=True, dtype = tf.float32)
        self.conv_filters_gyro  = tf.Variable(conv_init, name = 'conv_filters_gyro', trainable=True, dtype = tf.float32)
        self.conv_filters_acel  = tf.Variable(conv_init, name = 'conv_filters_acel', trainable=True, dtype = tf.float32)
        self.gyro_bias = tf.Variable(np.zeros(((len(gyr)-6)//2,3)), name = 'gyro_bias', trainable=True, dtype = tf.float32)
        self.accel_bias = tf.Variable([[0,0,0]], name = 'accel_bias', trainable=True, dtype = tf.float32)
        #self.gyro_scale = tf.Variable([[1,1,1]], name = 'gyro_scale', trainable=True, dtype = tf.float32)
        
        gyr = self.callconv_acel(self.gyr)
        gyr = tf.linalg.norm(gyr*[5,0,5], axis = -1)
        #acs = tf.reshape(acs, (-1,))
        #self.acs_error_scale = tf.Variable(1/(1+acs*5), name = 'acs_error_scale', trainable=False, dtype = tf.float32)
        # self.excluded_epoch = tf.Variable(tf.concat([[0],tf.cast(gyr < 1.0,tf.float32)],axis=0), name = 'acs_error_scale', trainable=False, dtype = tf.float32)
        # self.excluded_epoch = tf.Variable(tf.concat([[0],tf.cast(gyr < 1.0,tf.float32)],axis=0), name = 'acs_error_scale', trainable=False, dtype = tf.float32)

    def build(self, input_shape):
        super(RigidModelImu, self).build(input_shape)


    def rotate_z(self, vec, angle):
        q = tf.stack([tf.math.cos(angle)*vec[:,0]-tf.math.sin(angle)*vec[:,1],tf.math.cos(angle)*vec[:,1]+tf.math.sin(angle)*vec[:,0] ,tf.ones_like(angle)*vec[:,2]], axis = 1)
        return q
    def rotate_x(self, vec, angle):
        q = tf.stack([tf.ones_like(angle)*vec[:,0], tf.math.cos(angle)*vec[:,1]-tf.math.sin(angle)*vec[:,2],tf.math.cos(angle)*vec[:,2]+tf.math.sin(angle)*vec[:,1]], axis = 1)
        return q
    def rotate_y(self, vec, angle):
        q = tf.stack([tf.math.cos(angle)*vec[:,0]+tf.math.sin(angle)*vec[:,2],tf.ones_like(angle)*vec[:,1], tf.math.cos(angle)*vec[:,2]-tf.math.sin(angle)*vec[:,0]], axis = 1)
        return q
    def get_fwd(self, omega):
        return self.rotate_z(self.fwd, omega[:,2])
    def get_right(self, omega):
        self.right.assign(tf.linalg.l2_normalize(self.right))
        return self.rotate_z(self.right, omega[:,2])
    def get_up(self, omega):
        q = tf.stack([-tf.math.sin(omega[:,1]),-tf.math.cos(omega[:,1])*tf.math.sin(omega[:,0]),-tf.math.cos(omega[:,1])*tf.math.cos(omega[:,0])], axis = 1)
        return q

    def callconv2(self, inputs, conv_filters, padding = 'SAME', scale = 1):
        inputs = tf.expand_dims(tf.transpose(inputs),axis=-1)
        inputs = tf.nn.conv1d(inputs, conv_filters, scale, padding = padding)
        inputs = tf.squeeze(inputs)
        return tf.transpose(inputs)


    def callconv_gyro(self, inputs):
        conv_filters = self.conv_filters_gyro
        conv_filters = conv_filters*2/tf.reduce_sum(conv_filters)
        return self.callconv2(inputs,conv_filters,padding = 'VALID',scale=2)+self.gyro_bias/100
    def callconv_acel(self, inputs):
        conv_filters = self.conv_filters_acel
        conv_filters = conv_filters*2/tf.reduce_sum(conv_filters)
        return self.callconv2(inputs,conv_filters,padding = 'VALID',scale=2)+self.accel_bias/100

    def get_local_acs(self, quats):
        orientation = tf.linalg.l2_normalize(self.ort, axis = 0)
        
        acs = self.callconv_acel(self.acs)
        acsl = tf.linalg.matvec(orientation, acs)
        quats = (quats[1:]+quats[:-1])/2
        acsl = transform(acsl, quats)
        return acsl

    def call(self, inputs):
        speed, angles_cum, times_dif, rotate_gyro = inputs
        speed = speed*1000/times_dif
        angles_cum_mid = (angles_cum[1:]+angles_cum[:-1])/2
        quats = quat_from_euler(angles_cum)
        acsi = self.get_local_acs(quats)

        acsi = acsi*self.g_scale  + self.down*self.g
        speed_loc = tf.cumsum(acsi, axis = 0)
        speed_loc = tf.concat([[[0,0,0]],speed_loc], axis = 0)

        acs_grad =  speed - speed_loc  
        acs_grad_average  = tf.reshape(tf.nn.avg_pool1d(tf.reshape(acs_grad,(1,-1,3)),64,1,'SAME'),(-1,3))
        acs_grad  = acs_grad - acs_grad_average
        acs_loss = tf.reduce_sum(tf.abs(acs_grad)/1000+tf.square(acs_grad), axis = -1)
        #acs_loss = tf.reduce_sum(tf.square(acsi - acsr)+tf.abs(acsi - acsr)/1000, axis = -1)
 
        right_dir = self.get_right(angles_cum)
        speed_grad = tf.reduce_sum(speed*right_dir, axis = -1)
        speed_loss = tf.square(speed_grad)
        speed_grad = tf.reshape(speed_grad,(-1,1))*right_dir


        pred_gyr = -mult(inv(quats[1:]),quats[:-1])[:,:3]
        #gyrl_grad_average  = tf.reshape(tf.nn.avg_pool1d(tf.reshape(gyrl_grad,(1,-1,3)),64,1,'SAME'),(-1,3))
        #gyrl_grad = gyrl_grad - gyrl_grad_average
        gyrl = self.callconv_gyro(self.gyr)
        gyrl = tf_to_quat(gyrl)[:,:3]
        quat_loss = tf.reduce_sum(tf.square(pred_gyr - gyrl), axis = -1)*100 + 100*tf.reduce_mean(tf.square(self.gyro_bias[1:]-self.gyro_bias[:-1]))

        # pred_gyr_cum = tf.cumsum(pred_gyr, axis = 0)
        # gyrl_cum = tf.cumsum(gyrl, axis = 0)
        # gyrl_grad =  pred_gyr_cum - gyrl_cum
        # gyrl_grad_average  = tf.reshape(tf.nn.avg_pool1d(tf.reshape(gyrl_grad,(1,-1,3)),64,1,'SAME'),(-1,3))
        # gyrl_grad = gyrl_grad - gyrl_grad_average
        # quat_loss = tf.reduce_sum(tf.square(gyrl_grad), axis = -1)*100 + 100*tf.reduce_mean(tf.square(self.gyro_bias[1:]-self.gyro_bias[:-1]))


        quat_loss = tf.concat([[0],quat_loss], axis = 0)

        #quat_loss = quat_loss*1e-5 + tf.concat([[0],my_norm(angles_cum[1:] - angles_cum[:-1])], axis = 0)

        #quat_loss = tf.reduce_sum(tf.square(angles-gyrl), axis = -1)*100
        # quat_loss = tf.reduce_sum(tf.square(angles-gyrl)[:,2:], axis = -1)*1000\
        #     + tf.reduce_sum(tf.square(angles)[:,:2], axis = -1)*10

        gyr_stable = my_norm(gyrl)
        acs_stable = my_norm(acsi)
        spd_stable = my_norm(speed[1:]+speed[:-1])
        stable_poses = tf.logical_and(tf.logical_and(gyr_stable<1e-3,acs_stable<1e-2),spd_stable<1e-1)
        stable_poses = tf.cast(stable_poses,tf.float32)
        
        return acs_loss, acs_grad, quat_loss, speed_loss, speed_grad, self.g_scale, stable_poses

    def get_angles(self, inputs):
        speed, angles_cum = inputs
        angles_cum_mid = (angles_cum[1:]+angles_cum[:-1])/2
        quats = quat_from_euler(angles_cum)
        gyrl = self.callconv_gyro(self.gyr)
        return tf_to_quat(gyrl), -mult(inv(quats[1:]),quats[:-1])


    def get_acses(self, inputs):
        speed, speedgt,  angles_cum, times_dif = inputs
        speed = speed*1000/times_dif
        speedgt = speedgt*1000/times_dif
        angles_cum_mid = (angles_cum[1:]+angles_cum[:-1])/2
        quats = quat_from_euler(angles_cum)
        acsl = self.get_local_acs(quats)
        acsr = speed[1:] - speed[:-1]
        acst = speedgt[1:] - speedgt[:-1]
        acsi = acsl*self.g_scale + self.down*self.g
        # acsi = self.rotate_z(acsi,-angles_cum_mid[:,2])
        # acsr = self.rotate_z(acsr,-angles_cum_mid[:,2])
        # acst = self.rotate_z(acst,-angles_cum_mid[:,2])

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

def createRigidModel(epochtimes, acsvalues,acstimes,gyrvalues,gyrtimes):

    nummesures = len(epochtimes)*10+10 # ten measrurment per epoch + 2 sec window for convolution

    gyrvalues -= np.median(gyrvalues, axis = 0)
    epochtimesdif = (epochtimes[1:] - epochtimes[:-1])/2
    epochtimesdif = np.tile(epochtimesdif,(2,1))
    epochtimesdif = epochtimesdif.T.reshape(-1)
    epochtimesnew = np.cumsum(epochtimesdif)
    epochtimesnew = np.concatenate([[-1000,-500,0],
                                    epochtimesnew,
                                    np.array([500, 1000])+epochtimesnew[-1]
                                    ])+epochtimes[0]


    epochtimes = epochtimesnew

    # gyrvalues[np.linalg.norm(gyrvalues,axis=-1) < 0.0001] = 0

    gyrvalues = gyrvalues[1:]
    gyrvalues = gyrvalues*(gyrtimes[1:]-gyrtimes[:-1]).reshape((-1,1))/1000
    gyrtimes = gyrtimes[1:]

    acsvalues = acsvalues[1:]
    acsvalues = acsvalues*(acstimes[1:]-acstimes[:-1]).reshape((-1,1))/1000
    acstimes = acstimes[1:]

    gyrcum = np.cumsum(gyrvalues, axis=0)
    acscum = np.cumsum(acsvalues, axis=0)

    def interp_nd(epochtimes, valtimes, valcum):
        dim = []
        for i in range(len(valcum[0])):
            dim.append(np.interp(epochtimes, valtimes, valcum[:,i]))
        return np.array(dim).T

    gyr_epochs = interp_nd(epochtimes, gyrtimes, gyrcum)
    acs_epochs = interp_nd(epochtimes, acstimes, acscum)

    gyr_epochs = gyr_epochs[1:] - gyr_epochs[:-1]
    acs_epochs = acs_epochs[1:] - acs_epochs[:-1]
    acs_epochs[0:6] = acs_epochs[10]

    return RigidModelImu(acs_epochs, gyr_epochs, np.array([[1.,0.,0],[0.,1.,0.],[0.,0.,1]]))
    
