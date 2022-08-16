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
from ..utils.predict_angle import predict_angle

from tensorflow.python.framework import function
from tensorflow.python.framework import function
from .tf_numpy_tools import *

class RigidModelImu(tf.keras.layers.Layer):
    def __init__(self, acs, gyr, orientations, fps, **kwargs):
        super(RigidModelImu, self).__init__(**kwargs)
        self.fps  = tf.Variable(fps, name = 'fps', trainable=False, dtype = tf.float32)
        self.acs  = tf.Variable(acs, name = 'acs', trainable=False, dtype = tf.float32)/self.fps
        self.gyr  = tf.Variable(gyr, name = 'gyr', trainable=False, dtype = tf.float32)/self.fps

        self.down  = tf.Variable([[0.,0.,-1.]], trainable=False, dtype = tf.float32, name = 'down')
        self.fwd  = tf.Variable([[0.,1.,0.]], trainable=False, dtype = tf.float32, name = 'fwd')
        self.right_ang = tf.Variable(0, trainable=True, dtype = tf.float32, name = 'right')
        self.idquat = tf.Variable([[0.,0.,0.,1.]], trainable=False, dtype = tf.float32, name = 'idquat')
        self.north = self.fwd

        self.angles  = tf.Variable(orientations, name = 'angles', trainable=True, dtype = tf.float32)
        self.g      = tf.Variable([[9.81]], name = 'g', trainable=True, dtype = tf.float32)
        # self.g_scale      = tf.Variable([[1.00]], name = 'g_scale', trainable=False, dtype = tf.float32)
        #self.time_shift_acel      = tf.Variable(0.6, name = 'acel_timeshift', trainable=True, dtype = tf.float32)

        #self.gyro_bias = tf.Variable(np.zeros_like(gyr), name = 'gyro_bias', trainable=True, dtype = tf.float32)
        self.accel_bias = tf.Variable([[0,0,0]], name = 'accel_bias', trainable=True, dtype = tf.float32)

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
        
        right = tf.stack([tf.cos(self.right_ang), tf.sin(self.right_ang), 0])
        return self.rotate_z(right[tf.newaxis,:], omega[:,2])
    def get_up(self, omega):
        q = tf.stack([-tf.math.sin(omega[:,1]),-tf.math.cos(omega[:,1])*tf.math.sin(omega[:,0]),-tf.math.cos(omega[:,1])*tf.math.cos(omega[:,0])], axis = 1)
        return q

    def callconv2(self, inputs, conv_filters, padding = 'SAME', scale = 1):
        inputs = tf.expand_dims(tf.transpose(inputs),axis=-1)
        inputs = tf.nn.conv1d(inputs, conv_filters, scale, padding = padding)
        inputs = tf.squeeze(inputs)
        return tf.transpose(inputs)


    def get_global_acs(self):
        return tf.linalg.matvec(mat_from_euler(self.angles), self.acs+self.accel_bias)

    def get_local_gyr(self, gyrpred):
        return self.gyr, tf.linalg.matvec(rot_mat_from_euler((self.angles[1:]+self.angles[:-1])/2), gyrpred)

    def call(self, inputs):
        speeds30hz = inputs

        acsi = self.get_global_acs()
        acsi = acsi  + self.down*self.g/self.fps
        acsg = speeds30hz[1:] - speeds30hz[:-1]
        acsg = tf.concat([[[0,0,0]],acsg], axis = 0)
        acs_grad =  acsg - acsi
        acs_grad_average  = tf.reshape(tf.nn.avg_pool1d(tf.reshape(acs_grad,(1,-1,3)),16*30,1,'SAME'),(-1,3))
        #acsi += acs_grad_average
        speed_i = tf.cumsum(acsi, axis = 0)

        acs_grad  = speeds30hz - speed_i
        acs_grad_average  = tf.reshape(tf.nn.avg_pool1d(tf.reshape(acs_grad,(1,-1,3)),16*30,1,'SAME'),(-1,3))
        acs_grad -= acs_grad_average
        acs_loss = tf.reduce_sum(tf.square(acs_grad*30), axis = -1)# +tf.reduce_mean(tf.square(speed_loc))/300
 
        right_dir = self.get_right(self.angles)
        speed_grad = tf.reduce_sum(speeds30hz*right_dir, axis = -1)
        speed_loss = tf.square(speed_grad)#/(1+my_norm(speeds30hz)*30)
        speed_grad = tf.reshape(speed_grad,(-1,1))*right_dir


        pred_gyr = self.angles[1:]-self.angles[:-1]
        gyrl_tr, gyrl_pr = self.get_local_gyr(pred_gyr)
        gyrl_dif = gyrl_pr-gyrl_tr
        gyrl_dif_average  = tf.reshape(tf.nn.avg_pool1d(tf.reshape(gyrl_dif,(1,-1,3)),16*30,1,'SAME'),(-1,3)) # five minutes
        gyrl_tr += gyrl_dif_average
        gyrl_dif -= gyrl_dif_average
        or_tr = tf.cumsum(gyrl_tr, axis = 0)
        or_pr = tf.cumsum(gyrl_pr, axis = 0)
        gyrl_dif = or_tr-or_pr
        gyrl_dif_average  = tf.reshape(tf.nn.avg_pool1d(tf.reshape(gyrl_dif,(1,-1,3)),16*30,1,'SAME'),(-1,3)) # five minutes
        gyrl_dif -= gyrl_dif_average
        quat_loss = tf.reduce_sum(tf.square(gyrl_dif*300), axis = -1)\
                     #+ tf.reduce_mean(tf.square(gyrl_dif[1:]-gyrl_dif[:-1]))

        #quat_loss = quat_loss*1e-5 + tf.concat([[0],my_norm(angles_cum[1:] - angles_cum[:-1])], axis = 0)


        # gyr_stable = my_norm(gyrl_tr)
        # acs_stable = my_norm(acsi)
        # spd_stable = my_norm(speed[1:]+speed[:-1])
        # stable_poses = tf.logical_and(tf.logical_and(gyr_stable<1e-3,acs_stable<1e-2),spd_stable<1e-1)
        # stable_poses = tf.cast(stable_poses,tf.float32)
        
        return tf.reduce_mean(acs_loss), acs_grad, tf.reduce_mean(quat_loss), tf.reduce_mean(speed_loss), speed_grad, self.g, None

    def get_angles(self, epoch_times):
        pred_gyr = self.angles[1:]-self.angles[:-1]
        gyrl_tr, gyrl_pr = self.get_local_gyr(pred_gyr)
        gyrl_dif = gyrl_pr-gyrl_tr
        gyrl_dif_average  = tf.reshape(tf.nn.avg_pool1d(tf.reshape(gyrl_dif,(1,-1,3)),5*60*30,1,'SAME'),(-1,3)) # five minutes
        gyrl_dif -= gyrl_dif_average
        return (gyrl_tr*self.fps).numpy(), (gyrl_pr*self.fps).numpy(), (gyrl_dif*self.fps).numpy()

    def get_euler(self, epoch_times):
        speed_indexes_float = epoch_times*self.fps/1000
        speed_indexes_int = tf.cast(speed_indexes_float,tf.int32)
        speed_indexes_rest = (speed_indexes_float - tf.cast(speed_indexes_int,tf.float32))[:,tf.newaxis]


        orient_0_pr = tf.gather(self.angles, speed_indexes_int, axis=0)
        orient_1_pr = tf.gather(self.angles, speed_indexes_int+1, axis=0)
        # linear aproximation between two values shoud helps to find time shift
        orient_pr = orient_0_pr * (1-speed_indexes_rest) + orient_1_pr * speed_indexes_rest

        return orient_pr.numpy()

    def get_yaw(self, epoch_times):
        return self.get_euler(epoch_times)[:,2]

    def get_acses(self, speed, gt_speed, epoch_times):
        speed    = speed   *1000/(epoch_times[1:] - epoch_times[:-1])[:,np.newaxis]
        gt_speed = gt_speed*1000/(epoch_times[1:] - epoch_times[:-1])[:,np.newaxis]

        acsi = self.get_global_acs() 
        acsi = acsi  + self.down*self.g/self.fps
        speed_loc = tf.cumsum(acsi, axis = 0)
        epoch_times = (epoch_times[1:] + epoch_times[:-1])/2
        speed_indexes_float = epoch_times*self.fps/1000
        speed_indexes_int = tf.cast(speed_indexes_float,tf.int32)
        speed_indexes_rest = (speed_indexes_float - tf.cast(speed_indexes_int,tf.float32))[:,tf.newaxis]
        speeds_0 = tf.gather(speed_loc, speed_indexes_int, axis=0)
        speeds_1 = tf.gather(speed_loc, speed_indexes_int+1, axis=0)
        # linear aproximation between two values shoud helps to find time shift
        speed_loc = speeds_0 * (1-speed_indexes_rest) + speeds_1 * speed_indexes_rest
        acsi = speed_loc[1:] - speed_loc[:-1]
        acsr = speed[1:] - speed[:-1]
        acst = gt_speed[1:] - gt_speed[:-1]

        return acsi.numpy(), acst.numpy(), acsr.numpy()

    def compute_output_shape(self, _):
        return (1)
    def get_config(self):
        config = super().get_config().copy()
        return config        

from scipy import ndimage, misc
def running_median_insort(seqin, window_size):
    """Contributed by Peter Otten"""
    return ndimage.median_filter(seqin,size=(window_size,1))

def createRigidModel(epochtimes, acsvalues,acstimes,gyrvalues,gyrtimes, baselines, gt_np):

    fps = 30
    baseline_epochs = epochtimes.copy()
    timedif = epochtimes[-1] - epochtimes[0] + 2000
    nummesures = int(timedif*fps/1000)

    starttime = epochtimes[0] - 1000
    epochtimes = np.arange(nummesures)/fps*1000+starttime

    gyrvalues -= np.median(gyrvalues, axis = 0)
    def interp_nd(epochtimes, valtimes, valcum):
        dim = []
        for i in range(len(valcum[0])):
            dim.append(np.interp(epochtimes, valtimes, valcum[:,i]))
        return np.array(dim).T

    gyrvalues = np.cumsum((gyrvalues[1:]+gyrvalues[:-1])*(gyrtimes[1:] - gyrtimes[:-1])[:,np.newaxis]/2000, axis = 0)
    gyr_epochs = interp_nd(epochtimes, (gyrtimes[1:] + gyrtimes[:-1])/2, gyrvalues)
    gyr_epochs = (gyr_epochs[1:] - gyr_epochs[:-1])*fps

    acs_epochs = interp_nd(epochtimes, acstimes, acsvalues)
    mat = np.array([[1.,0.,0],[0.,0.,-1.],[0.,1.,0]]) # just switch to more comfortable coord system
    gyr_epochs = mat.dot(gyr_epochs.T).T
    acs_epochs = mat.dot(acs_epochs.T).T

    orientation = predict_angle(acs_epochs, np.concatenate([[gyr_epochs[0]],gyr_epochs], axis = 0),epochtimes,baseline_epochs,baselines, gt_np, fps)
    orientation[:60] = orientation[61]
    return RigidModelImu(acs_epochs, gyr_epochs, orientation, fps)
    
