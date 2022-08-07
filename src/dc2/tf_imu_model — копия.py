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
    def __init__(self, acs, gyr, dn, **kwargs):
        super(RigidModelImu, self).__init__(**kwargs)
        self.acs  = tf.Variable(acs, name = 'acs', trainable=False, dtype = tf.float32)
        self.gyr  = tf.Variable(gyr, name = 'gyr', trainable=False, dtype = tf.float32)

        self.down  = tf.Variable([[0.,0.,-1.]], trainable=False, dtype = tf.float32, name = 'down')
        self.fwd  = tf.Variable([[0.,1.,0.]], trainable=False, dtype = tf.float32, name = 'fwd')
        self.idquat = tf.Variable([[0.,0.,0.,1.]], trainable=False, dtype = tf.float32, name = 'idquat')
        self.north = self.fwd

        self.g      = tf.Variable([[9.81]], name = 'g', trainable=False, dtype = tf.float32)
        self.g_scale      = tf.Variable([[1]], name = 'g_scale', trainable=True, dtype = tf.float32)
        #self.ort    = tf.Variable(np.ones((len(acs),4))*np.array([get_quat([0,1,0], [0,0,1])]), name = 'orientaiton', trainable=True, dtype = tf.float32)
        self.ort    = tf.Variable(np.array([get_quat(dn, [0,0,1])]), name = 'orientaiton', trainable=True, dtype = tf.float32)

    def build(self, input_shape):
        super(RigidModelImu, self).build(input_shape)


    def get_angles(self, inputs):
        sp, quats = inputs

        quats, _ = tf.linalg.normalize(quats, axis = -1)
        orientation, _ = tf.linalg.normalize(self.ort, axis = -1)

        qt1 = quats[1:]
        qt2 = quats[:-1]


        iqt1 = inv(qt1)
        true_quat = mult(iqt1, qt2)
        local_gyr = mult(mult(inv(orientation), self.gyr), orientation)
        
        true_quat = tf.linalg.l2_normalize(true_quat, axis = -1)
        local_gyr = tf.linalg.l2_normalize(local_gyr, axis = -1)
        true_quat = true_quat*tf.reshape(tf.math.sign(true_quat[:,3]),(-1,1))
        local_gyr = local_gyr*tf.reshape(tf.math.sign(local_gyr[:,3]),(-1,1))

        return true_quat, local_gyr
        return true_quat, inv(local_gyr)


    def get_acses(self, inputs):
        speed, speedgt, quats = inputs

        quats, _ = tf.linalg.normalize(quats, axis = -1)
        orientation, _ = tf.linalg.normalize(self.ort, axis = -1)

        acsr = speed[1:] - speed[:-1]
        acst = speedgt[1:] - speedgt[:-1]

        qt1 = quats[1:]
        qt2 = quats[:-1]

        acsi = transform(self.acs,orientation)
        qtm = tf.linalg.l2_normalize((qt1 + qt2)/2, axis = -1)
        acsi = transform(acsi, qtm)
        acsi = acsi + self.down*self.g


        acsi = transform(acsi, inv(qtm)) 
        acsr = transform(acsr, inv(qtm)) 
        acst = transform(acst, inv(qtm)) 

        return acsi, acst, acsr

    def call(self, inputs):
        speed, quats = inputs
        quats, _ = tf.linalg.normalize(quats, axis = -1)
        orientation, _ = tf.linalg.normalize(self.ort, axis = -1)

        acsr = speed[1:] - speed[:-1]

        qt1 = quats[1:]
        qt2 = quats[:-1]

        acsi = transform(self.acs,orientation)
        qtm = tf.linalg.l2_normalize((qt1 + qt2)/2, axis = -1)
        acsi = transform(acsi, qtm)
        acsi = acsi*self.g_scale  + self.down*self.g
        #acs_loss = tf.reduce_sum(tf.square(acsi - acsr)*0.01, axis = -1)
        acs_loss = tf.reduce_sum(tf.abs(acsi - acsr)*0.01, axis = -1) + tf.abs(1.-tf.reduce_mean(self.g_scale))/10.
        #acs_loss = my_norm(acsi - acsr)*0.1

        speed_loss = ((my_norm(speed) - tf.reduce_sum(speed*transform(self.fwd,quats), axis = -1)))/10
        speed_loss = (speed_loss[1:]+speed_loss[:-1])/10

        iqt1 = inv(qt1)
        true_quat = mult(iqt1, qt2)
        local_gyr = mult(mult(inv(orientation), self.gyr), orientation)
        
        true_quat = tf.linalg.l2_normalize(true_quat, axis = -1)
        local_gyr = tf.linalg.l2_normalize(local_gyr, axis = -1)
        true_quat = true_quat*tf.reshape(tf.math.sign(true_quat[:,3]),(-1,1))
        local_gyr = local_gyr*tf.reshape(tf.math.sign(local_gyr[:,3]),(-1,1))

        quat_loss = tf.reduce_sum(tf.abs(true_quat-(local_gyr)), axis = -1)*100
        #quat_loss = tf.reduce_sum(tf.abs(true_quat-inv(local_gyr)), axis = -1)*100


        #quat_loss = (1 - tf.abs(tf.reduce_sum(true_quat*local_gyr, axis = -1)))*1000
        #angle_loss = (tf.square(quats[:,0]) + tf.square(quats[:,1]))/100.
        #quat_loss += angle_loss[1:]+angle_loss[:-1]
        #quat_loss = tf.nn.relu(quat_loss - 1e-5)
        quat_loss += (tf.abs(quats[1:,1])+tf.abs(quats[1:,0]))/10

        # or_loss = my_norm(orientation[1:] - orientation[:-1])
        # or_loss = tf.concat([or_loss,[0]], axis = 0)
        # quat_loss += or_loss

        stable_poses = tf.reduce_sum(tf.square(self.gyr[:,:3]),axis=-1)
        speed_norm = my_norm(speed)[:-1]
        stable_poses = tf.cast(stable_poses<1e-6,tf.float32)*tf.cast(speed_norm<0.1,tf.float32)
        # stable_poses = stable_poses*(speed_norm + my_norm(qt1-qt2))
        # quat_loss += stable_poses*1000.
        
        return acs_loss, quat_loss, speed_loss, self.g_scale, stable_poses

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
    running_median_insort(acsvalues,100)
    # print("Filtering gyro...")
    # running_median_insort(gyrvalues,30)
    covmat = np.cov(acsvalues.T)
    sc, vecs = np.linalg.eig(covmat)
    ind = np.argmax(sc)
    pr = np.sum(np.array([vecs[ind]])*acsvalues)
    if pr < 0:
        vecs[ind] = -vecs[ind]


    accum_values(acstimes - 600 - 500,acsvalues,acs_epochs)
    accum_values(gyrtimes - 600 - 500,gyrvalues,gyr_epochs)

    for i in range(nummesures):
        quat_epochs[i] = to_quat(gyr_epochs[i], 1)
    quat_epochs = quat_epochs[1:-1]
    acs_epochs = acs_epochs[1:-1]

    return RigidModelImu(acs_epochs, quat_epochs, vecs[ind])
    
