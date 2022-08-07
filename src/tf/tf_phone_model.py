from re import T
import pickle

from numpy.core.fromnumeric import reshape
from numpy.lib.function_base import append
from tensorflow.python.ops.variables import Variable
from src.laika.lib.coordinates import ecef2geodetic, geodetic2ecef
from src.laika import AstroDog
from src.laika.gps_time import GPSTime

import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.ops.array_ops import zeros
from tensorflow.python.training.tracking import base

import pymap3d as pm
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from skimage.io import imread, imsave
import time
import pandas as pd
import itertools
from matplotlib import pyplot
import datetime


class WeightsData(tf.keras.layers.Layer):
    def __init__(self, value, regularizer = None, trainable = True, **kwargs):
        super(WeightsData, self).__init__(**kwargs)
        self.value = value
        self.tr = trainable
        self.regularizer = regularizer

    def build(self, input_shape):
        super(WeightsData, self).build(input_shape)
        self.W = self.add_weight(name='W', shape=self.value.shape, 
                                dtype = tf.float32,
                                initializer=tf.keras.initializers.Constant(self.value),
                                regularizer=self.regularizer,
                                trainable=self.tr)

    def call(self, inputs):
        return tf.gather_nd(self.W, inputs)
    def compute_output_shape(self, input_shape):
        return [(input_shape[0], 3)]
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'initializer': self.initializer,
            'inshape': self.inshape,
        })
        return config

class PsevdoDistancesLayer2(tf.keras.layers.Layer):
    def __init__(self, num_sats, num_epochs, sat_types, sat_poses, psevdo_dist, psevdoweights, **kwargs):
        super(PsevdoDistancesLayer2, self).__init__(**kwargs)
        self.num_sats = num_sats
        self.num_epochs = num_epochs
        self.sat_poses = tf.Variable(sat_poses, trainable=False, dtype = tf.float32)
        self.psevdo_dist = tf.Variable(psevdo_dist, trainable=False, dtype = tf.float32)
        self.psevdoweights = tf.Variable(psevdoweights, trainable=False, dtype = tf.float32)
        self.sat_types = tf.Variable(sat_types, trainable=False, dtype = tf.float32)

    def build(self, input_shape):
        super(PsevdoDistancesLayer2, self).build(input_shape)
        self.isrbm_bias = self.add_weight(name='psevdo_isrbm_bias', shape=(8, self.num_epochs), 
                                dtype = tf.float32,
                                initializer=tf.keras.initializers.Constant(np.zeros((8, self.num_epochs))),
                                trainable=True)

    def call(self, inputs):
        distance = tf.linalg.norm(inputs - self.sat_poses, axis = -1)
        isrbm = tf.gather(self.isrbm_bias, self.sat_types)
        isrbm = tf.transpose(isrbm)
        errors = tf.abs(self.psevdoweights*(distance -  self.psevdo_dist - isrbm))
        errors = errors - tf.nn.relu(errors - 1)*0.7

        #return tf.reduce_mean((self.psevdoweights*tf.nn.softsign(tf.abs(errors)/5))), errors
        return tf.reduce_mean(errors), errors

    def compute_output_shape(self, _):
        return (1)

class PsevdoDistancesLayer(tf.keras.layers.Layer):
    def __init__(self, _num_sats_psevdo, _num_epochs_psevdo, _base_poses, _sat_types, _sat_dirs, _psevdo_shift, _psevdoweights, **kwargs):
        super(PsevdoDistancesLayer, self).__init__(**kwargs)
        self.num_sats_psevdo = _num_sats_psevdo
        self.num_epochs_psevdo = _num_epochs_psevdo
        self.base_poses_ = _base_poses
        self.sat_types_ = _sat_types
        self.sat_dirs_ = _sat_dirs
        self.psevdo_shift_ = _psevdo_shift
        self.psevdoweights_ = _psevdoweights
        self.base_poses = tf.Variable(_base_poses, trainable=False, dtype = tf.float32, name = 'base_poses_var')
        self.sat_types = tf.Variable(_sat_types, trainable=False, name = 'sat_types_var')
        self.sat_dirs = tf.Variable(_sat_dirs, trainable=False, dtype = tf.float32, name = 'sat_dirs_var')
        self.psevdo_shift = tf.Variable(_psevdo_shift, trainable=False, dtype = tf.float32, name = 'psevdo_shift_var')
        self.psevdoweights = tf.Variable(_psevdoweights, trainable=False, dtype = tf.float32, name = 'psevdoweights_var')

    def build(self, input_shape):
        super(PsevdoDistancesLayer, self).build(input_shape)
        self.isrbm_bias = self.add_weight(name='psevdo_isrbm_bias', shape=(8, self.num_epochs_psevdo), 
                                dtype = tf.float32,
                                initializer=tf.keras.initializers.Constant(np.zeros((8, self.num_epochs_psevdo))),
                                trainable=True)

    def call(self, inputs):
        shiftsnow = tf.reduce_sum( (inputs - self.base_poses)*self.sat_dirs, axis = -1)
        isrbm = tf.gather(self.isrbm_bias, self.sat_types)
        isrbm = tf.transpose(isrbm)
        errors = tf.abs(self.psevdoweights*(shiftsnow -  self.psevdo_shift - isrbm))
        errors = errors - tf.nn.relu(errors - 3)*0.7

        #return tf.reduce_mean((self.psevdoweights*tf.nn.softsign(tf.abs(errors)/5))), errors
        return tf.reduce_mean(errors), errors

    def compute_output_shape(self, _):
        return (1)
    def get_config(self):
        '''
        def __init__(self, num_sats, num_epochs, base_poses, sat_types, sat_dirs, psevdo_shift, psevdoweights, **kwargs):
        super(PsevdoDistancesLayer, self).__init__(**kwargs)
        self.num_sats = num_sats
        self.num_epochs = num_epochs
        self.base_poses_ = base_poses
        self.sat_types_ = sat_types
        self.sat_dirs_ = sat_dirs
        self.psevdo_shift_ = psevdo_shift
        self.psevdoweights_ = psevdoweights
        '''
        config = super().get_config().copy()
        config.update({
            '_num_sats_psevdo': self.num_sats_psevdo,
            '_num_epochs_psevdo': self.num_epochs_psevdo,
            '_base_poses': self.base_poses_,
            '_sat_types': self.sat_types_,
            '_sat_dirs': self.sat_dirs_,
            '_psevdo_shift': self.psevdo_shift_,
            '_psevdoweights': self.psevdoweights_,
        })
        return config        

class DeltaRangeLayer(tf.keras.layers.Layer):
    def __init__(self, _num_sats, _num_epochs, _sat_directions, _sat_deltarange, _sat_deltavalid, **kwargs):
        super(DeltaRangeLayer, self).__init__(**kwargs)
        self.num_sats = _num_sats
        self.num_epochs = _num_epochs
        self.sat_directions_ = _sat_directions
        self.sat_deltarange_ = _sat_deltarange
        self.sat_deltavalid_ = _sat_deltavalid
        self.sat_directions = tf.Variable(_sat_directions,trainable=False, dtype = tf.float32, name = 'sat_directions_var')
        self.sat_deltarange = tf.Variable(_sat_deltarange,trainable=False, dtype = tf.float32, name = 'sat_deltarange_var')
        self.sat_deltavalid = tf.Variable(_sat_deltavalid,trainable=False, dtype = tf.float32, name = 'sat_deltavalid_var')

    def build(self, input_shape):
        super(DeltaRangeLayer, self).build(input_shape)
        self.delta_epoch_bias = self.add_weight(name='delta_epoch_bias', shape=(self.num_epochs-1,1), 
                                dtype = tf.float32,
                                initializer=tf.keras.initializers.Constant(np.zeros((self.num_epochs-1,1))),
                                trainable=True)

    def call(self, inputs):
        shift = inputs[1:] - inputs[:-1]
        scalar = tf.reduce_sum(shift*self.sat_directions, axis = -1)
        errors = (scalar +  self.sat_deltarange - self.delta_epoch_bias)*self.sat_deltavalid
        errors = tf.abs(errors) - tf.nn.relu(tf.abs(errors) - 0.2)*0.7
        #return tf.reduce_mean(tf.nn.softsign(tf.abs(errors))), errors
        return tf.reduce_mean(tf.abs(errors)), errors

    def compute_output_shape(self, _):
        return (1)

    def get_config(self):
        '''
        def __init__(self, num_sats, num_epochs, sat_directions, sat_deltarange, sat_deltavalid, **kwargs):
        super(DeltaRangeLayer, self).__init__(**kwargs)
        self.num_sats = num_sats
        self.num_epochs = num_epochs
        self.sat_directions_ = sat_directions
        self.sat_deltarange_ = sat_deltarange
        self.sat_deltavalid_ = sat_deltavalid
        '''
        config = super().get_config().copy()
        config.update({
            '_num_sats': self.num_sats,
            '_num_epochs': self.num_epochs,
            '_sat_directions': self.sat_directions_,
            '_sat_deltarange': self.sat_deltarange_,
            '_sat_deltavalid': self.sat_deltavalid_,
        })
        return config        

def check_deltas_valid(sat_dir_in, lens_in, valid):

    n = np.sum(valid > 0)
    if n < 5:
        return False, [], np.zeros((len(valid)))
    
    sat_dir = sat_dir_in[valid > 0].copy()
    lens = lens_in[valid > 0]

    sat_dir = np.append(sat_dir, np.ones((n,1)), axis = 1)
    min_err = 1e10
    res_shift = [[0.,0.,0.,0.]]
    for _ in range(50):
        indexes = np.random.choice(n, 4, replace=False)
        mat = sat_dir[indexes]
        vec = lens[indexes]
        try:
            mat = np.linalg.inv(mat)
        except:
            continue

        x_hat = np.dot(mat, vec)
        x_hat = np.reshape(x_hat,(1,4))
        cur_shifts = np.sum(sat_dir*x_hat, axis=1)
        rows = np.abs(cur_shifts - lens)
        curerr = np.sum(np.minimum(rows, 0.1)) + np.abs(x_hat[0,2])/10
        if curerr < min_err:
            min_err = curerr
            res_shift = x_hat
    
    sat_dir = sat_dir_in.copy()
    lens = lens_in
    sat_dir = np.append(sat_dir, np.ones((len(sat_dir),1)), axis = 1)
    cur_shifts = np.sum(sat_dir*res_shift, axis=1)
    rows = np.abs(cur_shifts - lens)
    return np.sum(valid) >= 6, res_shift[0], rows

def createGpsPhoneModel(m):
    sat_psevdovalid = m['sat_psevdovalid']
    sat_psevdodist = m['sat_psevdodist']
    baselines = m['baseline']
    sat_positions = m['sat_positions']
    sat_psevdoweights = m['sat_psevdoweights']
    sat_deltarange = m['sat_deltarange']
    sat_deltavalid = m['sat_deltavalid']
    sat_types = m['sat_types']

    sat_psevdovalid[np.isnan(sat_psevdodist)] = 0
    sat_psevdodist[sat_psevdovalid == 0] = 0

    baselines = np.reshape(baselines,(-1,1,3))
    sat_realdist = np.linalg.norm(sat_positions-baselines, axis = -1)
    dists = sat_psevdovalid*(sat_psevdodist - sat_realdist)
    dists_corr = dists.copy()

    num_epochs = len(sat_positions)
    num_used_satelites = len(sat_positions[0])
    for i in range(num_epochs):
        isbrms = [[],[],[],[],[],[],[],[]]
        for j in range(num_used_satelites):
            if sat_psevdovalid[i,j] == 0:
                continue
            isbrms[sat_types[j]].append(dists_corr[i,j])
        for j in range(8):
            if len(isbrms[j]) == 0:
                isbrms[j] = 0
            else:
                isbrms[j] = np.median(isbrms[j])
        for j in range(num_used_satelites):
            if sat_psevdovalid[i,j] == 0:
                continue
            dists_corr[i,j] -= isbrms[sat_types[j]]
            sat_psevdodist[i,j] -= isbrms[sat_types[j]]


    print(np.sum(abs(dists_corr) > 100))
    sat_psevdovalid[abs(dists_corr) > 100] = 0
    sat_psevdoweights = sat_psevdoweights*sat_psevdovalid
    dists_corr = dists_corr*sat_psevdoweights
    loss = np.mean(np.abs(dists_corr[sat_psevdovalid > 0]))
    print("Initial psevdo range loss ", loss)

    sat_directions = sat_positions - baselines
    sat_directions = sat_directions/np.linalg.norm(sat_directions,axis=-1,keepdims=True)

    sat_psevdoshift = sat_psevdodist - sat_realdist
    psevdo_layer = PsevdoDistancesLayer(num_used_satelites,num_epochs,baselines,sat_types,sat_directions,-sat_psevdoshift,sat_psevdoweights)

    sat_directions = sat_directions[1:]
    sat_deltarange = sat_deltarange[1:] - sat_deltarange[:-1]
    sat_deltavalid = sat_deltavalid[1:]*sat_deltavalid[:-1]

    baselines = baselines[:-1]
    sat_distanse_first = np.linalg.norm(baselines - sat_positions[:-1],axis=-1)
    sat_distanse_next  = np.linalg.norm(baselines - sat_positions[1:],axis=-1)
    sat_distanse_dif = sat_distanse_next - sat_distanse_first
    sat_deltarange -= sat_distanse_dif
    sat_deltarange_sorted = np.sort(sat_deltarange[sat_deltavalid > 0])
    median_deltarange =  np.median(sat_deltarange_sorted)

    detected_shifts= []
    accum_rows = []

    sat_deltavalid[np.isnan(sat_deltarange)] = 0
    for i in tqdm(range(num_epochs-1)):
        sat_deltavalid_cur = sat_deltavalid[i].copy()
        sat_deltavalid_cur[ sat_deltarange[i] < -1000 ] = 0
        sat_deltavalid_cur[ sat_deltarange[i] > 30000 ] = 0
        ret, xhat, rows = check_deltas_valid(sat_directions[i],sat_deltarange[i], sat_deltavalid_cur)
        rows = rows[sat_deltavalid[i] > 0]
        rows = rows[np.abs(rows) > 1000]
        accum_rows.extend(rows)

    accum_rows = sorted(accum_rows)
    accum_counts = []
    j = 0
    for i in range(len(accum_rows)):
        while j < len(accum_rows) and accum_rows[j] - accum_rows[i] < 0.1:
            j += 1
        if j - i > 100:
            accum_counts.append((j - i,accum_rows[i]))

    accum_counts = sorted(accum_counts)[::-1]
    accum_counts_res = []
    for r in accum_counts:
        found = False
        for t in accum_counts_res:
            if abs(r[1]-t[1]) < 100:
                found = True
                break
        if found == False:
            accum_counts_res.append(r)

    for i in tqdm(range(num_epochs-1)):
        for j in range(len(sat_deltarange[i])):
            for t in accum_counts_res:
                if abs(sat_deltarange[i,j]-t[1]) < 100:
                    sat_deltarange[i,j] -= t[1]
                if abs(sat_deltarange[i,j]+t[1]) < 100:
                    sat_deltarange[i,j] += t[1]

        sat_deltavalid_cur = sat_deltavalid[i].copy()
        sat_deltavalid_cur[ sat_deltarange[i] < -3000 ] = 0
        sat_deltavalid_cur[ sat_deltarange[i] > 30000 ] = 0
        ret, xhat, rows = check_deltas_valid(sat_directions[i],sat_deltarange[i], sat_deltavalid_cur)
        if  ret == False:
            sat_deltavalid[i] = 0
            sat_deltarange[i] = 0
            detected_shifts.append([])
            continue

        ind = (sat_deltavalid[i]>0)
        sat_deltarange[i,ind] -= xhat[3]
        sat_deltavalid[i,np.abs(np.array(rows)) > 0.5] = 0
        sat_deltarange[i,sat_deltavalid[i] == 0] = 0

        if np.sum(ind) > 6:
            detected_shifts.append(xhat[:3])
        else:
            detected_shifts.append([])

    sat_deltavalid[np.abs(sat_deltarange) >  30] = 0 #
    sat_deltarange[np.abs(sat_deltarange) >  30] = 0 #
    print(np.sum(sat_deltavalid > 0))
    print(np.sum(sat_psevdovalid > 0))

    delta_layer = DeltaRangeLayer(num_used_satelites,num_epochs,sat_directions, sat_deltarange, sat_deltavalid)

    model_input = tf.keras.layers.Input((num_epochs,3), dtype=tf.float32)
    positions = tf.reshape(model_input,(num_epochs,1,3))
    psevdo_loss, psevdo_errors = psevdo_layer(positions)
    delta_loss, delta_errors = delta_layer(positions)
    
    gps_phone_model = tf.keras.Model(model_input, [psevdo_loss, delta_loss])

    return gps_phone_model, m['epoch_times']

