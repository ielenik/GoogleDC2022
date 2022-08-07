from scipy import signal
from ast import Assign
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import datetime
from src.utils.kml_writer import KMLWriter
from pathlib import Path

from src.laika.lib.coordinates import ecef2geodetic, geodetic2ecef
from src.laika.astro_dog import AstroDog
from src.laika.gps_time import GPSTime
from src.laika.downloader import download_cors_station
from .tf.tf_numpy_tools import inv, transform, get_quat, mult

from .utils.loader import myLoadRinexIndexed, myLoadRinex
from .utils.gnss_log_reader import gnss_log_to_dataframes
from .utils.gnss_log_processor import process_gnss_log
from .utils.coords_tools import calc_pos_fix, calc_speed
from math import pi as PI

GOOGLE_DATA_ROOT = r'D:\databases\smartphone-decimeter-2022'

NaN = float("NaN")


def pos_from_shift(sat_positions, deltas, deltas_weights):
    # solve for pos
    def Fx_pos(x_hat, poses = sat_positions, delt = deltas, weights=deltas_weights):
        sat_distances = np.linalg.norm(sat_positions-x_hat[:3],axis=-1)
        sat_shifts = sat_distances[1:]-sat_distances[:-1] - deltas
        sat_shifts -= np.nanmedian(sat_shifts,axis=-1,keepdims=True)
        #sat_shifts = sigmoid(sat_shifts*10)
        rows = np.abs(np.nansum(sat_shifts,axis=0))
        rows = rows[rows > 0]
        return np.sum(rows)
    return Fx_pos

def calc_pos(sat_positions, deltas, deltas_weights, x0):
    Fx_pos = pos_from_shift(sat_positions, deltas, deltas_weights)
    x0 = np.array(x0)
    import scipy.optimize as opt
    
    #opt_pos = opt.least_squares(Fx_pos, x0).x
    opt_pos = opt.minimize(Fx_pos, x0, method='nelder-mead',
              options={'xatol': 1e-8, 'disp': True}).x
    return opt_pos, 0

    minerr = 100000
    for i in range(-20,20,1):
        for j in range(-20,20,1):
            for k in range(-100,100,1):
                opt_pos = np.array([i,j,k])
                errs = Fx_pos(opt_pos)
                if errs < minerr:
                    minerr = errs
                    res = opt_pos
                    print(errs,i,j,k)
    return res, errs

def shift_from_deltas(shifts, weights, dirs):
    # solve for pos
    nlen = len(shifts)*8//10
    if nlen < 8:
        nlen = 8
    def Fx_pos(x_hat, shifts = shifts, weights = weights, dirs=dirs):
        sat_shifts = (np.sum(dirs*x_hat[:3],axis=-1) - shifts + x_hat[3])*weights
        return np.mean(np.sort(np.abs(sat_shifts))[:nlen]) + np.abs(x_hat[2])/(np.linalg.norm(x_hat[:3])+1)/5
    return Fx_pos

import scipy.optimize as opt
def calc_shift(shifts, weights, dirs, x0):
    shifts = shifts[weights > 0]
    dirs = dirs[weights > 0]
    weights = weights[weights > 0]

    res = np.array([NaN, NaN, NaN, NaN])
    n = len(shifts)
    if n < 6:
        return res, NaN

    Fx_pos = shift_from_deltas(shifts, weights, dirs)
    x0 = np.array(x0).copy()
    
    #opt_pos = opt.least_squares(Fx_pos, x0).x
    opt_pos = opt.minimize(Fx_pos, x0, method='nelder-mead',
              options={'xatol': 1e-8, 'disp': False}).x
    err = Fx_pos(opt_pos)
    sat_shifts = (np.sum(dirs*opt_pos[:3],axis=-1) - shifts + opt_pos[3])*weights
    if np.sum(np.abs(sat_shifts) < 0.5) < 0.5:
        return res, NaN
    return opt_pos, err

def get_speed_fromrange(shifts, weights, dirs):
    shifts = shifts[weights > 0]
    dirs = dirs[weights > 0]
    weights = weights[weights > 0]

    res = np.array([NaN, NaN, NaN, NaN])
    n = len(shifts)
    if n < 6:
        return res, NaN

    ones = np.ones((n,1))
    dirs = np.concatenate((dirs,ones), axis = -1)

    best_err = 1e7
    for _ in range(100):
        indexes = np.random.choice(n, 4, replace=False)
        mat = dirs[indexes]
        vec = shifts[indexes]
        try:
            mat = np.linalg.inv(mat)
        except:
            continue
        
        x_hat = np.dot(mat, vec)
        x_hat = np.reshape(x_hat,(1,4))

        cur_shifts = np.sum(dirs*x_hat, axis=1)
        rows = np.abs(cur_shifts - shifts)*weights
        if np.sum(rows) < best_err and np.sum(rows < 1) >= 6:
            best_err = np.sum(rows)
            res = x_hat[0]

    if best_err == 1e7: 
        return res, NaN

    best_err01 = 1e7
    for _ in range(50):
        indexes = np.random.choice(n, 4, replace=False)
        mat = dirs[indexes]
        vec = shifts[indexes]
        try:
            mat = np.linalg.inv(mat)
        except:
            continue
        
        x_hat = np.dot(mat, vec)
        if np.linalg.norm(x_hat[:2] - res[:2]) < 0.5:
            continue
        
        x_hat = np.reshape(x_hat,(1,4))

        cur_shifts = np.sum(dirs*x_hat, axis=1)
        rows = np.abs(cur_shifts - shifts)*weights
        if np.sum(rows) < best_err01:
            best_err01 = np.sum(rows)
    return res, best_err/best_err01    

def speed_from_dopler(speeds, weights, dirs, types):
    # solve for pos
    nlen = len(speeds)*8//10
    if nlen < 8:
        nlen = 8
    def Fx_pos(x_hat, speeds = speeds, weights = weights, dirs=dirs):
        sat_shifts = np.sum(dirs*x_hat[:3],axis=-1) - speeds 
        m = np.zeros((8,))
        for i in range(8):
            m[i] = np.median(sat_shifts[types == i])

        sat_shifts = (sat_shifts-m[types])*weights
        return np.mean(np.sort(np.abs(sat_shifts))[:nlen]) + np.abs(x_hat[2])/(np.linalg.norm(x_hat[:3])+1)
    return Fx_pos

import scipy.optimize as opt
def calc_speed_from_dopler2(speeds, weights, dirs, types, x0):
    speeds = speeds[weights > 0]
    dirs = dirs[weights > 0]
    types  = types[weights > 0]
    weights = weights[weights > 0]

    res = np.array([NaN, NaN, NaN])
    n = len(speeds)
    if n < 6:
        return res, NaN

    Fx_pos = speed_from_dopler(speeds, weights, dirs, types)
    x0 = np.array(x0).copy()
    
    #opt_pos = opt.least_squares(Fx_pos, x0).x
    opt_pos = opt.minimize(Fx_pos, x0, method='nelder-mead',
              options={'xatol': 1e-8, 'disp': False}).x
    err = Fx_pos(opt_pos)
    return opt_pos, err

def calc_speed_from_dopler(speeds, weights, dirs, types, x0):
    speeds = speeds[weights > 0]
    dirs = dirs[weights > 0]
    types  = types[weights > 0]
    weights = weights[weights > 0]

    res = np.array([NaN, NaN, NaN])
    n = len(speeds)
    if n < 6:
        return res, NaN

    scale = 1000
    lasterr = 1e7
    for j in range(1000):
        sat_shifts = np.sum(dirs*x0,axis=-1) - speeds 
        m = np.zeros((8,))
        for i in range(8):
            m[i] = np.median(sat_shifts[types == i])

        sat_shifts = (sat_shifts-m[types])*weights
        err = np.sum(np.abs(sat_shifts))
        if lasterr + 1 < err:
            scale *= 2

        delta = np.sum(sat_shifts.reshape((-1,1))*dirs, axis = 0)
        x0 -= delta / scale
        

        lasterr = err
        if np.linalg.norm(delta) < scale/10000:
            break
    if False:#abs(x0[2]) > 1:
        x0[2] = 0
        scale = 1000
        lasterr = 1e7
        for j in range(1000):
            sat_shifts = np.sum(dirs*x0,axis=-1) - speeds 
            m = np.zeros((8,))
            for i in range(8):
                m[i] = np.median(sat_shifts[types == i])

            sat_shifts = (sat_shifts-m[types])*weights
            err = np.sum(np.abs(sat_shifts))
            if lasterr + 1 < err:
                scale *= 2

            delta = np.sum(sat_shifts.reshape((-1,1))*dirs, axis = 0)
            x0[:2] -= delta[:2] / scale
            

            lasterr = err
            if np.linalg.norm(delta[:2]) < scale/10000:
                break

    return x0, 0

def calc_speed_from_dopler_md(speeds, weights, dirs, types, x0):
    num_epoch = len(speeds)
    scale = np.ones((num_epoch,1))*1e-3
    lasterr = np.ones(num_epoch)*1e7
    speeds[weights == 0] = NaN
    m = np.zeros((num_epoch, 8))
    for i in range(8):
        m[:,i] = np.nanmedian(speeds[:,types == i], axis =-1)
    speeds = speeds-m[:,types]
    weights[np.abs(speeds) > 50] = 0
    speeds[weights == 0] = NaN

    for j in tqdm(range(1000)):
        x10 = np.reshape(x0, (-1,1,3))
        sat_shifts = np.sum(dirs*x10,axis=-1) - speeds 
        for i in range(8):
            m[:,i] = np.nanmedian(sat_shifts[:,types == i], axis =-1)

        sat_shifts = (sat_shifts-m[:,types])*weights
        err = np.nansum(np.abs(sat_shifts), axis = -1)
        scale[lasterr + 1 < err] /= 2

        delta = np.nansum(sat_shifts.reshape((num_epoch, -1, 1))*dirs, axis = 1)
        x0 -= delta * scale
        lasterr = err
    err /= len(sat_shifts[0])
    print("Error:", np.mean(err))
    return err

from scipy import ndimage, misc

def dif(sig):
    return sig[1:] - sig[:-1]

def filter_outliers(baselines, angles, times):
    baselines[:,2] = 0
    dirs = np.stack([-np.sin(angles), np.cos(angles), np.zeros_like(angles)],axis = 1)
    dirs_ort = np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)],axis = 1)

    times_delta = (times[1:]-times[:-1]).reshape((-1,1))
    for j in range(5):
        speed = (baselines[1:] - baselines[:-1])*1000/times_delta
        errors_dir1 = np.sum(speed*dirs_ort, axis = -1, keepdims=True)/(np.linalg.norm(speed, axis = -1, keepdims=True)/30 + 1)
        errors_dir2 = np.sum(speed*dirs, axis = -1, keepdims=True)
        errors_dir2[errors_dir2>0] = 0
        acs = speed[1:] - speed[:-1]
        weights = np.abs(errors_dir1)+np.abs(errors_dir2)
        weights = weights[1:] + weights[:-1]
        weights += np.linalg.norm(acs, axis = -1, keepdims=True)/100
        edge = 25
        while np.sum(weights>edge) < 3:
            edge -= 0.1
        weights[weights<edge] = 0
        weights[weights>0] = 1
        print("Filtering", np.sum(weights), "outliers")
        weights = np.concatenate([[[0]],weights,[[0]]], axis = 0)
        weights = 1-weights
        nans, x = weights[:,0] == 0, lambda z: z.nonzero()[0]
        for i in range(3):
            baselines[nans, i]= np.interp(times[x(nans)], times[x(~nans)], baselines[~nans, i])

    res = baselines.copy()


    for i in tqdm(range(10000)):
        speed = (res[1:] - res[:-1])*1000/times_delta
        errors_dir = np.sum(speed*dirs_ort, axis = -1, keepdims=True)
        res[1:] -= errors_dir*dirs_ort/4 #- speed/100
        res[:-1] += errors_dir*dirs_ort/4 #- speed/100
        errors_dir = np.sum(speed*dirs, axis = -1, keepdims=True)
        errors_dir[errors_dir>0] = 0
        res[1:] -= errors_dir*dirs/4
        res[:-1] += errors_dir*dirs/4
        acs = speed[1:] - speed[:-1]
        res[1:-1] += acs/16
        
        dists = np.linalg.norm(res-baselines, axis = -1, keepdims=True)
        mt = 0.1 #weights/(dists*10 + 10)
        res = res*(1-mt)+baselines*mt

    return res

def calc_initial_speed_guess(sat_deltashifts, sat_deltarangeweigths, sat_deltaspeed, sat_deltaspeedweights, sat_types, sat_dirs, gt_np, acses, acses_times, gyros, gyros_times, epoch_times, baselines):
    shifts = baselines[1:]-baselines[:-1]
    sh_gt = gt_np[1:]-gt_np[:-1]
    angles = np.arctan2(-shifts[:,0], shifts[:,1])
    angles[np.linalg.norm(shifts,axis=-1)<4] = NaN

    acses_med = np.median(acses, axis = 0)
    acses_med /= np.linalg.norm(acses_med)
    gyros = np.sum(gyros*acses_med, axis = -1)
    gyros -= np.median(gyros, axis = 0)
    gyros = gyros[1:]
    gyros = gyros*(gyros_times[1:]-gyros_times[:-1])/1000
    gyros_times = gyros_times[1:]

    # acsvalues = acsvalues[1:]
    # acsvalues = acsvalues*(acstimes[1:]-acstimes[:-1]).reshape((-1,1))/1000
    # acstimes = acstimes[1:]

    gyroscum = np.cumsum(gyros, axis=0)

    angles_gt = np.arctan2(-sh_gt[:,0], sh_gt[:,1])
    angles_gt[np.linalg.norm(sh_gt, axis = -1) < 1] = NaN

    def resolve_shift(ang):
        nans, x= np.isnan(ang), lambda z: z.nonzero()[0]
        st = ang[0]
        ang[nans]= 0#np.interp(x(nans), x(~nans), ang[~nans])
        shifts = ang[1:] - ang[:-1]
        shifts[shifts>PI] -= 2*PI
        shifts[shifts<-PI] += 2*PI
        if ~np.isnan(st):
            shifts = np.concatenate([[st],shifts], axis = 0)
        else:
            shifts = np.concatenate([[0],shifts], axis = 0)
        ang = np.cumsum(shifts)
        ang[nans] = NaN
        return ang



    angles = resolve_shift(angles)
    angles_gt = resolve_shift(angles_gt)


    def get_gyro_err_fun(gyro, ang):
        idx = np.arange(len(gyro))/2000
        def gyro_err(x):
            error = np.abs(gyro - ang + x[0] + x[1]*idx)#+ x[2]*idx**2+ x[3]*idx**3)#+ x[4]*idx**4+ x[5]*idx**5+ x[6]*idx**6+ x[7]*idx**7)
            error -= (error/(2*PI)).astype(np.int32)*2*PI
            error -= (error/PI).astype(np.int32)*2*PI
            error = np.nansum(np.abs(error))
            return error**0.1
        return gyro_err
    def fit_gyro(gyro, ang):
        minlen = min(len(gyro),len(ang))
        gyro = gyro[:minlen]
        ang  = ang[:minlen]
        x0 = np.zeros((2))
        x0 = opt.least_squares(get_gyro_err_fun(gyro,ang), x0).x
        return opt.minimize(get_gyro_err_fun(gyro,ang), x0, method='nelder-mead',
                 options={'xatol': 1e-8, 'disp': True}).x


    gyro = np.interp((epoch_times[1:]+epoch_times[:-1])/2, gyros_times, gyroscum)
    # gyro_short = gyroscum[::factor]
    # gyro = gyro_short[:,1]
    # gyro = gyro[:len(angles)]


    # plt.clf()
    # plt.plot( np.arange(len(gyro)-1), dif(gyro))
    # plt.plot( np.arange(len(angles_gt)-1), dif(angles_gt))
    # plt.plot( np.arange(len(angles)-1), dif(angles))
    # # plt.plot( np.arange(len(sh_gt)), sh_gt2[:,0] + 3)
    # # plt.plot( np.arange(len(sh_gt)), sh_gt2[:,1] + 4)
    # # plt.plot( np.arange(len(sh_gt)), sh_gt3[:,0] + 5)
    # # plt.plot( np.arange(len(sh_gt)), sh_gt3[:,1] + 6)
    # plt.legend(['gyroscop', 'real', 'shift pred'])
    # plt.show()

    mean = np.nanmedian(gyro-angles_gt)
    shift = (mean/(PI*2)).astype(np.int32)
    gyro -= shift*2*PI
    mean -= shift*2*PI
    shift = (mean/PI).astype(np.int32)
    gyro -= shift*2*PI
    mean -= shift*2*PI

    shift = ((gyro - angles)/(PI*2)).astype(np.int32)
    angles += shift*2*PI
    shift = ((gyro - angles)/PI).astype(np.int32)
    angles += shift*2*PI
    shift = np.nanmedian(np.abs(((gyro - angles)/PI)))
    for i in range(3):
        
        gfit = fit_gyro(gyro, angles)
        idx = np.arange(len(gyro))/2000
        gyro = gyro+gfit[0]+gfit[1]*idx#+gfit[2]*idx**2+gfit[3]*idx**3#+gfit[4]*idx**4+gfit[5]*idx**5+gfit[6]*idx**6+gfit[7]*idx**7
        mean = np.nanmedian(gyro-angles_gt)
        shift = (mean/(PI*2)).astype(np.int32)
        gyro -= shift*2*PI
        mean -= shift*2*PI
        shift = (mean/PI).astype(np.int32)
        gyro -= shift*2*PI
        mean -= shift*2*PI
        
        

        shift = ((gyro - angles)/(PI*2)).astype(np.int32)
        angles += shift*2*PI
        shift = ((gyro - angles)/PI).astype(np.int32)
        angles += shift*2*PI
        shift = ((gyro - angles_gt)/(PI*2)).astype(np.int32)
        angles_gt += shift*2*PI
        shift = ((gyro - angles_gt)/PI).astype(np.int32)
        angles_gt += shift*2*PI
        
        shift = np.nanmedian(np.abs(((gyro - angles)/PI)))
        if abs(shift) > 0.6:
            gyro += PI

            mean = np.nanmedian(gyro-angles_gt)
            shift = (mean/(PI*2)).astype(np.int32)
            gyro -= shift*2*PI
            mean -= shift*2*PI
            shift = (mean/PI).astype(np.int32)
            gyro -= shift*2*PI
            mean -= shift*2*PI

            shift = ((gyro - angles)/(PI*2)).astype(np.int32)
            angles += shift*2*PI
            shift = ((gyro - angles)/PI).astype(np.int32)
            angles += shift*2*PI
            shift = np.nanmedian(np.abs(((gyro - angles)/PI)))
        shift = np.nanmedian(np.abs(((gyro - angles)/PI)))

        # nans, x= np.isnan(angles), lambda z: z.nonzero()[0]
        # angles[nans]= np.interp(x(nans), x(~nans), angles[~nans])
        # corr = signal.correlate(angles[1:]-angles[:-1], gyro[1:] - gyro[:-1], mode='same')
        # start = np.argmax(corr)-len(gyro)//2
        # gyro_short = gyroscum[start*factor::factor]
        # gyro = gyro_short[:,1]
        # gyro = gyro[:len(angles)]
        # angles[nans] = NaN
    # gyro = gyro+gfit[0]+gfit[1]*idx#+gfit[2]*idx**2+gfit[3]*idx**3+gfit[4]*idx**4
    # plt.clf()
    # plt.plot( np.arange(len(angles)), angles)
    # plt.plot( np.arange(len(gyro)), gyro)
    # plt.plot( np.arange(len(angles_gt)), angles_gt)
    # plt.legend(['shift pred', 'gyroscop', 'real'])
    # plt.show()

    shift = gyro-angles
    shift[np.isnan(shift)] = np.random.normal(0,0.1)
    shift = ndimage.median_filter(shift,size=(50))

    # plt.clf()
    # plt.plot( np.arange(len(angles)), angles)
    # plt.plot( np.arange(len(gyro)), gyro - shift)
    # plt.plot( np.arange(len(angles_gt)), angles_gt)
    # plt.legend(['shift pred', 'gyroscop', 'real'])
    # plt.show()


    #gyro = gyro-shift
    baselines2 = filter_outliers(baselines,gyro,epoch_times)
    shifts2 = baselines2[1:]-baselines2[:-1]
    angles2 = np.arctan2(-shifts2[:,0], shifts2[:,1])
    angles2[np.linalg.norm(shifts2,axis=-1)<4] = NaN

    # plt.clf()
    # plt.plot( np.arange(len(angles2)), angles2)
    # plt.plot( np.arange(len(gyro)), gyro - shift)
    # plt.plot( np.arange(len(angles_gt)), angles_gt)
    # plt.legend(['shift pred', 'gyroscop', 'real'])
    # plt.show()
    # plt.clf()
    # plt.plot( np.arange(len(baselines)), (baselines[:,0,:]-gt_np)[:,:2])
    # plt.plot( np.arange(len(baselines)), (baselines2    -gt_np)[:,:2])
    # plt.legend(['bdx', 'bdy', 'pdx', 'pdy'])
    # plt.show()

    shift = ((gyro - angles2)/(PI*2)).astype(np.int32)
    angles2 += shift*2*PI
    shift = ((gyro - angles2)/PI).astype(np.int32)
    angles2 += shift*2*PI

    # plt.clf()
    # plt.plot( np.arange(len(angles)), angles)
    # plt.plot( np.arange(len(gyro)), gyro)
    # plt.plot( np.arange(len(angles2)), angles2)
    # plt.plot( np.arange(len(angles_gt)), angles_gt)
    # plt.legend(['shift pred', 'gyroscop','corrected', 'real'])
    # plt.show()
    return baselines2, gyro

    shifts = []
    shifts2 = []
    sures  = []
    sh_gt = gt_np[1:] - gt_np[:-1]
    sat_deltashifts[sat_deltarangeweigths == 0] = NaN
    sat_deltashifts -= np.nanmedian(sat_deltashifts, axis = -1, keepdims = True)
    sat_deltarangeweigths[sat_deltashifts > 30] = 0
    sat_deltashifts[sat_deltarangeweigths == 0] = 0

    s = NaN
    sat_deltaspeedweights[sat_deltaspeedweights < 0.5] = 0
    sat_deltaspeed[sat_deltaspeedweights == 0] = 0

    # shifts2 = np.zeros((len(sat_deltaspeed),3))
    # errs = calc_speed_from_dopler_md(sat_deltaspeed, sat_deltaspeedweights,sat_dirs, sat_types, shifts2)
    # shifts2[np.abs(shifts2[:,2]) > 1] = NaN
    # shifts2 = (shifts2[1:] + shifts2[:-1])/2
    # errs = errs[1:]+errs[:-1]
    # plt.clf()
    # plt.plot( np.arange(len(shifts2)), np.linalg.norm(shifts2-sh_gt, axis = -1))
    # plt.plot( np.arange(len(errs)), errs)
    # plt.legend(['errs true', 'errs pred'])
    # plt.show()

    '''
    shifts = np.zeros((len(sat_deltashifts),3))
    sat_types2 = np.zeros_like(sat_types)
    errs = calc_speed_from_dopler_md(-sat_deltashifts, sat_deltarangeweigths,sat_dirs[1:], sat_types2, shifts)
    shifts[np.abs(shifts[:,2]) > 1] = NaN
    shifts[np.sum(sat_deltarangeweigths > 0 , axis = -1) < 7] = NaN
    shifts[errs > 0.5] = NaN
    errs[np.isnan(shifts[:,0])] = NaN
    plt.clf()
    plt.plot( np.arange(len(shifts)), np.linalg.norm(shifts-sh_gt, axis = -1))
    plt.plot( np.arange(len(errs)), errs)
    plt.legend(['errs true', 'errs pred'])
    plt.show()
    '''


    #calc_initial_speed_guess_old(sat_deltashifts, sat_deltarangeweigths, sat_deltaspeed, sat_deltaspeedweights, sat_types, sat_dirs, gt_np, acses, acses_times, gyros, gyros_times, epoch_times)
    shifts = []
    shifts2 = []
    sures  = []
    sh_gt = gt_np[1:] - gt_np[:-1]
    sat_deltashifts[sat_deltarangeweigths == 0] = NaN
    sat_deltashifts -= np.nanmedian(sat_deltashifts, axis = -1, keepdims = True)
    sat_deltarangeweigths[sat_deltashifts > 30] = 0
    sat_deltashifts[sat_deltarangeweigths == 0] = 0

    s = NaN
    sat_deltaspeedweights[sat_deltaspeedweights < 0.5] = 0
    sat_deltaspeed[sat_deltaspeedweights == 0] = 0

    shifts2 = np.zeros((len(sat_deltaspeed),3))
    calc_speed_from_dopler_md(sat_deltaspeed, sat_deltaspeedweights,sat_dirs, sat_types, shifts2)
    shifts2[np.abs(shifts2[:,2]) > 1] = NaN

    if False:
        angles2 = np.arctan2(shifts2[:,1], shifts2[:,0])
        angles2[np.linalg.norm(shifts2, axis = -1) < 1] = NaN
        angles2_dif = dif(angles2)
        angles2_dif -= (angles2_dif/PI).astype(np.int32)*2*PI
        angles2_dif[np.isnan(angles2_dif)] = 0
        timegrid = np.arange(len(epoch_times)*100)*10+epoch_times[0]
        angles2_dif_inter = np.interp(timegrid, (epoch_times[1:]+epoch_times[:-1])/2, angles2_dif)
        gyro_inter = np.interp(timegrid, gyros_times, gyros[:,1])
        corr = signal.correlate(angles2_dif_inter, gyro_inter, mode='same')
        timeshift = (np.argmax(corr) - len(angles2_dif_inter)//2)*10
        print('timeshift',timeshift)
        
        delay_arr = np.linspace(-0.5*len(angles2_dif_inter)/10, 0.5*len(angles2_dif_inter)/10, len(angles2_dif_inter))
        delay = delay_arr[np.argmax(corr)]
        print('y2 is ' + str(delay) + ' behind y1')
        
        gyros_times += timeshift
        gyro_inter2 = np.interp(timegrid, gyros_times, gyros[:,1])
        plt.clf()
        plt.plot( np.arange(len(corr)), corr)
        plt.show()

        plt.clf()
        plt.plot( np.arange(len(gyro_inter)), gyro_inter)
        plt.plot( np.arange(len(gyro_inter)), gyro_inter2)
        plt.plot( np.arange(len(angles2_dif_inter)), angles2_dif_inter)
        plt.show()
        shifts2_copy = shifts2.copy()
    shifts2 = (shifts2[1:] + shifts2[:-1])/2

    shifts = np.zeros((len(sat_deltashifts),3))
    sat_types2 = np.zeros_like(sat_types)
    calc_speed_from_dopler_md(sat_deltashifts, sat_deltarangeweigths,sat_dirs[1:], sat_types, shifts)

    shifts = -np.array(shifts)
    sures = np.array(sures)
    shifts[sures>0.01,:] = NaN
    shifts = shifts[:,:3]
    shifts[np.abs(shifts[:,2])>1,:] = NaN
    shifts[np.isnan(shifts)] = shifts2[np.isnan(shifts)]
    shifts[np.abs(shifts[:,2])>1,:] = NaN
    shifts_large = np.ones_like(shifts)*NaN
    shifts_len = np.linalg.norm(shifts, axis = -1)
    shifts_large[shifts_len > 3] = shifts[shifts_len > 3]
    angles = np.arctan2(-shifts_large[:,0], shifts_large[:,1])


    
    gyros = gyros[1:]
    gyros = gyros*(gyros_times[1:]-gyros_times[:-1]).reshape((-1,1))/1000
    gyros_times = gyros_times[1:]

    # acsvalues = acsvalues[1:]
    # acsvalues = acsvalues*(acstimes[1:]-acstimes[:-1]).reshape((-1,1))/1000
    # acstimes = acstimes[1:]

    gyroscum = np.cumsum(gyros, axis=0)


    def resolve_shift(ang):
        nans, x= np.isnan(ang), lambda z: z.nonzero()[0]

        ang[nans]= 0#np.interp(x(nans), x(~nans), ang[~nans])
        shifts = ang[1:] - ang[:-1]
        shifts[shifts>PI] -= 2*PI
        shifts[shifts<-PI] += 2*PI
        ang = np.cumsum(shifts)
        ang[nans[1:]] = NaN
        return ang


    angles_gt = np.arctan2(-sh_gt[:,0], sh_gt[:,1])
    angles_gt[np.linalg.norm(sh_gt, axis = -1) < 1] = NaN

    angles = resolve_shift(angles)
    angles_gt = resolve_shift(angles_gt)
    gyro = np.interp(epoch_times[1:-1], gyros_times, gyroscum[:,1])

    # plt.clf()
    # plt.plot( np.arange(len(angles_gt)), angles_gt)
    # plt.plot( np.arange(len(angles)), angles)
    # plt.plot( np.arange(len(gyro)), gyro)
    # plt.show()

    def get_gyro_err_fun(gyro, ang):
        idx = np.arange(len(gyro))/2000
        def gyro_err(x):
            error = np.abs(gyro - ang + x[0] + x[1]*idx)
            error -= (error/(2*PI)).astype(np.int32)*2*PI
            error -= (error/PI).astype(np.int32)*2*PI
            error = np.abs(error)

            return np.nansum(error) + np.abs(x[1])*100# + x[2]*idx**2+ x[3]*idx**3+ x[4]*idx**4))
        return gyro_err
    def fit_gyro(gyro, ang):
        minlen = min(len(gyro),len(ang))
        gyro = gyro[:minlen]
        ang  = ang[:minlen]
        x0 = np.zeros((2))
        return opt.minimize(get_gyro_err_fun(gyro,ang), x0, method='nelder-mead',
                options={'xatol': 1e-8, 'disp': True}).x


    # gyro_short = gyroscum[::factor]
    # gyro = gyro_short[:,1]
    # gyro = gyro[:len(angles)]


    # plt.clf()
    # plt.plot( np.arange(len(gyro)-1), dif(gyro))
    # plt.plot( np.arange(len(angles_gt)-1), dif(angles_gt))
    # plt.plot( np.arange(len(angles)-1), dif(angles))
    # # plt.plot( np.arange(len(sh_gt)), sh_gt2[:,0] + 3)
    # # plt.plot( np.arange(len(sh_gt)), sh_gt2[:,1] + 4)
    # # plt.plot( np.arange(len(sh_gt)), sh_gt3[:,0] + 5)
    # # plt.plot( np.arange(len(sh_gt)), sh_gt3[:,1] + 6)
    # plt.legend(['gyroscop', 'real', 'shift pred'])
    # plt.show()

    gfit_cum = np.zeros((2,))
    for i in range(3):
        
        gfit = fit_gyro(gyro, angles)
        gfit_cum += gfit
        idx = np.arange(len(gyro))/2000
        gyro = gyro+gfit[0]+gfit[1]*idx#+gfit[2]*idx**2+gfit[3]*idx**3+gfit[4]*idx**4
        mean = np.nanmedian(gyro-angles_gt)
        shift = (mean/(PI*2)).astype(np.int32)
        gyro -= shift*2*PI
        mean -= shift*2*PI
        shift = (mean/PI).astype(np.int32)
        gyro -= shift*2*PI
        mean -= shift*2*PI
        
        
        shift = ((gyro - angles)/(PI*2)).astype(np.int32)
        angles += shift*2*PI
        shift = ((gyro - angles)/PI).astype(np.int32)
        angles += shift*2*PI

        shift = ((gyro - angles_gt)/(PI*2)).astype(np.int32)
        angles_gt += shift*2*PI
        shift = ((gyro - angles_gt)/PI).astype(np.int32)
        angles_gt += shift*2*PI
        # nans, x= np.isnan(angles), lambda z: z.nonzero()[0]
        # angles[nans]= np.interp(x(nans), x(~nans), angles[~nans])
        # corr = signal.correlate(angles[1:]-angles[:-1], gyro[1:] - gyro[:-1], mode='same')
        # start = np.argmax(corr)-len(gyro)//2
        # gyro_short = gyroscum[start*factor::factor]
        # gyro = gyro_short[:,1]
        # gyro = gyro[:len(angles)]
        # angles[nans] = NaN
    
    # gyro = gyro+gfit[0]+gfit[1]*idx#+gfit[2]*idx**2+gfit[3]*idx**3+gfit[4]*idx**4

    # plt.clf()
    # plt.plot( np.arange(len(gyro)), gyro)
    # plt.plot( np.arange(len(angles_gt)), angles_gt)
    # plt.plot( np.arange(len(angles)), angles)
    # # plt.plot( np.arange(len(sh_gt)), sh_gt2[:,0] + 3)
    # # plt.plot( np.arange(len(sh_gt)), sh_gt2[:,1] + 4)
    # # plt.plot( np.arange(len(sh_gt)), sh_gt3[:,0] + 5)
    # # plt.plot( np.arange(len(sh_gt)), sh_gt3[:,1] + 6)
    # plt.legend(['gyroscop', 'real', 'shift pred'])
    # plt.show()


    return shifts, gfit_cum
