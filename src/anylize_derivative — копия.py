import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
import os 
from tqdm import tqdm
import matplotlib.pyplot as plt
import datetime
from src.utils.kml_writer import KMLWriter

from src.laika.lib.coordinates import ecef2geodetic, geodetic2ecef
from src.laika.astro_dog import AstroDog
from src.laika.gps_time import GPSTime
from src.laika.downloader import download_cors_station
from .tf.tf_numpy_tools import inv, transform, get_quat, mult

from .utils.loader import myLoadRinexIndexed, myLoadRinex
from .utils.gnss_log_reader import gnss_log_to_dataframes
from .utils.gnss_log_processor import process_gnss_log
from .utils.coords_tools import calc_pos_fix, calc_speed

GOOGLE_DATA_ROOT = r'D:\databases\smartphone-decimeter-2022'

NaN = float("NaN")

def preload_gnss_log(path_folder):
    try:
        with open(path_folder+'/data.cache.pkl', 'rb') as f:
            cur_measure =  pickle.load(f)
    except:
        constellations = ['GPS', 'GLONASS', 'BEIDOU','GALILEO']
        dog = AstroDog(valid_const=constellations, pull_orbit=True)

        df_raw = pd.read_csv(f'{path_folder}/device_gnss.csv')
        datetimenow = datetime.datetime.utcfromtimestamp(df_raw['utcTimeMillis'].iat[0]/1000.0)
        gpsdatetimenow = GPSTime.from_datetime(datetimenow)

        cache_path = '\\'.join(path_folder.split('\\')[:-1])
        path_basestation = download_cors_station(gpsdatetimenow,'slac', cache_path)
        basestation, coords, satregistry = myLoadRinexIndexed(path_basestation)
        
        datetimenow1 = datetime.datetime.utcfromtimestamp(df_raw['utcTimeMillis'].iat[-1]/1000.0)
        if datetimenow1.day !=  datetimenow.day:
            gpsdatetimenow1 = GPSTime.from_datetime(datetimenow1)
            path_basestation = download_cors_station(gpsdatetimenow1,'slac', cache_path)
            basestation1, coords1, satregistry = myLoadRinexIndexed(path_basestation, satregistry)
            basestation.extend(basestation1)


        basestation = { 
            'times': np.array([r[0] for r in basestation]), 
            'values':np.array([r[1] for r in basestation]),
            'deltas':np.array([r[2] for r in basestation]),
            'coords':coords,
            'sat_registry':satregistry,
            }

        cur_measure = process_gnss_log(df_raw, basestation, dog)
        num_epoch = len(cur_measure['epoch_times'])
        x0=[0, 0, 0, 0,0,0,0,0,0]
        baseline_poses = []
        for i in tqdm(range(num_epoch)):
            x0, err = calc_pos_fix(cur_measure['sat_positions'][i], cur_measure['sat_psevdodist'][i], cur_measure['sat_psevdoweights'][i],cur_measure['sat_psevdovalid'][i] > 0, cur_measure['sat_types'], x0)
            if np.sum(err < 1000) > 10:
                break #initial guess
        for i in tqdm(range(num_epoch)):
            if i == 1275:
                _ = 12321
            x0, _ = calc_pos_fix(cur_measure['sat_positions'][i], cur_measure['sat_psevdodist'][i], cur_measure['sat_psevdoweights'][i],cur_measure['sat_psevdovalid'][i] > 0, cur_measure['sat_types'], x0)
            baseline_poses.append(x0[:3])

        cur_measure['baseline'] = np.array(baseline_poses)
        cur_measure['basestation'] = basestation

    with open(path_folder+'/data.cache.pkl', 'wb') as f:
        pickle.dump(cur_measure, f, pickle.HIGHEST_PROTOCOL)

    return cur_measure

from .dc2.tf_dopler_model import createDoplerModel
from .dc2.tf_phase_model import createPhaseModel
from .dc2.tf_imu_model import createRigidModel
from .dc2.tf_psevdo_model import createPsevdoModel

def save_fig(path):
    figure = plt.gcf() # get current figure
    figure.set_size_inches(24, 18)
    # when saving, specify the DPI
    plt.savefig(path, dpi = 100)    

def sigmoid(z):
    return 1/(1 + np.exp(-z))

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
    for _ in range(50):
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

    return x0, 0

from scipy import ndimage, misc

def calc_initial_speed_guess(sat_deltashifts, sat_deltarangeweigths, sat_deltaspeed, sat_deltaspeedweights, sat_types, sat_dirs, gt_np, acses, acses_times, gyros, gyros_times, epoch_times):
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

    # for i in tqdm(range(len(sat_deltaspeed))):
    #     if np.isnan(s):
    #         sh = np.zeros((3,))
    #     sh, s = calc_speed_from_dopler(sat_deltaspeed[i], sat_deltaspeedweights[i],sat_dirs[i], sat_types, sh)
    #     shifts2.append(sh.copy())
    # shifts2 = np.array(shifts2)
    # shifts2 = (shifts2[1:] + shifts2[:-1])/2

    for i in tqdm(range(len(sat_deltashifts))):
        sh, s = get_speed_fromrange(sat_deltashifts[i], sat_deltarangeweigths[i],sat_dirs[i+1])
        shifts.append(sh)
        sures.append(s)
        if np.linalg.norm(sh_gt[i,:2] - sh[:2]) > 1:
            sh, s = get_speed_fromrange(sat_deltashifts[i], sat_deltarangeweigths[i],sat_dirs[i+1])


    gt_np = gt_np[0:len(shifts)+1]

    shifts = -np.array(shifts)
    sures = np.array(sures)
    shifts = shifts[:,:3]
    # shifts[np.isnan(shifts)] = shifts2[np.isnan(shifts)]
    #sures[np.isnan(sures)] =  0.9
    # shifts[sures>0.01,:] = NaN

    sh_gt = gt_np[1:] - gt_np[:-1] - shifts[:,:3]
    # sh_gt2 = gt_np[1:] - gt_np[:-1] - shifts2[:,:3]
    # shifts3 = shifts.copy()
    # shifts3[np.linalg.norm(shifts[:,:2]-shifts2[:,:2]) > 0.1,:] = NaN
    # sh_gt3 = gt_np[1:] - gt_np[:-1] - shifts3[:,:3]

    # plt.clf()
    # plt.plot( np.arange(len(sh_gt)), sh_gt[:,0])
    # plt.plot( np.arange(len(sh_gt)), sh_gt[:,1] + 1)
    # plt.plot( np.arange(len(sures)), sures + 2)
    # # plt.plot( np.arange(len(sh_gt)), sh_gt2[:,0] + 3)
    # # plt.plot( np.arange(len(sh_gt)), sh_gt2[:,1] + 4)
    # # plt.plot( np.arange(len(sh_gt)), sh_gt3[:,0] + 5)
    # # plt.plot( np.arange(len(sh_gt)), sh_gt3[:,1] + 6)
    # plt.legend(['px', 'py', 's','px1', 'py1', 's1'])
    # plt.show()

    shifts = shifts[:,:3]

    angls = np.arccos(np.sum(shifts[1:]*shifts[:-1], axis = -1)/(np.linalg.norm(shifts[1:], axis = -1)*np.linalg.norm(shifts[:-1], axis = -1)+1e-7))
    angls[np.linalg.norm(shifts[1:], axis = -1)*np.linalg.norm(shifts[:-1], axis = -1) < 1] = 0#NaN
    gyroscum = np.cumsum(gyros, axis = 0)
    epoch_time_index = np.zeros((len(epoch_times)),dtype = np.int32)
    curind = 0

    for i in range(len(epoch_time_index)):
        while curind < len(gyros_times) - 1 and gyros_times[curind] + 500  < epoch_times[i] - 1000:
            curind += 1
        epoch_time_index[i] = curind
    bestsh = 0
    besterr = 1e8
    bestval = []
    errors_gyr = []
    for sh in range(-1000,1000, 10):
        for i in range(len(epoch_time_index)):
            while epoch_time_index[i] < len(gyros_times) - 1 and gyros_times[epoch_time_index[i]] + 500 < epoch_times[i]  + sh:
                epoch_time_index[i] += 1
        gyrcur = []
        for i in range(1,len(epoch_time_index)-1):
            gyrcur.append(np.sum(np.abs
                (gyroscum[epoch_time_index[i]] - gyroscum[epoch_time_index[i-1]]), axis = -1)
                /(epoch_time_index[i] -epoch_time_index[i-1]))
        gyrcur = np.array(gyrcur)
        curerr = np.nanmean(np.abs(angls - gyrcur))
        errors_gyr.append(curerr)
        if curerr < besterr:
            bestval = gyrcur.copy()
            besterr = curerr
            bestsh = sh
    print(bestsh, besterr)
    plt.clf()
    plt.plot( np.arange(len(bestval)), bestval - 0.02)
    plt.plot( np.arange(len(angls)), angls)
    plt.legend(['p angls', 't angls'])
    plt.show()



    acs = np.linalg.norm(shifts[1:]-shifts[:-1], axis = -1)
    acses = acses - np.median(acses,axis=0,keepdims=True)
    #acses2 = ndimage.median_filter(acses,size=(15000,1))
    acses2 = ndimage.uniform_filter(acses,size=(15000,1))
    acses -= acses2
    plt.clf()
    #plt.plot( np.arange(len(acses)), acses + np.array([[0.,1.,2.]]))
    plt.plot( np.arange(len(acses)), acses2 + np.array([[3.,4.,5.]]))
    plt.legend(['p acs', 't acs'])
    plt.show()

    #acses = ndimage.median_filter(acses,size=(50,1))
    acsescum = np.cumsum(acses, axis = 0)
    epoch_time_index = np.zeros((len(epoch_times)),dtype = np.int32)
    curind = 0
    for i in range(len(epoch_time_index)):
        while curind < len(acses_times) - 1 and acses_times[curind] + 500  < epoch_times[i] - 1000:
            curind += 1
        epoch_time_index[i] = curind
    bestsh = 0
    besterr = 1e8
    bestval = []
    errors_acs = []
    acs_mean = np.nanmean(np.linalg.norm(acses,axis=-1))
    for sh in range(-1000,1000, 10):
        for i in range(len(epoch_time_index)):
            while epoch_time_index[i] < len(acses_times) - 1 and acses_times[epoch_time_index[i]] + 500 < epoch_times[i]  + sh:
                epoch_time_index[i] += 1
        acscur = []
        for i in range(1,len(epoch_time_index)-1):
            acscur.append(np.linalg.norm(
                (acsescum[epoch_time_index[i]] - acsescum[epoch_time_index[i-1]]), axis = -1)/(epoch_time_index[i] -epoch_time_index[i-1]))
        acscur = np.array(acscur)
        curerr = np.nanmean(np.abs(acs - acscur))
        errors_acs.append(curerr)
        if curerr < besterr:
            bestval = acscur.copy()
            besterr = curerr
            bestsh = sh
    print(bestsh, besterr)
    plt.clf()
    plt.plot( np.arange(len(bestval)), bestval)
    plt.plot( np.arange(len(acs)), acs)
    plt.legend(['p acs', 't acs'])
    plt.show()

    plt.clf()
    plt.plot( np.arange(len(errors_acs)), errors_acs)
    plt.plot( np.arange(len(errors_gyr)), np.array(errors_gyr)*20)
    plt.legend(['acs', 'gyr'])
    plt.show()
def calc_track_speed(trip_id):
    gt_raw = pd.read_csv(f'{GOOGLE_DATA_ROOT}{trip_id}/ground_truth.csv')
    gt_np = gt_raw[['LatitudeDegrees','LongitudeDegrees','AltitudeMeters']].to_numpy()
    gt_np[np.isnan(gt_np)] = 0
    gt_np = geodetic2ecef(gt_np)

    imu_raw = pd.read_csv(f'{GOOGLE_DATA_ROOT}{trip_id}/device_imu.csv')
    image_path = f'{GOOGLE_DATA_ROOT}{trip_id}/plots/'
    os.makedirs(image_path, exist_ok=True)

    cur_measure = preload_gnss_log(f'{GOOGLE_DATA_ROOT}{trip_id}')
    st_seg = 0
    en_seg = -1
    # st_seg = 1180
    # en_seg = 1220

    baselines            = cur_measure["baseline"][st_seg:en_seg]
    sat_positions        = cur_measure["sat_positions"][st_seg:en_seg]
    sat_velosities       = cur_measure["sat_velosities"][st_seg:en_seg]
    sat_deltarange       = cur_measure["sat_deltarange"][st_seg:en_seg]
    sat_deltastate       = cur_measure["sat_deltastate"][st_seg:en_seg]
    sat_deltarangeuncert = cur_measure["sat_deltarangeuncert"][st_seg:en_seg]
    sat_deltaspeed       = cur_measure["sat_deltaspeed"][st_seg:en_seg]
    sat_deltaspeeduncert = cur_measure["sat_deltaspeeduncert"][st_seg:en_seg]
    sat_basedeltarange   = cur_measure["sat_basedeltarange"][st_seg:en_seg]
    utcTimeMillis        = cur_measure["utcTimeMillis"][st_seg:en_seg]
    sat_types            = cur_measure["sat_types"]
    sat_names            = cur_measure["sat_names"]
    sat_frequencis       = cur_measure["sat_frequencis"][st_seg:en_seg]
    sat_psevdovalid      = cur_measure["sat_psevdovalid"][st_seg:en_seg]
    sat_psevdodist       = cur_measure["sat_psevdodist"][st_seg:en_seg]
    sat_psevdoweights    = cur_measure["sat_psevdoweights"][st_seg:en_seg]

    sat_wavelength       = 299_792_458/sat_frequencis
    sat_basedeltarange *= sat_wavelength
    gt_np = gt_np[st_seg:en_seg]


    num_epochs  = len(utcTimeMillis)
    num_used_satelites  = len(sat_types)

    coords_start = baselines[0].copy()
    
    mat_local = np.zeros((3,3))
    mat_local[2] = coords_start/np.linalg.norm(coords_start, axis = -1)
    mat_local[1] = np.array([0,0,1])
    mat_local[1] = mat_local[1] - mat_local[2]*np.sum(mat_local[2]*mat_local[1])
    mat_local[1] = mat_local[1]/np.linalg.norm(mat_local[1], axis = -1)
    mat_local[0] = np.cross(mat_local[1], mat_local[2])
    mat_local = np.transpose(mat_local)

    locl_trans = mat_local
    locl_shift = coords_start
    def local_transform(v):
        return np.matmul(v - locl_shift,locl_trans)
    def local_transform_inv(v):
        return np.matmul(v,locl_trans.T) + locl_shift
    def local_vector_transform(v):
        return np.matmul(v,locl_trans)

    baselines = local_transform(baselines)
    sat_positions = local_transform(sat_positions)
    gt_np = local_transform(gt_np)
    sat_velosities = local_vector_transform(sat_velosities)
    base_coords = local_transform([cur_measure['basestation']['coords']])


    sat_dst = np.linalg.norm(sat_positions - np.reshape(base_coords,(1,1,3)),axis=-1)
    sat_deltastate[np.isnan(sat_basedeltarange)] = 0
    sat_basedeltarange[np.isnan(sat_basedeltarange)] = 0
    sat_deltarange_shift = (sat_basedeltarange[1:] - sat_basedeltarange[:-1]) - (sat_dst[1:] - sat_dst[:-1])
    sat_deltarange[1:] -= np.cumsum(sat_deltarange_shift,axis=0)


    sat_rest_cycles_best = None
    bestcnt = 0
    gt_r = np.reshape(gt_np[0],(-1,1,3))
    

    NaN = float("NaN")
    # sat_deltarange[sat_deltarangeuncert > 0.02] = NaN
    # sat_deltarange[sat_deltarangeuncert <= 0] = NaN
    # sat_deltarange[sat_deltastate % 2 == 0] = NaN
    #sat_deltarange[:,sat_types != 1] = NaN
    sat_deltastate[np.isnan(sat_deltarange)] = 0
    sat_deltarangeuncert[np.isnan(sat_deltarange)] = 100
    sat_deltarange[np.isnan(sat_deltarange)] = 0

    x0 = np.zeros((3,)) 
    #x0[:3] = gt_r[0,0,:]
    # x0,err = calc_pos(sat_positions, sat_deltarange[1:]-sat_deltarange[:-1], 1/(sat_deltarangeuncert[1:]*sat_deltarangeuncert[:-1]), x0)
    # x0,err = calc_pos(sat_positions, sat_deltarange[1:]-sat_deltarange[:-1], 1/(sat_deltarangeuncert[1:]*sat_deltarangeuncert[:-1]), x0)
    # x0,err = calc_pos(sat_positions, sat_deltarange[1:]-sat_deltarange[:-1], 1/(sat_deltarangeuncert[1:]*sat_deltarangeuncert[:-1]), x0)
    # x0,err = calc_pos(sat_positions, sat_deltarange[1:]-sat_deltarange[:-1], 1/(sat_deltarangeuncert[1:]*sat_deltarangeuncert[:-1]), x0)

    deltas = (sat_deltarange[-1] - sat_deltarange[0]) - (np.linalg.norm(sat_positions[-1] - x0, axis = -1) - np.linalg.norm(sat_positions[0] - x0, axis = -1))
    deltas -= np.nanmedian(deltas)
    deltas /= sat_wavelength[0]

    #gt_r[0,0,:] = x0[:3]

    #for i in range(-100,100,10):
    if False:
        #gt_r[0,0,2] = i
        sta_vec = sat_positions-gt_r
        #sta_vec = sat_positions - np.reshape(base_coords,(1,1,3))
        sat_dst = np.linalg.norm(sta_vec,axis=-1)
        sta_dir = sta_vec / np.linalg.norm(sta_vec,axis=-1, keepdims=True)
        sta_col = np.nanmedian(sta_dir,axis=0)

        sat_deltarange = sat_deltarange_copy.copy()
        sat_deltarange = (sat_deltarange[1:] - sat_deltarange[:-1]) - (sat_dst[1:] - sat_dst[:-1])
        # medians = np.zeros((len(sat_deltarange), 8))
        # for i in range(8):
        #     medians[:,i] = np.nanmedian(sat_deltarange[:,sat_types == i],axis=-1)

        # for i in range(len(sat_deltarange[0])):
        #     sat_deltarange[:,i] -= medians[:,sat_types[i]]

        sat_deltarange -= np.nanmedian(sat_deltarange,axis=-1,keepdims=True)
        sat_deltarange[np.abs(sat_deltarange) > 0.1] = NaN
        #sat_deltarange -= np.nanmedian(sat_deltarange,axis=-1,keepdims=True)
        sat_deltarange[np.isnan(sat_deltarange)] = 0
        sat_deltarange = np.cumsum(sat_deltarange, axis=0)

        plt.clf()
        colors = [
            (1,0,0,1),
            (0,1,0,1),
            (0,0,1,1),
            (1,0,1,1),
            (0,1,1,1),
            (1,1,0,1),
            (1,0,0,1),
            (0,1,0,1),
            (0,0,1,1),
            (1,0,1,1),
            (0,1,1,1),
            (1,1,0,1),
        ]


        leg = []
        for i in range(37):
            if np.sum(sat_deltarange[:,i] == 0) == len(sat_deltarange[:,i]):
                continue
            col = [(sta_col[i,0]+1)/2,(sta_col[i,1]+1)/2,(sta_col[i,2]+1)/2,1]
            plt.plot( np.arange(len(sat_deltarange[:,i])), sat_deltarange[:,i],color = col)
            leg.append(sat_names[i])
        plt.legend(leg)
        plt.show()

        return



    baselines = np.reshape(baselines,(-1,1,3))
    sat_directions = sat_positions - baselines  
    sat_distances = np.linalg.norm(sat_directions,axis=-1,keepdims=True)
    sat_directions = sat_directions/sat_distances
    sat_scalar_velosities = np.sum(sat_velosities*sat_directions,axis=-1)
    sat_deltaspeed -= sat_scalar_velosities

    imu_model = createRigidModel(utcTimeMillis,imu_raw)
    dopler_model = createDoplerModel(sat_directions, -sat_deltaspeed, sat_deltaspeeduncert, sat_types)

        # 'sat_psevdovalid':sat_psevdovalid,
        # 'sat_psevdodist':sat_psevdodist,
        # 'sat_psevdoweights':sat_psevdoweights,
    sat_psevdo_shift = sat_psevdodist - sat_distances[:,:,0]
    sat_psevdoweights[sat_psevdovalid == 0] = 0
    sat_psevdo_shift[sat_psevdoweights == 0] = NaN
    medians = np.zeros((8))
    for i in range(8):
        medians[i] = np.nanmedian(sat_psevdo_shift[:,sat_types == i])

    for i in range(len(sat_deltarange[0])):
       sat_psevdo_shift[:,i] -= medians[sat_types[i]]
    sat_psevdo_shift[sat_psevdoweights == 0] = 0

    test1 = np.sum(sat_directions*(baselines-np.reshape(gt_np,(-1,1,3))), axis = -1)
    test2 = (test1 + sat_psevdo_shift)*sat_psevdoweights
    test = np.mean(np.abs(test2),axis = -1)
    times = tf.Variable(np.reshape(np.arange(0,len(sat_directions))*2/len(sat_directions)-1,(-1,1)),trainable=False,dtype=tf.float32)
    psevdo_model = createPsevdoModel(sat_directions, sat_psevdo_shift, sat_psevdoweights, sat_types)




    #sat_directions = (sat_directions[1:]+sat_directions[:-1])/2
    sat_deltarange = sat_deltarange[1:] - sat_deltarange[:-1]
    sat_distanse_dif = np.linalg.norm(sat_positions[1:] - baselines[:-1],axis=-1) - np.linalg.norm(sat_positions[:-1] - baselines[:-1],axis=-1)
    sat_deltarange -= sat_distanse_dif


    sat_deltarangeuncert[sat_deltarangeuncert<1e-8] = 1
    sat_deltastate[sat_deltastate == 17] = 25
    sat_deltaweights = (sat_deltastate == 25)/sat_deltarangeuncert
    sat_deltaweights = np.minimum(sat_deltaweights[1:],sat_deltaweights[:-1])
    sat_deltarange[sat_deltaweights == 0] = NaN
    sat_deltarange -= np.nanmedian(sat_deltarange, axis = -1, keepdims=True)
    sat_deltarange[sat_deltaweights == 0] = 0

    acs = imu_raw[imu_raw['MessageType'] == 'UncalAccel']
    acstimes = acs['utcTimeMillis'].to_numpy()
    acsvalues = acs[['MeasurementX','MeasurementY','MeasurementZ']].to_numpy()
    gir = imu_raw[imu_raw['MessageType'] == 'UncalGyro']
    gyrtimes = gir['utcTimeMillis'].to_numpy()
    gyrvalues = gir[['MeasurementX','MeasurementY','MeasurementZ']].to_numpy()

    #calc_initial_speed_guess(sat_deltarange,sat_deltaweights,-sat_deltaspeed, 1/sat_deltaspeeduncert, sat_types, sat_directions, gt_np,acsvalues,acstimes,gyrvalues,gyrtimes,utcTimeMillis)
    phase_model = createPhaseModel(sat_directions[1:],-sat_deltarange, sat_deltaweights)

    speeds_init_gt = (gt_np[1:] - gt_np[:-1])
    speeds_init = gt_speed = tf.Variable(speeds_init_gt,trainable=False,dtype=tf.float32)

    baselines = np.reshape(baselines,(-1,3))
    speeds_init = (baselines[1:] - baselines[:-1])
    speeds = tf.Variable(speeds_init,trainable=True,dtype=tf.float32)
    
    speeds_np = speeds_init
    j = 0
    for i in range(len(speeds_np)):
        if(np.linalg.norm(speeds_np[i]) > 1):
            while j < i:
                speeds_np[j] = speeds_np[i]
                j += 1
            j += 1
    while j < len(speeds_np):
        speeds_np[j] = speeds_np[j-1]
        j += 1

    or_init = np.zeros((num_epochs-1,3))

    or_init = or_init.astype(np.float32)
    orients = tf.Variable(or_init,trainable=True,dtype=tf.float32)

    true_pos = gt_np
    baselines += np.array([[-1.5,+0.5,0]])
    pred_pos = baselines
    pos_error = np.linalg.norm((true_pos-pred_pos)[:,:2],axis=-1)
    pos_error = np.sort(pos_error)
    pos_error_abs = (pos_error[len(pos_error)//2] + pos_error[len(pos_error)*95//100])/2
    print("Baseline error:", pos_error_abs)

    def minimize_speederror(useimuloss, trainorient, trainspeed, optimizer):
        @tf.custom_gradient
        def norm(x):
            y = tf.linalg.norm(x, 'euclidean', -1)
            def grad(dy):
                return tf.expand_dims(dy,-1) * (x / (tf.expand_dims(y + 1e-19,-1)))

            return y, grad
        for _ in range(4):
            with tf.GradientTape(persistent=True) as tape:
                # speeds_loss_glob = tf.abs(norm(speeds[1:])-norm(speeds[:-1]))
                # speeds_loss_glob += (norm(speeds[1:])*norm(speeds[:-1]) - tf.reduce_sum(speeds[1:]*speeds[:-1], axis = -1))/10
                acs = speeds[1:]-speeds[:-1]
                acs = tf.concat([acs,[[0.,0.,0.,]]], axis = 0)
                speeds_loss_glob = tf.reduce_sum(tf.abs(acs), axis = -1)
                acs = tf.concat([[[0.,0.,0.,]], acs], axis = 0)
                speeds_loss_glob += tf.reduce_sum(tf.abs(acs[1:] - acs[:-1]), axis = -1)

                speeds_loss_Z = tf.square(speeds[:,2])
                phase_loss = phase_model([speeds,orients])
                dopler_loss = dopler_model([speeds,orients])
                acs_loss, quat_loss, speed_loss, g, stable_poses = imu_model([speeds,orients])
                poses = tf.cumsum(speeds, axis = 0)
                poses = tf.concat([[[0,0,0]],poses], axis = 0)

                psevdo_loss = psevdo_model([poses - baselines, times])
                imu_loss = quat_loss + speed_loss + acs_loss
                imu_loss = tf.concat([[0.],imu_loss], axis = 0)
                if not useimuloss:
                    total_loss = imu_loss*1e-3 + psevdo_loss + speeds_loss_glob*10 + speeds_loss_Z*10  + phase_loss + dopler_loss 
                else:   
                    total_loss = imu_loss*10 + phase_loss*10 + psevdo_loss + dopler_loss + speeds_loss_Z
                imu_loss = imu_loss/10
                # total_loss = tf.reduce_mean(total_loss)
                # imu_loss = tf.reduce_mean(imu_loss)
                stable_poses = tf.concat([[0],stable_poses], axis=0)
                stable_loss = tf.reduce_mean(tf.abs(speeds*tf.reshape(stable_poses,(-1,1))))
                total_loss += stable_loss
            
            # stable_poses = tf.concat([[0],stable_poses], axis=0)
            # speeds.assign(speeds*(1-tf.reshape(stable_poses,(-1,1))))      

            if trainspeed:
                grads = tape.gradient(total_loss, [speeds])
                optimizer.apply_gradients(zip(grads, [speeds]))   
            
            # stable_poses = tf.concat([[0],stable_poses], axis=0)
            # speeds.assign(speeds*(1-tf.reshape(stable_poses,(-1,1))))      
            
            if trainorient:
                grads = tape.gradient(imu_loss, [orients])
                optimizer.apply_gradients(zip(grads, [orients]))        

            grads = tape.gradient(psevdo_loss, psevdo_model.trainable_weights)
            optimizer.apply_gradients(zip(grads, psevdo_model.trainable_weights))        

            grads = tape.gradient(phase_loss, phase_model.trainable_weights)
            optimizer.apply_gradients(zip(grads, phase_model.trainable_weights))        
            grads = tape.gradient(dopler_loss, dopler_model.trainable_weights)
            optimizer.apply_gradients(zip(grads, dopler_model.trainable_weights))        
#            if trainimu:
            grads = tape.gradient(imu_loss, imu_model.trainable_weights)
            optimizer.apply_gradients(zip(grads, imu_model.trainable_weights))     
            
            good_qt = tf.concat([[[1.]],tf.cast(tf.reduce_sum(orients[1:]*orients[:-1], axis = -1, keepdims = True) < 0,tf.float32)],axis = 0)
            good_qt = tf.cumsum(good_qt, axis = 0)%2
            #tf.linalg.l2_normalize,axis=-1
            orients.assign(((2*good_qt-1)*orients))

            del tape


        return  total_loss, phase_loss, dopler_loss, acs_loss, quat_loss, speed_loss, psevdo_loss, speeds_loss_Z, g, psevdo_model.get_poses([poses, times])


    def to_float(x):
        return tf.reduce_mean(x).numpy()
    

    trainspeed = True
    #for step in range(0, 32):#
    for step in range(0, 128):
        # if step == -16:
        #     lr = 1e-3
        #     optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
        #     train_step = tf.function(minimize_speederror).get_concrete_function(False, True, False, optimizer)
        # if step == -8:
        #     lr = 1e-3
        #     optimizer = tf.keras.optimizers.Adam(learning_rate=lr)#, epsilon= 0.0001)
        #     train_step = tf.function(minimize_speederror).get_concrete_function(False, True, False, optimizer)
        if step == 0:
            lr = 1e-2
            optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
            #optimizer = tf.keras.optimizers.Adam(learning_rate=lr)#, epsilon= 0.0001)
            #minimize_speederror(False, True, trainspeed, optimizer)
            train_step = tf.function(minimize_speederror).get_concrete_function(False, True, False, optimizer)
            #train_step = tf.function(minimize_speederror).get_concrete_function(False, True, trainspeed, optimizer)
        elif step == 4:
            lr = 1e-3
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)#, epsilon= 0.0001)
            train_step = tf.function(minimize_speederror).get_concrete_function(False, True, trainspeed, optimizer)
        elif step == 8:
            lr = 1e-3
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)#, epsilon= 0.0001)
            speeds_np = speeds.numpy()
            j = 0
            shift = np.array([0.,0.,0.])
            for i in range(len(speeds_np)):
                shift += speeds_np[i]
                while np.linalg.norm(shift) > 3 and j < i:
                    shift_save = shift
                    shift -= speeds_np[j]
                    speeds_np[j] = shift_save
                    j += 1
            while j < len(speeds_np):
                speeds_np[j] = speeds_np[j-1]
                j += 1

            speeds_np[:,2] = 0
            speeds_np = speeds_np/np.linalg.norm(speeds_np,axis=-1,keepdims=True)
            fwd = np.array([0.,1.,0])
            for i in range(or_init.shape[0]):
                or_init[i] = get_quat(fwd, speeds_np[i])

            or_init = or_init.astype(np.float32)
            orients.assign(or_init)
            train_step = tf.function(minimize_speederror).get_concrete_function(False, False, trainspeed, optimizer)
        elif step == 16:
            lr = 1e-3
            #optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
            train_step = tf.function(minimize_speederror).get_concrete_function(False, True, trainspeed, optimizer)
        elif step == 24:
            lr = 1e-3
            #optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
            train_step = tf.function(minimize_speederror).get_concrete_function(True, True, trainspeed, optimizer)
        elif step == 32:
            lr = 1e-3
            #optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            train_step = tf.function(minimize_speederror).get_concrete_function(True, True, True, optimizer)
        elif step == 96:
            lr = 1e-3
            #optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            train_step = tf.function(minimize_speederror).get_concrete_function(True, True, True, optimizer)


        for i in range(128):
            total_loss, phase_loss, dopler_loss, acs_loss, quat_loss, speed_loss,psevdo_loss, speeds_loss_glob, g, poses  = train_step()

        speeds_init_pred = speeds_np = speeds.numpy()
        speeds_e = np.mean(np.linalg.norm((speeds_init_pred-speeds_init_gt)[:,:2],axis=-1))
        
        true_pos = gt_np
        pred_pos = poses.numpy()+np.array([[-1.5,+0.5,0]])
        pos_error = np.linalg.norm((true_pos-pred_pos)[:,:2],axis=-1)
        pos_error = np.sort(pos_error)
        pos_error_abs = (pos_error[len(pos_error)//2] + pos_error[len(pos_error)*95//100])/2

        print( "Training loss at step %d: %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f  lr %.4f er %.4f ser %.4f" % (step, 
            to_float(total_loss), 
            to_float(phase_loss), 
            to_float(dopler_loss), 
            to_float(acs_loss), 
            to_float(quat_loss), 
            to_float(speed_loss), 
            to_float(psevdo_loss), 
            to_float(g), 
            lr,
            pos_error_abs,
            speeds_e
            ))#, end='\r')
        if pos_error_abs < 1.5:
            break
        lr *= 0.99
        optimizer.learning_rate = lr
        if True:#step%16 == 0:
            def conv_numpy(x):
                x = x.numpy()
                return x/np.max(x)
            # phase_loss = conv_numpy(phase_loss)
            # dopler_loss = conv_numpy(dopler_loss)
            # acs_loss = conv_numpy(acs_loss)
            # quat_loss = conv_numpy(quat_loss)
            # speed_loss = conv_numpy(speed_loss)
            acsp, acsgt, acsr = imu_model.get_acses([speeds, gt_speed,orients])
            plt.clf()
            plt.ylim([-2,24])
            plt.plot( np.arange(len(acsp)), acsp[:,0])
            plt.plot( np.arange(len(acsp)), acsp[:,1]+6)
            plt.plot( np.arange(len(acsp)), acsp[:,2]+12)
            plt.plot( np.arange(len(acsr)), acsr[:,0]+2)
            plt.plot( np.arange(len(acsr)), acsr[:,1]+8)
            plt.plot( np.arange(len(acsr)), acsr[:,2]+14)
            plt.plot( np.arange(len(acsgt)), acsgt[:,0]+4)
            plt.plot( np.arange(len(acsgt)), acsgt[:,1]+10)
            plt.plot( np.arange(len(acsgt)), acsgt[:,2]+16)
            plt.plot( np.arange(len(acsr)), acsr[:,0]-acsp[:,0]+18)
            plt.plot( np.arange(len(acsr)), acsr[:,1]-acsp[:,1]+20)
            plt.plot( np.arange(len(acsr)), acsr[:,2]-acsp[:,2]+22)

            plt.legend(['px', 'py', 'pz','tx','ty','tz','gtx','gty','gtz','dx','dy', 'dz'])
            save_fig(image_path+'accel_'+str(step).zfill(3)+'.png')


            tr_quat, pred_quat = imu_model.get_angles([speeds, orients])
            tr_quat *= 10
            pred_quat *= 10
            plt.clf()
            plt.plot( np.arange(len(tr_quat)), tr_quat[:,0])
            plt.plot( np.arange(len(tr_quat)), tr_quat[:,1]+6)
            plt.plot( np.arange(len(tr_quat)), tr_quat[:,2]+12)
            plt.plot( np.arange(len(pred_quat)), pred_quat[:,0]+3)
            plt.plot( np.arange(len(pred_quat)), pred_quat[:,1]+9)
            plt.plot( np.arange(len(pred_quat)), pred_quat[:,2]+15)
            plt.plot( np.arange(len(pred_quat)), pred_quat[:,0] - tr_quat[:,0] + 18)
            plt.plot( np.arange(len(pred_quat)), pred_quat[:,1] - tr_quat[:,1] + 21)
            plt.plot( np.arange(len(pred_quat)), pred_quat[:,2] - tr_quat[:,2] + 24)

            plt.legend(['tx','ty','tz','px', 'py', 'pz','dx','dy', 'dz'])
            save_fig(image_path+'gyro_'+str(step).zfill(3)+'.png')

            total_loss = conv_numpy(total_loss)
            plt.clf()
            # plt.plot( np.arange(len(speeds_init_gt)), speeds_init_pred[:,0] - speeds_init_gt[:,0])
            # plt.plot( np.arange(len(speeds_init_gt)), speeds_init_pred[:,1] - speeds_init_gt[:,1])
            plt.plot( np.arange(len(speeds_init_gt)), speeds_init_pred[:,0])
            plt.plot( np.arange(len(speeds_init_gt)), speeds_init_gt[:,0])
            plt.plot( np.arange(len(speeds_init_gt)), speeds_init_pred[:,1])
            plt.plot( np.arange(len(speeds_init_gt)), speeds_init_gt[:,1])
            # plt.plot( np.arange(len(total_loss)), total_loss)
            # plt.plot( np.arange(len(phase_loss)), phase_loss/10)
            # plt.plot( np.arange(len(dopler_loss)), dopler_loss/10)
            plt.plot( np.arange(len(acs_loss)), acs_loss)
            plt.plot( np.arange(len(acs_loss)), quat_loss)
            plt.plot( np.arange(len(acs_loss)), speed_loss)
            plt.plot( np.arange(len(speeds_loss_glob)), speeds_loss_glob)
            
            # plt.plot( np.arange(len(psevdo_loss)), psevdo_loss/100)
            
            # plt.plot( np.arange(len(quat_loss)), quat_loss)
            # plt.plot( np.arange(len(speed_loss)), speed_loss)

            plt.legend(['dx', 'dy', 'acs','quat','speed', 'speed dif'])
            #plt.legend(['dx', 'dy', 'phase', 'dopler', 'acs','quat','speed', 'psevdo'])
            #plt.show()
            save_fig(image_path+'speed_diff'+str(step).zfill(3)+'.png')


            # plt.plot( np.arange(len(speeds_init_gt)), speeds_init_gt[:,0])
            # plt.plot( np.arange(len(speeds_init_gt)), speeds_init_gt[:,1])
            # plt.plot( np.arange(len(speeds_init_gt)), speeds_init_pred[:,0])
            # plt.plot( np.arange(len(speeds_init_gt)), speeds_init_pred[:,1])
            # plt.legend(['true x', 'true y', 'pred x', 'pred y'])
            valid_weights = np.sum(sat_deltaweights>0, axis = -1).astype(np.float32)
            valid_weights[valid_weights>=6] = NaN
            valid_weights[valid_weights<6] = 0
            plt.clf()
            plt.plot( np.arange(len(true_pos)), true_pos[:,0] - pred_pos[:,0])
            plt.plot( np.arange(len(true_pos)), true_pos[:,1] - pred_pos[:,1])
            plt.plot( np.arange(len(pred_pos)-1), pred_pos[1:,2] - pred_pos[:-1,2])
            plt.plot( np.arange(len(valid_weights)), valid_weights, marker='o')
            sat_deltaspeeduncertcount = np.sum(sat_deltaspeeduncert<1, axis = -1).astype(np.float32)
            sat_deltaspeeduncertcount[sat_deltaspeeduncertcount>=15] = NaN
            sat_deltaspeeduncertcount[sat_deltaspeeduncertcount< 15] = 0
            plt.plot( np.arange(len(sat_deltaspeeduncertcount)), sat_deltaspeeduncertcount, marker='o', linestyle = None)
            sat_deltaspeeduncertcount = np.sum(sat_deltaspeeduncert<1, axis = -1).astype(np.float32)
            sat_deltaspeeduncertcount[sat_deltaspeeduncertcount>=10] = NaN
            sat_deltaspeeduncertcount[sat_deltaspeeduncertcount< 10] = 0
            plt.plot( np.arange(len(sat_deltaspeeduncertcount)), sat_deltaspeeduncertcount, marker='o', linestyle = None)
            plt.legend(['dif x', 'dif y', 'speed z', 'deltas', 'speeds15', 'speeds10'])
            save_fig(image_path+'track_shift'+str(step).zfill(3)+'.png')

            plt.clf()
            speeds_np = speeds.numpy()
            speeds_np = speeds_np/(np.linalg.norm(speeds_np, axis=-1,keepdims=True) + 0.5)
            speeds_np1 = gt_np[1:]-gt_np[:-1]
            speeds_np1 = speeds_np1/(np.linalg.norm(speeds_np1, axis=-1,keepdims=True) + 0.5)
            quats_np = orients.numpy()
            quats_np = quats_np/(np.linalg.norm(quats_np, axis=-1,keepdims=True))
            #speeds_np_f = transform(speeds_np,quats_np)
            #speeds_np_b = transform(speeds_np,inv(quats_np))
            plt.clf()
            plt.plot( np.arange(len(quats_np)), speeds_np[:,0])
            plt.plot( np.arange(len(quats_np)), speeds_np[:,1])
            plt.plot( np.arange(len(speeds_np1)), speeds_np1[:,0])
            plt.plot( np.arange(len(speeds_np1)), speeds_np1[:,1])
            plt.plot( np.arange(len(quats_np)), quats_np[:,0])
            plt.plot( np.arange(len(quats_np)), quats_np[:,1])
            plt.plot( np.arange(len(quats_np)), quats_np[:,2])
            plt.legend(['bx', 'by','tx', 'ty','q x', 'q y','q z'])
            save_fig(image_path+'quat'+str(step).zfill(3)+'.png')

            kml = KMLWriter(image_path+ "kml_"+str(step).zfill(3)+".kml", "predicted")
            wgs_poses = ecef2geodetic(local_transform_inv(true_pos))
            kml.addTrack('true_'+str(step),'FF00FF00', wgs_poses)
            wgs_poses = ecef2geodetic(local_transform_inv(pred_pos))
            kml.addTrack('predicted_'+str(step),'FFFF0000', wgs_poses)
            kml.finish()


    ind = np.argpartition(np.linalg.norm(true_pos[:,:2] - pred_pos[:,:2],axis=-1), -10)[-10:]            
    abc = 0




def calc_local_speed(sat_dir, lens):
    sat_dir[np.isnan(sat_dir)] = 0
    lens[np.isnan(lens)] = 0

    n = len(lens)
    if n < 8: 
        return False, [], np.zeros((n))
    sat_dir = np.append(sat_dir, np.ones((n,1)), axis = 1)
    min_err = 1e10
    res_shift = [[0.,0.,0.,0.]]
    for _ in range(500):
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
        curerr = np.sum(np.minimum(rows,0.2))
        if curerr < min_err:
            min_err = curerr
            res_shift = x_hat
            if np.sum(rows<0.1) > 8:
                break
    
    cur_shifts = np.sum(sat_dir*res_shift, axis=1)
    rows = np.abs(cur_shifts - lens)
    return np.sum(rows<0.1) > 8, res_shift[0], rows



def calc_shift_from_deltas(sat_dir_in, lens_in, valid):

    n = np.sum(valid > 0)
    if n < 6:
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
        curerr = np.sum(np.minimum(rows,0.2))
        if curerr < min_err:
            min_err = curerr
            res_shift = x_hat
            #if np.sum(rows<0.1) > 7:
            #    break
    
    sat_dir = sat_dir_in.copy()
    lens = lens_in
    sat_dir = np.append(sat_dir, np.ones((len(sat_dir),1)), axis = 1)
    cur_shifts = np.sum(sat_dir*res_shift, axis=1)
    rows = np.abs(cur_shifts - lens)
    return np.sum(rows<0.1) > 8, res_shift[0], rows

def load_track_accrange(trip_id):
    return preload_gnss_log(f'{GOOGLE_DATA_ROOT}{trip_id}')

def analyze_derivative(trip_id):
    NaN = float("NaN")

    cur_measure = load_track_accrange(trip_id)
    baselines = cur_measure["baseline"]
    sat_positions = cur_measure["sat_positions"]
    sat_deltarange = cur_measure["sat_deltarange"]
    sat_deltavalid = cur_measure["sat_deltavalid"]
    epoch_times = cur_measure["epoch_times"]
    num_epochs  = len(epoch_times)
    num_used_satelites  = len(sat_positions[0])

    gt_raw = pd.read_csv(f'{GOOGLE_DATA_ROOT}{trip_id}/ground_truth.csv')
    gt_np = gt_raw[['LatitudeDegrees','LongitudeDegrees','AltitudeMeters']].to_numpy()
    gt_np[np.isnan(gt_np)] = 0
    gt_np = geodetic2ecef(gt_np)

    
    coords_start = baselines[0].copy()
    mat_local = np.zeros((3,3))
    mat_local[2] = coords_start/np.linalg.norm(coords_start, axis = -1)
    mat_local[1] = np.array([0,0,1])
    mat_local[1] = mat_local[1] - mat_local[2]*np.sum(mat_local[2]*mat_local[1])
    mat_local[1] = mat_local[1]/np.linalg.norm(mat_local[1], axis = -1)
    mat_local[0] = np.cross(mat_local[1], mat_local[2])
    mat_local = np.transpose(mat_local)

    locl_trans = mat_local
    locl_shift = coords_start
    def local_transform(v):
        return np.matmul(v - locl_shift,locl_trans)
    def local_transform_inv(v):
        return np.matmul(v,locl_trans.T) + locl_shift

    baselines = local_transform(baselines)
    sat_positions = local_transform(sat_positions)
    gt_np = local_transform(gt_np)




    baselines = np.reshape(baselines,(-1,1,3))
    sat_directions = sat_positions - baselines  
    sat_directions = sat_directions/np.linalg.norm(sat_directions,axis=-1,keepdims=True)

    sat_directions = (sat_directions[1:] + sat_directions[:-1])/2
    sat_deltarange = sat_deltarange[1:] - sat_deltarange[:-1]
    sat_deltavalid = sat_deltavalid[1:]*sat_deltavalid[:-1]

    baselines = baselines[:-1]
    sat_distanse_dif = np.sum((sat_positions[1:] - sat_positions[:-1])*sat_directions, axis = -1)
    sat_deltarange -= sat_distanse_dif

    
    
    
    true_shifts = gt_np[1:] - gt_np[:-1]
    pred_shifts = np.zeros((num_epochs-1,3))

    sat_deltarange[sat_deltavalid == 0] = NaN
    sat_deltarange_true = sat_deltarange.copy()

    for i in range(num_epochs-1):
        for j in range(num_used_satelites):
            sat_deltarange_true[i,j] = np.sum(sat_directions[i,j]*true_shifts[i],axis=-1)
        sat_deltarange_true[i] += sat_deltarange[i]
        sat_deltarange_true[i] -= np.median(sat_deltarange_true[i,~np.isnan(sat_deltarange_true[i])])
    
    plt.clf()
    sat_deltarange_true[np.abs(sat_deltarange_true)>10] = NaN
    for j in range(num_used_satelites):
        plt.plot( np.arange(len(sat_deltarange)), j + sat_deltarange_true[:,j])
    plt.show()



    for i in range(num_epochs-1):
        ret, xhat, rows = calc_shift_from_deltas(sat_directions[i],sat_deltarange[i], sat_deltavalid[i].copy())
        if ret and np.linalg.norm(xhat[:3]) < 30:
            pred_shifts[i] = xhat[:3]
        else:
            pred_shifts[i] = [NaN,NaN,NaN]

    plt.clf()
    plt.plot( np.arange(len(true_shifts)), true_shifts[:,0] + pred_shifts[:,0])
    plt.plot( np.arange(len(true_shifts)), true_shifts[:,1] + pred_shifts[:,1])
    #plt.plot( np.arange(len(pred_shifts)), pred_shifts[:,0])
    #plt.plot( np.arange(len(pred_shifts)), pred_shifts[:,1])

    plt.legend(['dif x', 'dif y'])
    plt.show()

    replace_nans = np.isnan(pred_shifts)
    pred_shifts[replace_nans] = -true_shifts[replace_nans]

    true_pos = np.cumsum(true_shifts,0)
    pred_pos = np.cumsum(pred_shifts,0)

    plt.clf()
    plt.plot( np.arange(len(true_shifts)), true_pos[:,0] + pred_pos[:,0])
    plt.plot( np.arange(len(true_shifts)), true_pos[:,1] + pred_pos[:,1])
    #plt.plot( np.arange(len(pred_shifts)), pred_shifts[:,0])
    #plt.plot( np.arange(len(pred_shifts)), pred_shifts[:,1])

    plt.legend(['dif x', 'dif y'])
    plt.show()

def get_num_good_shifts(trip_id):
    NaN = float("NaN")

    cur_measure = load_track_accrange(trip_id)
    baselines = cur_measure["baseline"]
    sat_positions = cur_measure["sat_positions"]
    sat_deltarange = cur_measure["sat_deltarange"]
    sat_deltavalid = cur_measure["sat_deltavalid"]
    epoch_times = cur_measure["epoch_times"]
    num_epochs  = len(epoch_times)
    num_used_satelites  = len(sat_positions[0])

    coords_start = baselines[0].copy()
    mat_local = np.zeros((3,3))
    mat_local[2] = coords_start/np.linalg.norm(coords_start, axis = -1)
    mat_local[1] = np.array([0,0,1])
    mat_local[1] = mat_local[1] - mat_local[2]*np.sum(mat_local[2]*mat_local[1])
    mat_local[1] = mat_local[1]/np.linalg.norm(mat_local[1], axis = -1)
    mat_local[0] = np.cross(mat_local[1], mat_local[2])
    mat_local = np.transpose(mat_local)

    locl_trans = mat_local
    locl_shift = coords_start
    def local_transform(v):
        return np.matmul(v - locl_shift,locl_trans)
    def local_transform_inv(v):
        return np.matmul(v,locl_trans.T) + locl_shift

    baselines = local_transform(baselines)
    sat_positions = local_transform(sat_positions)


    baselines = np.reshape(baselines,(-1,1,3))
    sat_directions = sat_positions - baselines  
    sat_directions = sat_directions/np.linalg.norm(sat_directions,axis=-1,keepdims=True)

    sat_directions = (sat_directions[1:] + sat_directions[:-1])/2
    sat_deltarange = sat_deltarange[1:] - sat_deltarange[:-1]
    sat_deltavalid = sat_deltavalid[1:]*sat_deltavalid[:-1]

    baselines = baselines[:-1]
    sat_distanse_dif = np.sum((sat_positions[1:] - sat_positions[:-1])*sat_directions, axis = -1)
    sat_deltarange -= sat_distanse_dif
  

    num_good = 0
    for i in range(num_epochs-1):
        ret, xhat, rows = calc_shift_from_deltas(sat_directions[i],sat_deltarange[i], sat_deltavalid[i].copy())
        if ret and np.linalg.norm(xhat[:3]) < 30:
            num_good += 1
        else:
            ret, xhat, rows = calc_shift_from_deltas(sat_directions[i],sat_deltarange[i], sat_deltavalid[i].copy())
    return num_good, num_epochs

def get_num_good_speeds(trip_id):
    NaN = float("NaN")

    gt_raw = pd.read_csv(f'{GOOGLE_DATA_ROOT}{trip_id}/ground_truth.csv')
    gt_np = gt_raw[['LatitudeDegrees','LongitudeDegrees','AltitudeMeters']].to_numpy()
    gt_np[np.isnan(gt_np)] = 0
    gt_np = geodetic2ecef(gt_np)

    cur_measure = load_track_accrange(trip_id)
    baselines = cur_measure["baseline"]
    sat_positions  = cur_measure["sat_positions"]
    sat_velosities = cur_measure["sat_velosities"]
    sat_deltaspeed = cur_measure["sat_deltaspeed"]
    sat_deltaspeeduncert = cur_measure["sat_deltaspeeduncert"]
    sat_deltarange = cur_measure["sat_deltarange"]
    sat_deltavalid = cur_measure["sat_deltavalid"]
    epoch_times = cur_measure["epoch_times"]
    num_epochs  = len(epoch_times)
    num_used_satelites  = len(sat_positions[0])

    coords_start = baselines[0].copy()
    mat_local = np.zeros((3,3))
    mat_local[2] = coords_start/np.linalg.norm(coords_start, axis = -1)
    mat_local[1] = np.array([0,0,1])
    mat_local[1] = mat_local[1] - mat_local[2]*np.sum(mat_local[2]*mat_local[1])
    mat_local[1] = mat_local[1]/np.linalg.norm(mat_local[1], axis = -1)
    mat_local[0] = np.cross(mat_local[1], mat_local[2])
    mat_local = np.transpose(mat_local)

    locl_trans = mat_local
    locl_shift = coords_start
    def local_transform(v):
        return np.matmul(v - locl_shift,locl_trans)
    def local_vector_transform(v):
        return np.matmul(v,locl_trans)
    def local_transform_inv(v):
        return np.matmul(v,locl_trans.T) + locl_shift

    valid = (np.linalg.norm(sat_velosities) > 10)*(np.linalg.norm(sat_positions) > 10)*(np.abs(sat_deltaspeed) > 0)*(sat_deltaspeeduncert < 0.2)
    baselines = local_transform(baselines)
    sat_positions = local_transform(sat_positions)
    gt_np = local_transform(gt_np)
    sat_velosities = local_vector_transform(sat_velosities)

    baselines = np.reshape(baselines,(-1,1,3))
    sat_directions = sat_positions - baselines  
    sat_directions = sat_directions/np.linalg.norm(sat_directions,axis=-1,keepdims=True)
    sat_scalar_velosities = np.sum(sat_velosities*sat_directions,axis=-1)
    
    sat_distanse = np.linalg.norm(baselines - sat_positions,axis=-1)

    sat_distanse_dif = np.sum((sat_positions[1:] - sat_positions[:-1])*sat_directions[1:], axis = -1)
    sat_deltarange = sat_deltarange[1:] - sat_deltarange[:-1]
    all = np.stack([sat_deltaspeed[:-1],sat_scalar_velosities[:-1], sat_distanse_dif,sat_deltarange], axis = -1)
    sat_deltaspeed -= sat_scalar_velosities
    sat_deltarange -= sat_distanse_dif
    num_good = 0
    x0=[0, 0, 0, 0,0,0,0,0,0]
    pred_shifts = np.zeros((num_epochs-1,3))
    true_shifts = gt_np[1:] - gt_np[:-1]
    for i in range(num_epochs-1):
        x0, err = calc_speed(sat_directions[i],sat_deltaspeed[i], np.ones((num_used_satelites)), valid[i],cur_measure['sat_types'],x0)
        if len(err) > 8 and np.sum(err<0.8) > 8:
            num_good += 1
            pred_shifts[i] = x0[:3]
        else:
            pred_shifts[i] = [NaN,NaN,NaN]

    plt.clf()
    plt.plot( np.arange(len(true_shifts)), true_shifts[:,0] + pred_shifts[:,0])
    plt.plot( np.arange(len(true_shifts)), true_shifts[:,1] + pred_shifts[:,1])
    #plt.plot( np.arange(len(pred_shifts)), pred_shifts[:,0])
    #plt.plot( np.arange(len(pred_shifts)), pred_shifts[:,1])

    plt.legend(['dif x', 'dif y'])
    plt.show()

    replace_nans = np.isnan(pred_shifts)
    pred_shifts[replace_nans] = -true_shifts[replace_nans]

    true_pos = np.cumsum(true_shifts,0)
    pred_pos = np.cumsum(pred_shifts,0)

    plt.clf()
    plt.plot( np.arange(len(true_shifts)), true_pos[:,0] + pred_pos[:,0])
    plt.plot( np.arange(len(true_shifts)), true_pos[:,1] + pred_pos[:,1])
    #plt.plot( np.arange(len(pred_shifts)), pred_shifts[:,0])
    #plt.plot( np.arange(len(pred_shifts)), pred_shifts[:,1])

    plt.legend(['dif x', 'dif y'])
    plt.show()
    return num_good, num_epochs