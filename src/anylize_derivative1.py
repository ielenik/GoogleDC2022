import numpy as np
import pandas as pd
import pickle
import os 
from tqdm import tqdm
import matplotlib.pyplot as plt
import datetime

from src.laika.lib.coordinates import ecef2geodetic, geodetic2ecef
from src.laika.astro_dog import AstroDog
from src.laika.gps_time import GPSTime
from src.laika.downloader import download_cors_station

from .utils.loader import myLoadRinexIndexed, myLoadRinex
from .utils.gnss_log_reader import gnss_log_to_dataframes
from .utils.gnss_log_processor import process_gnss_log
from .utils.coords_tools import calc_pos_fix, calc_speed


GOOGLE_DATA_ROOT = r'D:\databases\smartphone-decimeter-2022'

def preload_gnss_log(path_folder):
    try:
        with open(path_folder+'/data.cache.pkl', 'rb') as f:
            cur_measure =  pickle.load(f)
    except:
        constellations = ['GPS', 'GLONASS', 'BEIDOU','GALILEO']
        dog = AstroDog(valid_const=constellations, pull_orbit=True)

        trackname = path_folder.split('\\')[-2]
        date_str = trackname.split('-')

        datetimenow = datetime.datetime.strptime('/'.join(date_str[:3]), '%Y/%m/%d')
        datetimenow = GPSTime.from_datetime(datetimenow)

        path_basestation = download_cors_station(datetimenow,'slac', path_folder)
        basestation, coords, satregistry = myLoadRinexIndexed(path_basestation)

        basestation = { 
            'times': np.array([r[0] for r in basestation]), 
            'values':np.array([r[1] for r in basestation]),
            'coords':coords,
            'sat_registry':satregistry,
            }

        df_raw = pd.read_csv(f'{path_folder}/device_gnss.csv')
        cur_measure = process_gnss_log(df_raw, basestation, dog)
        num_epoch = len(cur_measure['epoch_times'])
        x0=[0, 0, 0, 0,0,0,0,0,0]
        baseline_poses = []
        for i in tqdm(range(num_epoch)):
            x0, err = calc_pos_fix(cur_measure['sat_positions'][i], cur_measure['sat_psevdodist'][i], cur_measure['sat_psevdoweights'][i],cur_measure['sat_psevdovalid'][i] > 0, cur_measure['sat_types'], x0)
            if np.sum(err < 1000) > 10:
                break #initial guess
        for i in tqdm(range(num_epoch)):
            x0, _ = calc_pos_fix(cur_measure['sat_positions'][i], cur_measure['sat_psevdodist'][i], cur_measure['sat_psevdoweights'][i],cur_measure['sat_psevdovalid'][i] > 0, cur_measure['sat_types'], x0)
            baseline_poses.append(x0[:3])

        cur_measure['baseline'] = np.array(baseline_poses)

    with open(path_folder+'/data.cache.pkl', 'wb') as f:
        pickle.dump(cur_measure, f, pickle.HIGHEST_PROTOCOL)

    return cur_measure

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
        curerr = np.sum(np.maximum(rows,0.2))
        if curerr < min_err:
            min_err = curerr
            res_shift = x_hat
    
    sat_dir = sat_dir_in.copy()
    lens = lens_in
    sat_dir = np.append(sat_dir, np.ones((len(sat_dir),1)), axis = 1)
    cur_shifts = np.sum(sat_dir*res_shift, axis=1)
    rows = np.abs(cur_shifts - lens)
    return np.sum(rows<0.1) > 7, res_shift[0], rows

def load_track_accrange(trip_id):
    return preload_gnss_log(f'{GOOGLE_DATA_ROOT}{trip_id}')

def analyze_derivative1(trip_id):
    NaN = float("NaN")
    gt_raw = pd.read_csv(f'{GOOGLE_DATA_ROOT}{trip_id}/ground_truth.csv')
    gt_np = gt_raw[['LatitudeDegrees','LongitudeDegrees','AltitudeMeters']].to_numpy()
    gt_np[np.isnan(gt_np)] = 0
    gt_np = geodetic2ecef(gt_np)

    '''
    cur_measure = load_track_accrange(trip_id)
    baselines = cur_measure["baseline"]
    sat_positions = cur_measure["sat_positions"]
    sat_deltarange = cur_measure["sat_deltarange"]
    sat_deltavalid = cur_measure["sat_deltavalid"]
    epoch_times = cur_measure["epoch_times"]
    num_epochs  = len(epoch_times)
    num_used_satelites  = len(sat_positions[0])
    '''

    df_raw = pd.read_csv(f'{GOOGLE_DATA_ROOT}{trip_id}/device_gnss.csv')

    df_raw['NanosSinceGpsEpoch'] = df_raw['TimeNanos'] - df_raw['FullBiasNanos']
    df_raw['SvNameType'] = df_raw['SignalType'] + '_' + df_raw['Svid'].apply(str)
    df_raw['SAT_FULL_INDEX'] = pd.factorize(df_raw['SvNameType'].tolist())[0]    

    df_raw['Epoch'] = 0
    df_raw.loc[df_raw['NanosSinceGpsEpoch'] - df_raw['NanosSinceGpsEpoch'].shift() > 10*1e6, 'Epoch'] = 1
    df_raw['Epoch'] = df_raw['Epoch'].cumsum()

    df_raw["adr_valid"] = (df_raw["AccumulatedDeltaRangeState"] & 2**0) != 0
    df_raw["adr_reset"] = (df_raw["AccumulatedDeltaRangeState"] & 2**1) != 0
    df_raw["adr_slip"]  = (df_raw["AccumulatedDeltaRangeState"] & 2**2) != 0

    df_raw["delta_valid"] = ~(~df_raw["adr_valid"] | df_raw["adr_reset"] | df_raw["adr_slip"])

    num_used_satelites = df_raw['SAT_FULL_INDEX'].max() + 1
    num_epochs         = df_raw['Epoch'].max() + 1

    baselines  = np.ones((num_epochs, 3))
    sat_positions  = np.ones((num_epochs, num_used_satelites, 3))
    sat_deltarange = np.zeros((num_epochs, num_used_satelites))
    sat_deltavalid = np.zeros((num_epochs, num_used_satelites))
    
    epoch_times    = np.zeros((num_epochs)).astype(np.int64)
    NaN = float("NaN")

    for epoch_number, epoch in tqdm(df_raw.groupby(['Epoch'])):
        time_nanos = epoch["NanosSinceGpsEpoch"].to_numpy()
        time_nanos = np.sort(time_nanos)
        time_nanos = time_nanos[len(time_nanos)//2]

        epoch_times[epoch_number] = time_nanos

        for _,r in epoch.iterrows():
            sat_index = r['SAT_FULL_INDEX']
            if sat_index == -1:
                continue

            sat_deltarange[epoch_number,sat_index] = r['AccumulatedDeltaRangeMeters']
            sat_positions[epoch_number,sat_index] = r[['SvPositionXEcefMeters','SvPositionYEcefMeters','SvPositionZEcefMeters']]
            baselines[epoch_number] = r[['WlsPositionXEcefMeters','WlsPositionYEcefMeters','WlsPositionZEcefMeters']]
            if sat_deltarange[epoch_number,sat_index] != 0:
                sat_deltavalid[epoch_number,sat_index] = r["delta_valid"]
    
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
    sat_distanse_first = np.linalg.norm(baselines - sat_positions[:-1],axis=-1)
    sat_distanse_next  = np.linalg.norm(baselines - sat_positions[1:],axis=-1)
    sat_distanse_dif = sat_distanse_next - sat_distanse_first
    sat_deltarange -= sat_distanse_dif

    
    
    
    true_shifts = gt_np[1:] - gt_np[:-1]
    pred_shifts = np.zeros((num_epochs-1,3))

    sat_deltarange[sat_deltavalid == 0] = NaN
    sat_deltarange_true = sat_deltarange.copy()

    for i in range(num_epochs-1):
        for j in range(num_used_satelites):
            sat_deltarange_true[i,j] = np.sum(sat_directions[i,j]*true_shifts[i],axis=-1)
    
    plt.clf()
    sat_deltarange_true += sat_deltarange
    sat_deltarange_true -= np.median(sat_deltarange_true,axis=-1,keepdims=True)
    
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

