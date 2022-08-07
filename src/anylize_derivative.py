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
from scipy import ndimage, misc

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
from .predict_angle import calc_initial_speed_guess

GOOGLE_DATA_ROOT = r'D:\databases\smartphone-decimeter-2022'

NaN = float("NaN")

def preload_gnss_log(path_folder):
    try:
        with open(path_folder+'/data.cache.pkl', 'rb') as f:
            cur_measure =  pickle.load(f)
    except:
        constellations = ['GPS', 'GLONASS', 'BEIDOU','GALILEO']
        dog = AstroDog(valid_const=constellations, pull_orbit=True)

        gnss_log = gnss_log_to_dataframes(f'{path_folder}/supplemental/gnss_log.txt')
        df_raw = gnss_log['Raw']

        acs = gnss_log['UncalAccel']
        acstimes = acs['utcTimeMillis'].to_numpy()
        acsvalues = acs[['UncalAccelXMps2','UncalAccelYMps2','UncalAccelZMps2']].to_numpy()
        gir = gnss_log['UncalGyro']
        gyrtimes = gir['utcTimeMillis'].to_numpy()
        gyrvalues = gir[['UncalGyroXRadPerSec','UncalGyroYRadPerSec','UncalGyroZRadPerSec']].to_numpy()

        #df_raw = pd.read_csv(f'{path_folder}/device_gnss.csv')
        datetimenow = datetime.datetime.utcfromtimestamp(df_raw['utcTimeMillis'].iat[0]/1000.0)
        gpsdatetimenow = GPSTime.from_datetime(datetimenow)

        cache_path = '\\'.join(path_folder.split('\\')[:-1])
        if 'LAX' in path_folder:
            path_basestation = download_cors_station(gpsdatetimenow,'vdcy', cache_path)
        else:
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
        x0=np.zeros((9))
        baseline_poses = []
        for i in tqdm(range(num_epoch)):
            x0, err = calc_pos_fix(cur_measure['sat_positions'][i], cur_measure['sat_psevdodist'][i], cur_measure['sat_psevdoweights'][i],cur_measure['sat_psevdovalid'][i] > 0, cur_measure['sat_types'], x0)
            if np.sum(err < 1000) > 10:
                break #initial guess
        for i in tqdm(range(num_epoch)):
            if i == 670:
                _ = 12321
            x0, err = calc_pos_fix(cur_measure['sat_positions'][i], cur_measure['sat_psevdodist'][i], cur_measure['sat_psevdoweights'][i],cur_measure['sat_psevdovalid'][i] > 0, cur_measure['sat_types'], x0)
            if np.sum(err < 1000) < 8:
                baseline_poses.append([NaN,NaN,NaN])
                x0=np.zeros((9))
            else:
                baseline_poses.append(x0[:3])

        baseline_poses = np.array(baseline_poses)
        for i in range(3):
            nans, x= np.isnan(baseline_poses[:,i]), lambda z: z.nonzero()[0]
            baseline_poses[nans,i]= np.interp(x(nans), x(~nans), baseline_poses[~nans,i])

        cur_measure['baseline'] = np.array(baseline_poses)
        cur_measure['basestation'] = basestation
        cur_measure['gyrtimes'] = gyrtimes
        cur_measure['gyrvalues'] = gyrvalues
        cur_measure['acstimes'] = acstimes
        cur_measure['acsvalues'] = acsvalues


    with open(path_folder+'/data.cache.pkl', 'wb') as f:
        pickle.dump(cur_measure, f, pickle.HIGHEST_PROTOCOL)

    return cur_measure

from .dc2.tf_dopler_model import createDoplerModel
from .dc2.tf_xiaomi_dopler_model import createXiaomiDoplerModel
from .dc2.tf_phase_model import createPhaseModel
from .dc2.tf_imu_model import createRigidModel
from .dc2.tf_psevdo_model import createPsevdoModel

def save_fig(path):
    try:
        figure = plt.gcf() # get current figure
        figure.set_size_inches(24, 18)
        # when saving, specify the DPI
        plt.savefig(path, dpi = 100)    
        with open(path+'.pltpkl','wb') as fid:
            pickle.dump(figure.get_axes(), fid)
    except:
        print("Failed saved fig", path)

def sigmoid(z):
    return 1/(1 + np.exp(-z))


def export_baseline(trip_id):
    cur_measure = preload_gnss_log(f'{GOOGLE_DATA_ROOT}{trip_id}')
    baselines            = cur_measure["baseline"]
    with open(f'{GOOGLE_DATA_ROOT}{trip_id}'+'/baseline.pkl', 'wb') as f:
        pickle.dump(baselines, f, pickle.HIGHEST_PROTOCOL)
def export_times(trip_id):
    cur_measure = preload_gnss_log(f'{GOOGLE_DATA_ROOT}{trip_id}')
    times = cur_measure["utcTimeMillis"]
    with open(f'{GOOGLE_DATA_ROOT}{trip_id}'+'/times.pkl', 'wb') as f:
        pickle.dump(times, f, pickle.HIGHEST_PROTOCOL)

def interp_nd(epochtimes, valtimes, valcum):
    dim = []
    shape = list(valcum.shape)
    valcum = valcum.reshape((len(valcum),-1))
    for i in range(len(valcum[0])):
        dim.append(np.interp(epochtimes, valtimes, valcum[:,i]))
    shape[0] = len(epochtimes)
    valcum = np.array(dim).T.reshape(shape)
    return valcum 
def calc_track_speed(trip_id):
    image_path = f'{GOOGLE_DATA_ROOT}{trip_id}/plots/'
    os.makedirs(image_path, exist_ok=True)
    os.makedirs(image_path+'\\qtr', exist_ok=True)
    os.makedirs(image_path+'\\spd', exist_ok=True)
    os.makedirs(image_path+'\\sft', exist_ok=True)
    os.makedirs(image_path+'\\kml', exist_ok=True)
    os.makedirs(image_path+'\\acc', exist_ok=True)
    os.makedirs(image_path+'\\gyr', exist_ok=True)
    os.makedirs(image_path+'\\crs', exist_ok=True)
    os.makedirs(image_path+'\\lss', exist_ok=True)

    cur_measure = preload_gnss_log(f'{GOOGLE_DATA_ROOT}{trip_id}')
    st_seg = 0
    en_seg = None
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

    gyrtimes = cur_measure['gyrtimes']
    gyrvalues = cur_measure['gyrvalues']
    acstimes = cur_measure['acstimes']
    acsvalues = cur_measure['acsvalues']

    sat_wavelength       = 299_792_458/sat_frequencis
    sat_basedeltarange *= sat_wavelength


    if 'train' in trip_id:
        gt_raw = pd.read_csv(f'{GOOGLE_DATA_ROOT}{trip_id}/ground_truth.csv')
        gt_np = gt_raw[['LatitudeDegrees','LongitudeDegrees','AltitudeMeters']].to_numpy()
        gt_npmillis = gt_raw['UnixTimeMillis'].to_numpy()
        gt_np[np.isnan(gt_np)] = 0
        gt_np = geodetic2ecef(gt_np)

        gt_np = interp_nd(utcTimeMillis,gt_npmillis,gt_np)
    else:
        gt_np = baselines.copy()

    kml = KMLWriter(image_path+ "kml\\kml_true.kml", "True")
    wgs_poses = ecef2geodetic(gt_np)
    kml.addTrack('true','FF00FF00', wgs_poses)
    kml.finish()

    kml = KMLWriter(image_path+ "kml\\kml_baseline.kml", "Baseline")
    wgs_poses = ecef2geodetic(baselines)
    kml.addTrack('baseline','FF0000FF', wgs_poses)
    kml.finish()

    num_epochs  = len(utcTimeMillis)
    num_used_satelites  = len(sat_types)
    # baselines += [ 0,0,40 ] chekc is shift found
    coords_start =baselines[0]
    
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
    gt_np = local_transform(gt_np) - np.array([[-1.5,+0.5,0]])
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
    baseline_stable = baselines.copy()

    baselines = np.reshape(baselines,(-1,1,3))
    baseline_stable = np.reshape(baseline_stable,(-1,1,3))

    sat_directions = sat_positions - baseline_stable
    sat_distances = np.linalg.norm(sat_directions,axis=-1,keepdims=True)
    sat_directions = sat_directions/sat_distances
    sat_scalar_velosities = np.sum(sat_velosities*sat_directions,axis=-1)
    sat_deltaspeed -= sat_scalar_velosities

    imu_model = createRigidModel(utcTimeMillis,acsvalues,acstimes,gyrvalues,gyrtimes)
    if 'Xiao' in trip_id:
        print("Configuring Xiaomi - reducing dopler weight")
        dopler_model = createXiaomiDoplerModel(sat_directions, -sat_deltaspeed, sat_deltaspeeduncert, sat_types)
    else:
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

    test1 = np.sum(sat_directions*(baseline_stable-np.reshape(gt_np,(-1,1,3))), axis = -1)
    test2 = (test1 + sat_psevdo_shift)*sat_psevdoweights
    test = np.mean(np.abs(test2),axis = -1)
    times = tf.Variable(np.reshape(np.arange(0,len(sat_directions))*2/len(sat_directions)-1,(-1,1)),trainable=False,dtype=tf.float32)
    psevdo_model = createPsevdoModel(sat_directions, sat_psevdo_shift, sat_psevdoweights, sat_types)




    #sat_directions = (sat_directions[1:]+sat_directions[:-1])/2
    sat_deltarange = sat_deltarange[1:] - sat_deltarange[:-1]
    sat_distanse_dif = np.linalg.norm(sat_positions[1:] - baseline_stable[:-1],axis=-1) - np.linalg.norm(sat_positions[:-1] - baselines[:-1],axis=-1)
    sat_deltarange -= sat_distanse_dif


    sat_deltarangeuncert[sat_deltarangeuncert<1e-8] = 1
    if np.sum(sat_deltastate == 17) > np.sum(sat_deltastate == 25):
        sat_deltastate[sat_deltastate == 17] = 25

    sat_deltaweights = (sat_deltastate == 25)/sat_deltarangeuncert
    sat_deltaweights = np.minimum(sat_deltaweights[1:],sat_deltaweights[:-1])
    sat_deltarange[sat_deltaweights == 0] = NaN
    sat_deltarange -= np.nanmedian(sat_deltarange, axis = -1, keepdims=True)
    sat_deltarange[sat_deltaweights == 0] = 0

 
    baseline_stable = baseline_stable.reshape((-1,3))
    phase_model = createPhaseModel(sat_directions[1:],-sat_deltarange, sat_deltaweights)
    baselines, orinit = calc_initial_speed_guess(sat_deltarange,sat_deltaweights,-sat_deltaspeed, 1/sat_deltaspeeduncert, sat_types, sat_directions, gt_np,acsvalues,acstimes,gyrvalues,gyrtimes,utcTimeMillis,baseline_stable.copy())
    
    kml = KMLWriter(image_path+ "kml\\kml_baselines_filtered.kml", "predicted")
    wgs_poses = ecef2geodetic(local_transform_inv(baselines))
    kml.addTrack('baselines_filtered','FFFF0000', wgs_poses)
    kml.finish()

    speeds_init_gt = (gt_np[1:] - gt_np[:-1])
    gt_speed = tf.Variable(speeds_init_gt,trainable=False,dtype=tf.float32)

    #baselines = ndimage.median_filter(baselines,size=(5,1))
    speeds_init = (baselines[1:] - baselines[:-1])
    #speeds_init = speeds_init_gt.copy()
    speeds_init[:,2] = 0
    #speeds_init = ndimage.median_filter(speeds_init,size=(5,1))
    speeds = tf.Variable(speeds_init,trainable=True,dtype=tf.float32)
    
    speeds_np = speeds_init
    or_init = np.zeros((num_epochs-1,3))
    ormed = np.median(orinit)
    orinit -= int((ormed/(2*PI)))*2*PI
    ormed = np.median(orinit)
    orinit -= int((ormed/(PI)))*2*PI

    or_init[:,2] = orinit
    orients = tf.Variable(or_init,trainable=True,dtype=tf.float32)

    true_pos = gt_np
    pred_pos = baselines
    pos_error = np.linalg.norm((true_pos-pred_pos)[:,:2],axis=-1)
    pos_error = np.sort(pos_error)
    pos_error_abs = (pos_error[len(pos_error)//2] + pos_error[len(pos_error)*95//100])/2
    print("Baseline error:", pos_error_abs)
    times_dif = tf.cast(tf.reshape(utcTimeMillis[1:]-utcTimeMillis[:-1],(-1,1)), tf.float32)
    
    
    
    np_firstlast = np.ones(len(times_dif)-1)
    acsp, acsgt, acsr = imu_model.get_acses([speeds, gt_speed,orients,times_dif])
    acsp = acsp.numpy()
    acsp = ndimage.median_filter(acsp,size=(5,1))
    len_frst = 0
    while len_frst < 50 and acsp[len_frst,2] < -3:
        len_frst += 1

    print("First",len_frst," positions not used")
    # np_firstlast[:len_frst+10]  = 0
    # np_firstlast[-5:] = 0
    firstlast_epoch = tf.Variable(np_firstlast,trainable=False,dtype=tf.float32)
    baseline_error_scale = tf.Variable(1e-2,trainable=False,dtype=tf.float32)
    accel_error_scale = tf.Variable(1.,trainable=False,dtype=tf.float32)
    dopler_error_scale = tf.Variable(1.,trainable=False,dtype=tf.float32)
    #@tf.autograph.experimental.do_not_convert
    def minimize_speederror(useimuloss, trainorient, trainspeed, rotate_gyro, optimizer):
        @tf.custom_gradient
        def norm(x):
            y = tf.linalg.norm(x, 'euclidean', -1)
            def grad(dy):
                return tf.expand_dims(dy,-1) * (x / (tf.expand_dims(y + 1e-19,-1)))

            return y, grad
        for _ in range(4):
            with tf.GradientTape(persistent=True) as tape:
                #speeds_loss_glob = tf.abs(norm(speeds[1:])-norm(speeds[:-1]))
                # speeds_loss_glob += (norm(speeds[1:])*norm(speeds[:-1]) - tf.reduce_sum(speeds[1:]*speeds[:-1], axis = -1))/10
                sp = norm(speeds*1000./times_dif)
                acs = sp[1:]-sp[:-1]
                acs = tf.concat([acs,[0.]], axis = 0)
                speeds_loss_glob = tf.square(acs)#norm(acs)*10+
                large_acs_loss = tf.square(tf.nn.relu(acs-5))#norm(acs)*10+
                # acs = tf.concat([[[0.,0.,0.,]], acs], axis = 0)
                # speeds_loss_glob += tf.reduce_sum(tf.abs(acs[1:] - acs[:-1]), axis = -1)

                speeds_loss_Z = tf.square(speeds[:,2])
                phase_loss = phase_model([speeds,orients])
                dopler_loss = dopler_model([speeds,orients,times_dif])
                acs_loss, acs_grad, quat_loss, speed_loss, speed_grad, g, stable_poses = imu_model([speeds,orients,times_dif,rotate_gyro])
                poses = tf.cumsum(speeds, axis = 0)
                poses = tf.concat([[[0.,0.,0.]],poses], axis = 0)

                fwd = tf.stack([tf.math.cos(orients[:,2]),tf.math.sin(orients[:,2]),tf.zeros_like(orients[:,2])], axis = 1)


                #psevdo_loss = psevdo_model([tf.concat([[[0.,0.,0.]],speeds], axis = 0), poses - baselines, times])
                psevdo_loss, psevdo_grad = psevdo_model([tf.concat([[[0.,0.,0.]],fwd], axis = 0), poses - baseline_stable, times])
                quat_loss = quat_loss*firstlast_epoch
                speed_loss = speed_loss*firstlast_epoch
                acs_loss = acs_loss*firstlast_epoch
                imu_loss = quat_loss + speed_loss + acs_loss
                dir_loss = quat_loss + speed_loss
                #*accel_error_scale
                imu_loss = tf.concat([[0.],imu_loss], axis = 0)
                if not useimuloss:
                    total_loss = phase_loss + dopler_loss + psevdo_loss*1e-6 + speeds_loss_glob/1000 + large_acs_loss
                else:   
                    #total_loss = phase_loss + dopler_loss*dopler_error_scale/100 + psevdo_loss + imu_loss 
                    total_loss = imu_loss + psevdo_loss*1e-6 + phase_loss + dopler_loss + large_acs_loss + speeds_loss_Z/100
                    #total_loss = imu_loss       + speeds_loss_Z    + phase_loss*10 + dopler_loss# + psevdo_loss*400 
                    #total_loss = psevdo_loss
                    # total_loss = imu_loss           + speeds_loss_Z*10 + psevdo_loss*10 + phase_loss + dopler_loss/10 
                # if not useimuloss:
                # else:   
                #     total_loss = imu_loss + phase_loss*10 + psevdo_loss*10 + dopler_loss + speeds_loss_Z*10
                dopler_loss = dopler_loss + psevdo_loss/100# adjast bias?
                phase_loss  = phase_loss  + psevdo_loss/100# adjast bias?
                imu_loss = imu_loss/100
                # total_loss = tf.reduce_mean(total_loss)
                # imu_loss = tf.reduce_mean(imu_loss)
                # stable_poses = tf.concat([[0],stable_poses], axis=0)
                # stable_loss = tf.reduce_mean(tf.abs(speeds*tf.reshape(stable_poses,(-1,1))))
                # total_loss += stable_loss*100

            # stable_poses = tf.concat([[0],stable_poses], axis=0)
            # speeds.assign(speeds*(1-tf.reshape(stable_poses,(-1,1))))      

            
            gradients = {
            }

            def norm_grad(gr):
                gr_scale = tf.linalg.norm(gr, axis = 0, keepdims=True) + 1e-3
                return gr/tf.maximum(gr_scale,1)

            gradients["phase"] = tape.gradient(phase_loss, [speeds])[0]/100.
            gradients["dopler"] = tape.gradient(dopler_loss, [speeds])[0]/100.
            gradients["speed_grad"]   = speed_grad
            gradients["acs_grad"] = acs_grad
            gradients["physics"] = tape.gradient(speeds_loss_glob, [speeds])[0]
            gradients["psevdo"] = psevdo_grad
            #gradients["psevdo"] = gradients["psevdo"][1:] - gradients["psevdo"][:-1]
            #gradients["acs_grad"]   = norm_grad(gradients["acs_grad"]) 
            #gradients["speed_grad"] = norm_grad(gradients["speed_grad"])
            gradients["physics"] = norm_grad(gradients["physics"])
            #gradients["psevdo"]   = norm_grad(gradients["psevdo"])
            #gradients["phase"] = norm_grad(gradients["phase"])*10
            #gradients["dopler"]   = norm_grad(gradients["dopler"])*10

            # grad = gradients["phase"]/100 + gradients["dopler"]/1000 + gradients["acs"]/1000 + gradients["speed"]/1000
            # #- gradients["phase"]/10000
            # grad = grad/(1e-4+tf.linalg.norm(grad, axis=-1, keepdims=True))/100
            # speeds.assign_sub(grad)
            # def aplly_grad_byerror(gr):
            #     speeds_error = gt_speed - speeds
            #     kf = 1./(1e-4+tf.square(tf.linalg.norm(gr,axis=0,keepdims=True)))
            #     speeds.assign_sub(kf*gr/1000)
            # aplly_grad_byerror(gradients["phase"])
            # aplly_grad_byerror(gradients["dopler"])
            # aplly_grad_byerror(gradients["acs"])
            # aplly_grad_byerror(gradients["speed"])

            grads = tape.gradient(psevdo_loss, psevdo_model.trainable_weights)
            optimizer.apply_gradients(zip(grads, psevdo_model.trainable_weights))        
            
            #psevdo_grad = psevdo_model.shift_pp[1:]-psevdo_model.shift_pp[:-1]
            #gradients['psevdo'] = psevdo_grad

            if not useimuloss:
                optimizer.apply_gradients(zip([gradients["phase"]], [speeds]))   
                optimizer.apply_gradients(zip([gradients["dopler"]/10], [speeds]))   
                optimizer.apply_gradients(zip([gradients["physics"]/100], [speeds]))   
                optimizer.apply_gradients(zip([gradients["psevdo"]/100], [speeds]))   
            else:
                optimizer.apply_gradients(zip([gradients["phase"]], [speeds]))   
                optimizer.apply_gradients(zip([gradients["dopler"]/10], [speeds]))   
                optimizer.apply_gradients(zip([gradients["speed_grad"]/20], [speeds]))   
                optimizer.apply_gradients(zip([gradients["acs_grad"]/20], [speeds]))   
                optimizer.apply_gradients(zip([gradients["psevdo"]/100], [speeds]))   
                #speeds.assign_sub(gradients["speed_grad"]/300)
                #speeds.assign_sub(gradients["acs_grad"]/300)

                #speeds.assign_sub(tf.sign(speeds)*1e-6)
                #optimizer.apply_gradients(zip([-psevdo_grad/100], [speeds]))   
                #psevdo_model.bias_pp.assign(tf.zeros_like(psevdo_model.bias_pp))
            
            # stable_poses = tf.concat([[0],stable_poses], axis=0)
            # speeds.assign(speeds*(1-tf.reshape(stable_poses,(-1,1))))      
            
            if trainorient:
                grads = tape.gradient(imu_loss, [orients])
                optimizer.apply_gradients(zip(grads, [orients]))        

            grads = tape.gradient(phase_loss, phase_model.trainable_weights)
            optimizer.apply_gradients(zip(grads, phase_model.trainable_weights))        
            grads = tape.gradient(dopler_loss, dopler_model.trainable_weights)
            optimizer.apply_gradients(zip(grads, dopler_model.trainable_weights))        
            grads = tape.gradient(imu_loss, imu_model.trainable_weights)
            optimizer.apply_gradients(zip(grads, imu_model.trainable_weights))  

            del tape


        return  total_loss, phase_loss, dopler_loss, acs_loss, quat_loss, speed_loss, psevdo_loss, speeds_loss_Z, g, psevdo_model.get_poses([tf.concat([[fwd[0]],fwd],axis=0),poses, times]), stable_poses, gradients


    def to_float(x):
        return tf.reduce_mean(x).numpy()
    

    trainspeed = True
    use_imu_loss = True
    if np.sum(utcTimeMillis[1:]-utcTimeMillis[:-1]>5000) > 0:
        use_imu_loss = False
    if 'Samsu' in trip_id:
        print("Configuring samsung")
        accel_error_scale.assign(1e-2)
        baseline_error_scale.assign(0)
        dopler_error_scale.assign(100)

    for step in range(0, 32):
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
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)#, epsilon= 0.0001)
            #optimizer = tf.keras.optimizers.SGD(learning_rate=lr)#, momentum=0.9)
            for i in range(1):
                total_loss, phase_loss, dopler_loss, acs_loss, quat_loss, speed_loss,psevdo_loss, speeds_loss_glob, g, poses, stable_poses, gradients  = minimize_speederror(False, True, False, False, optimizer)

                speeds_init_pred = speeds_np = speeds.numpy()
                speeds_e = np.linalg.norm((speeds_init_pred-speeds_init_gt)[:,:2],axis=-1)
                
                true_pos = gt_np
                pred_pos = poses.numpy()#+np.array([[-1.5,+0.5,0]])
                pred_pos = pred_pos + np.median(true_pos-pred_pos, axis = 0)
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
                    np.mean(speeds_e),
                    )
                    )#, end='\r')

            lr = 1e-3
            #optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)#, epsilon= 0.0001)
            #optimizer = tf.keras.optimizers.SGD(learning_rate=lr)#, momentum=0.9)
            train_step = tf.function(minimize_speederror).get_concrete_function(False, True, trainspeed, True, optimizer)
        elif step == 2:
            lr = 1e-3
        elif step == 6:
            lr = 1e-3
        elif step == 12:
            train_step = tf.function(minimize_speederror).get_concrete_function(True, True, trainspeed, True, optimizer)
        elif step == 24:
             lr = 1e-4
        # elif step == 4:
        #     lr = 1e-2
        # elif step == 6:
        #     total_loss, phase_loss, dopler_loss, acs_loss, quat_loss, speed_loss,psevdo_loss, speeds_loss_glob, g, poses, stable_poses, gradients  = minimize_speederror(False, True, trainspeed, True, optimizer)
        #     lr = 1e-2
        #     train_step = tf.function(minimize_speederror).get_concrete_function(True, True, trainspeed, True, optimizer)
        #     #train_step = tf.function(minimize_speederror).get_concrete_function(False, True, trainspeed, True, optimizer)
        #     # optimizer = tf.keras.optimizers.Adam(learning_rate=lr)#, epsilon= 0.0001)
        #     # train_step = tf.function(minimize_speederror).get_concrete_function(False, True, trainspeed, True, optimizer)
        # # elif step == 10:
        # #     lr = 1e-3
        # # elif step == 12:
        # #     if 'Samsu' in trip_id:
        # #         baseline_error_scale.assign(1e-3)
        # #     lr = 1e-3
        #     # optimizer = tf.keras.optimizers.Adam(learning_rate=lr)#, epsilon= 0.0001)
        #     # train_step = tf.function(minimize_speederror).get_concrete_function(use_imu_loss    , True, trainspeed, True, optimizer)
        # elif step == 32:
        #     lr = 1e-3

        if step == 12:
            wgs_poses = ecef2geodetic(local_transform_inv(poses))
            with open(f'{GOOGLE_DATA_ROOT}{trip_id}'+'/pre_result0.pkl', 'wb') as f:
                pickle.dump(wgs_poses, f, pickle.HIGHEST_PROTOCOL)

        if step != 0:
            for i in range(128):
                total_loss, phase_loss, dopler_loss, acs_loss, quat_loss, speed_loss,psevdo_loss, speeds_loss_glob, g, poses, stable_poses, gradients  = train_step()

        speeds_init_pred = speeds_np = speeds.numpy()
        true_pos = gt_np
        speeds_e = np.mean(np.linalg.norm((speeds_init_pred-speeds_init_gt)[:,:2],axis=-1))
    
    

        poses = poses.numpy()
        
        #poses = np.cumsum(speeds_np, axis = 0)
        #poses = np.concatenate([[[0.,0.,0.]],poses], axis = 0)
        #poses += psevdo_model.shift0.numpy()

        pred_pos = poses# + np.array([[-1.5,+0.5,0]])
        #pred_pos = pred_pos + np.median(true_pos-pred_pos, axis = 0)
        pos_error = np.linalg.norm((true_pos-pred_pos)[:,:2],axis=-1)
        pos_error = np.sort(pos_error)
        pos_error_abs2 = (pos_error[len(pos_error)//2] + pos_error[len(pos_error)*95//100])/2

        shift_median = np.median(true_pos-pred_pos, axis = 0)
        pred_pos2 = pred_pos + shift_median
        pos_error = np.linalg.norm((true_pos-pred_pos2)[:,:2],axis=-1)
        pos_error = np.sort(pos_error)
        pos_error_abs = (pos_error[len(pos_error)//2] + pos_error[len(pos_error)*95//100])/2

        print( "Training loss at step %d: %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f  lr %.4f er %.4f(%.4f) ser %.4f" % (step, 
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
            pos_error_abs2,
            speeds_e,
            )
            ,tf.reduce_sum(stable_poses).numpy()
            ,shift_median
            )#, end='\r')
        # print(imu_model.conv_filters_gyro.numpy().reshape((-1))/np.sum(imu_model.conv_filters_gyro.numpy()))
        # print(imu_model.conv_filters_acel.numpy().reshape((-1))/np.sum(imu_model.conv_filters_acel.numpy()))
        # if pos_error_abs < 0.8:
        #     break
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


            tr_quat, pred_quat = imu_model.get_angles([speeds, orients])
            pred_quat,tr_quat = pred_quat.numpy()*10, tr_quat.numpy()*10
            gyr_scale = 1
            if step > 8:
                gyr_scale = 100
            plt.clf()
            plt.plot( np.arange(len(tr_quat)), tr_quat[:,0]*10)
            plt.plot( np.arange(len(tr_quat)), tr_quat[:,1]*10+6)
            plt.plot( np.arange(len(tr_quat)), tr_quat[:,2]+12)
            plt.plot( np.arange(len(pred_quat)), pred_quat[:,0]*10+3)
            plt.plot( np.arange(len(pred_quat)), pred_quat[:,1]*10+9)
            plt.plot( np.arange(len(pred_quat)), pred_quat[:,2]+15)
            plt.plot( np.arange(len(pred_quat)), gyr_scale*(pred_quat[:,0] - tr_quat[:,0]) + 18)
            plt.plot( np.arange(len(pred_quat)), gyr_scale*(pred_quat[:,1] - tr_quat[:,1]) + 21)
            plt.plot( np.arange(len(pred_quat)), gyr_scale*(pred_quat[:,2] - tr_quat[:,2]) + 24)

            plt.legend(['tx','ty','tz','px', 'py', 'pz','dx','dy', 'dz'])
            save_fig(image_path+'gyr\\gyro_'+str(step).zfill(3)+'.png')

            total_loss = conv_numpy(total_loss)
            # plt.plot( np.arange(len(speeds_init_gt)), speeds_init_pred[:,0] - speeds_init_gt[:,0])
            # plt.plot( np.arange(len(speeds_init_gt)), speeds_init_pred[:,1] - speeds_init_gt[:,1])
            timedif = utcTimeMillis[1:]-utcTimeMillis[:-1]

            quats_np  = orients.numpy()
            ga = -quats_np[:,2]
            ga = (ga[1:]+ga[:-1])/2

            def rotate(v, ang):
                if len(ang) < len(v):
                    v = v[:len(ang)]
                    ga = ang
                else:
                    ga = ang[:len(v)]
                return np.stack([v[:,0]*np.cos(ga) - v[:,1]*np.sin(ga), v[:,1]*np.cos(ga) + v[:,0]*np.sin(ga), v[:,2] ], axis = -1)
            def rotate_all(vals, ang):
                res = []
                for v in vals:
                    res.append(rotate(v,ang))
                return res
            # speed_loss = conv_numpy(speed_loss)
            acsp, acsgt, acsr = imu_model.get_acses([speeds, gt_speed,orients,times_dif])
            acsp, acsgt, acsr = acsp.numpy(), acsgt.numpy(), acsr.numpy()
            acsp_rot, acsgt_rot, acsr_rot = rotate_all([acsp, acsgt, acsr], ga)

            plt.clf()
            plt.ylim([-2,24])
            plt.plot( np.arange(len(acsp_rot)), acsp_rot[:,0])
            plt.plot( np.arange(len(acsp_rot)), acsp_rot[:,1]+6)
            plt.plot( np.arange(len(acsp)), acsp[:,2]+12)
            plt.plot( np.arange(len(acsr_rot)), acsr_rot[:,0]+2)
            plt.plot( np.arange(len(acsr_rot)), acsr_rot[:,1]+8)
            plt.plot( np.arange(len(acsr)), acsr[:,2]+14)
            plt.plot( np.arange(len(acsgt_rot)), acsgt_rot[:,0]+4)
            plt.plot( np.arange(len(acsgt_rot)), acsgt_rot[:,1]+10)
            plt.plot( np.arange(len(acsgt)), acsgt[:,2]+16)
            plt.plot( np.arange(len(acsr_rot)), acsr_rot[:,0]-acsp_rot[:,0]+18)
            plt.plot( np.arange(len(acsr_rot)), acsr_rot[:,1]-acsp_rot[:,1]+20)
            plt.plot( np.arange(len(acsr)), acsr[:,2]-acsp[:,2]+22)

            plt.legend(['px', 'py', 'pz','tx','ty','tz','gtx','gty','gtz','dx','dy', 'dz'])
            save_fig(image_path+'acc\\accel_'+str(step).zfill(3)+'.png')
            ga = -quats_np[:,2]

            speeds_init_gt_rot, speeds_init_pred_rot = rotate_all([speeds_init_gt, speeds_init_pred], ga)

            plt.clf()
            plt.plot( np.arange(len(speeds_init_gt)), speeds_init_gt_rot/10)
            plt.plot( np.arange(len(speeds_init_gt)), speeds_init_pred_rot/10)
            plt.plot( np.arange(len(speeds_init_gt)), (speeds_init_gt_rot - speeds_init_pred_rot)*5 + [2,14])
            
            # plt.plot( np.arange(len(psevdo_loss)), psevdo_loss/100)
            
            # plt.plot( np.arange(len(quat_loss)), quat_loss)
            # plt.plot( np.arange(len(speed_loss)), speed_loss)


            axes = ['_x','_y']
            for i in range(2):
                leg = ['t', 'p', 'd']
                plt.clf()
                plt.plot( np.arange(len(speeds_init_gt)), speeds_init_gt_rot[:,i]/30)
                plt.plot( np.arange(len(speeds_init_gt)), speeds_init_pred_rot[:,i]/30)
                plt.plot( np.arange(len(speeds_init_gt)), (speeds_init_gt_rot - speeds_init_pred_rot)[:,i]*5 + 2)
                curs = 2
                for k,v in gradients.items():
                    leg.append(k+axes[i])
                    if v is None:
                        plt.plot( np.arange(1000), np.zeros((1000) + curs*2))
                    else:
                        v = v.numpy()[:,:2]
                        v_rot = rotate(v, ga)[:,:2]
                        plt.plot( np.arange(len(v)), v_rot[:,i]*5 + curs*2)
                    curs += 1
                plt.legend(leg)
                plt.grid()
                save_fig(image_path+'spd\\speed_diff'+axes[i]+str(step).zfill(3)+'.png')

            ga = np.append(ga,ga[-1])
            # plt.plot( np.arange(len(speeds_init_gt)), speeds_init_gt[:,0])
            # plt.plot( np.arange(len(speeds_init_gt)), speeds_init_gt[:,1])
            # plt.plot( np.arange(len(speeds_init_gt)), speeds_init_pred[:,0])
            # plt.plot( np.arange(len(speeds_init_gt)), speeds_init_pred[:,1])
            # plt.legend(['true x', 'true y', 'pred x', 'pred y'])
            valid_weights = np.sum(sat_deltaweights>0, axis = -1).astype(np.float32)
            valid_weights[valid_weights>=6] = NaN
            valid_weights[valid_weights<6] = 0


            error_sh = ndimage.median_filter(pred_pos - baselines, (100,1))
            error_sh = pred_pos-true_pos
            error_sh_rot = error_sh
            error_sh_rot = rotate(error_sh, ga)[:,:2]

            plt.clf()
            plt.plot( np.arange(len(true_pos)), error_sh_rot[:,0])
            plt.plot( np.arange(len(true_pos)), error_sh_rot[:,1])
            plt.plot( np.arange(len(pred_pos)-1), (pred_pos[1:,2] - pred_pos[:-1,2])/10)
            psevdo_grad = gradients["psevdo"].numpy()
            psevdo_grad = np.cumsum(psevdo_grad)
            psevdo_grad_rot = rotate(psevdo_grad,ga)[:,:2]
            # bias_np = -psevdo_model.shift_pp.numpy()[:,:2]
            # plt.plot( np.arange(len(bias_np)), bias_np)

            plt.plot( np.arange(len(valid_weights)), valid_weights, marker='o')
            sat_deltaspeeduncertcount = np.sum(sat_deltaspeeduncert<1, axis = -1).astype(np.float32)
            sat_deltaspeeduncertcount[sat_deltaspeeduncertcount>=15] = NaN
            sat_deltaspeeduncertcount[sat_deltaspeeduncertcount< 15] = 0
            plt.plot( np.arange(len(sat_deltaspeeduncertcount)), sat_deltaspeeduncertcount, marker='o', linestyle = None)
            sat_deltaspeeduncertcount = np.sum(sat_deltaspeeduncert<1, axis = -1).astype(np.float32)
            sat_deltaspeeduncertcount[sat_deltaspeeduncertcount>=10] = NaN
            sat_deltaspeeduncertcount[sat_deltaspeeduncertcount< 10] = 0
            plt.plot( np.arange(len(sat_deltaspeeduncertcount)), sat_deltaspeeduncertcount, marker='o', linestyle = None)
            
            plt.legend(['dif x', 'dif y', 'speed z','deltas', 'speeds15', 'speeds10'])
            save_fig(image_path+'sft\\track_shift'+str(step).zfill(3)+'.png')

            plt.clf()
            speeds_np = speeds.numpy()
            speeds_np = speeds_np/(np.linalg.norm(speeds_np, axis=-1,keepdims=True) + 0.5)
            speeds_np1 = gt_np[1:]-gt_np[:-1]
            speeds_np1 = speeds_np1/(np.linalg.norm(speeds_np1, axis=-1,keepdims=True) + 0.5)
            
            
            plt.clf()
            plt.plot( np.arange(len(quats_np)), speeds_np[:,0])
            plt.plot( np.arange(len(quats_np)), speeds_np[:,1])
            plt.plot( np.arange(len(speeds_np1)), speeds_np1[:,0])
            plt.plot( np.arange(len(speeds_np1)), speeds_np1[:,1])
            plt.plot( np.arange(len(quats_np)), quats_np[:,0]*10)
            plt.plot( np.arange(len(quats_np)), quats_np[:,1]*10)
            plt.plot( np.arange(len(quats_np)), quats_np[:,2]/PI)
            plt.legend(['bx', 'by','tx', 'ty','q x', 'q y','q z'])
            save_fig(image_path+'qtr\\quat'+str(step).zfill(3)+'.png')
            plt.clf()
            def plotloss(l):
                if not isinstance(l,(list,np.ndarray)):
                    l = l.numpy()
                plt.plot( np.arange(len(l)), l)
            plotloss(total_loss)
            plotloss(phase_loss)
            plotloss(dopler_loss)
            plotloss(acs_loss*100)
            plotloss(quat_loss*100)
            plotloss(speed_loss*100)
            plotloss(psevdo_loss)
            #plotloss(speeds_loss_Z)
            plt.legend(['total', 'phase','dopler', 'acs','quat', 'speed','psevdo'])
            save_fig(image_path+'lss\\loss'+str(step).zfill(3)+'.png')

            plt.clf()
            ta = np.arctan2(-speeds_init_gt[:,0], speeds_init_gt[:,1])
            pa = np.arctan2(-speeds_init_pred[:,0], speeds_init_pred[:,1])
            ta[np.linalg.norm(speeds_init_gt, axis=-1)<0.1] = NaN
            pa[np.linalg.norm(speeds_init_pred, axis=-1)<0.1] = NaN
            ga = quats_np[:,2]

            ta = ta - ((ta-ga)/(2*PI)).astype(np.int32)*2*PI
            ta = ta - ((ta-ga)/(PI)).astype(np.int32)*2*PI
            pa = pa - ((pa-ga)/(2*PI)).astype(np.int32)*2*PI
            pa = pa - ((pa-ga)/(PI)).astype(np.int32)*2*PI

            plt.plot( np.arange(len(pa)), pa)
            plt.plot( np.arange(len(ta)), ta)
            plt.plot( np.arange(len(ga)), ga)
            plt.legend(['pa', 'ta', 'ga'])
            save_fig(image_path+'crs\\crs_'+str(step).zfill(3)+'.png')
            
            kml = KMLWriter(image_path+ "kml\\0_"+str(step).zfill(3)+".kml", "predicted")
            wgs_poses = ecef2geodetic(local_transform_inv(pred_pos+np.array([[-1.5,+0.5,0]])))
            kml.addTrack('predicted_'+str(step),'FFFF0000', wgs_poses, False)
            wgs_poses = ecef2geodetic(local_transform_inv(gt_np+np.array([[-1.5,+0.5,0]])))
            kml.addTrack('gt_'+str(step),'FF00FF00', wgs_poses, False)
            sh = np.linalg.norm((gt_np-pred_pos)[:,:2], axis=-1)
            indexes = []
            for i in range(10):
                ind = np.argmax(sh)
                sh[max(ind-50,0):min(ind+50,len(sh))] = 0
                indexes.append(ind)
            kml.addPoints(ecef2geodetic(local_transform_inv(gt_np[indexes])))
            kml.addPoints(ecef2geodetic(local_transform_inv(pred_pos[indexes])))
            kml.finish()

            kml = KMLWriter(image_path+ "kml\\kml_baseline_stable.kml", "Baseline")
            wgs_poses = ecef2geodetic(local_transform_inv(baseline_stable))
            kml.addTrack('baseline','FF0000FF', wgs_poses)
            kml.finish()

    baseline_error_scale.assign(10.)
    wgs_poses = ecef2geodetic(local_transform_inv(poses))
    
    fname = f'{GOOGLE_DATA_ROOT}{trip_id}'+'/result0.pkl'
    if os.path.exists(fname):
        for i in range(99):
            if not os.path.exists(fname+'.'+str(i).zfill(2)):
                os.rename(fname,fname+'.'+str(i).zfill(2))
                break
    with open(fname, 'wb') as f:
        pickle.dump(wgs_poses, f, pickle.HIGHEST_PROTOCOL)

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