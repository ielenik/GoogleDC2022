from src.laika.lib.coordinates import geodetic2ecef
from src.laika.astro_dog import AstroDog
from .utils.loader import myLoadRinexIndexed, myLoadRinex
from .utils.gnss_log_reader import gnss_log_to_dataframes
from .utils.gnss_log_processor import process_gnss_log
from .utils.coords_tools import calc_pos_fix
from .analyze.analyze_basestation import doAnylizeBasestation
from .analyze.analyze_satbygtruth import doAnylizeSatByGroundTruth
from src.laika.lib.coordinates import ecef2geodetic

import numpy as np
import pandas as pd
import pickle
import os 
from tqdm import tqdm
from .utils.rinex_log_reader import load_rinex
from .utils.slac import loadSlac
from .laika.downloader import download_cors_station


from .tf.tf_train_loop import get_track
import datetime
from .analyze.analize_rigidtrans import analyze as analyzeRigid
from .laika.gps_time import GPSTime

# path to the folder with individual phone logs
def analize_sat_data(path_folder):

    trackname = path_folder.split('/')[-1]
    date_str = trackname.split('-')

    datetimenow = datetime.datetime.strptime('/'.join(date_str[:3]), '%Y/%m/%d')
    datetimenow = GPSTime.from_datetime(datetimenow)

    path_basestation = download_cors_station(datetimenow,'slac', "D:/databases/slac")
    #path_basestation = loadSlac(datetimenow)


    constellations = ['GPS', 'GLONASS', 'BEIDOU','GALILEO']
    dog = AstroDog(valid_const=constellations, pull_orbit=True)
    
    
    basestation, coords, satregistry = myLoadRinexIndexed(path_basestation)
    basestation = { 
        'times': np.array([r[0] for r in basestation]), 
        'values':np.array([r[1] for r in basestation]),
        'coords':coords,
        'sat_registry':satregistry,
        }

    #doAnylizeBasestation(basestation, dog)
    mat_local = np.zeros((3,3))
    mat_local[2] = coords/np.linalg.norm(coords, axis = -1)
    mat_local[1] = np.array([0,0,1])
    mat_local[1] = mat_local[1] - mat_local[2]*np.sum(mat_local[2]*mat_local[1])
    mat_local[1] = mat_local[1]/np.linalg.norm(mat_local[1], axis = -1)
    mat_local[0] = np.cross(mat_local[1], mat_local[2])
    mat_local = np.transpose(mat_local)

    locl_trans = mat_local
    locl_shift = np.array([0,0,np.linalg.norm(coords, axis = -1)])
    def local_transform(v):
        return np.matmul(v,locl_trans) - locl_shift
    def local_transform_inv(v):
        return np.matmul(v+locl_shift,locl_trans.T) 

    ground_truth = {}
    if os.path.isfile(path_folder + '/groundtruth.csv'):
        with open(path_folder + '/groundtruth.csv', 'rt') as f:
            lines = f.readlines()[1:]
            values = []
            times = []
            for l in lines:
                v = l.split(';')
                time = float(v[0])
                time = time%(24*3600)
                times.append(time)
                values.append(np.array(geodetic2ecef([float(v[1]),float(v[2]),float(v[3])])))
            ground_truth['times'] = np.array(times)
            ground_truth['values'] = local_transform(np.array(values))

    phone_glob = next(os.walk(path_folder))[1]
    print(path_folder, end=' ')

    phones = {}
    phone_names = []
    measures = []
    for phone in phone_glob:

        if phone.endswith('.skp'):
            continue

        if phone.endswith('cors_obs'):
            continue

        phones[phone] = len(phones)
        phone_names.append(phone)
        print(phone, end=' ')

        try:
            with open(path_folder+'/' + phone + '/' + phone +'.cache.pkl', 'rb') as f:
                cur_measure =  pickle.load(f)
        except:
            try:
                gnss_log = gnss_log_to_dataframes(path_folder+'/' + phone + '/' + phone +'_GnssLog.txt')
                cur_measure = process_gnss_log(gnss_log['Raw'], basestation, dog)
                cur_measure['acs'] = gnss_log['UncalAccel']
                cur_measure['gir'] = gnss_log['UncalGyro']
                cur_measure['mag'] = gnss_log['UncalMag']
                cur_measure['orientation'] = gnss_log['OrientationDeg']
            except:
                cur_measure = load_rinex(path_folder+'/' + phone + '/rover.obs', basestation, dog)
                cur_measure['acs'] = []
                cur_measure['gir'] = []
                cur_measure['mag'] = []
                cur_measure['orientation'] = []

            cur_measure['phone'] = phone
            num_epoch = len(cur_measure['sat_psevdovalid'])
            baseline_poses = []
            x0=[0, 0, 0, 0,0,0,0,0,0]
            for i in tqdm(range(num_epoch)):
                x0, err = calc_pos_fix(cur_measure['sat_positions'][i], cur_measure['sat_psevdodist'][i], cur_measure['sat_psevdoweights'][i],cur_measure['sat_psevdovalid'][i] > 0, cur_measure['sat_types'], x0)
                if np.sum(err < 1000) > 10:
                    break #initial guess
            for i in tqdm(range(num_epoch)):
                #if i == 72:
                #    breakpoint = 1
                x0, _ = calc_pos_fix(cur_measure['sat_positions'][i], cur_measure['sat_psevdodist'][i], cur_measure['sat_psevdoweights'][i],cur_measure['sat_psevdovalid'][i] > 0, cur_measure['sat_types'], x0)
                baseline_poses.append(local_transform(x0[:3]))

            cur_measure['sat_positions'] = local_transform(cur_measure['sat_positions'])
            cur_measure['baseline'] = np.array(baseline_poses)
            cur_measure['groundtrue'] = {}
            gt_path = path_folder +'/' + phone + '/ground_truth.csv'
            if os.path.exists(gt_path):
                truepos = pd.read_csv(gt_path)
                truepos.set_index('millisSinceGpsEpoch', inplace = True)
                baseline_times = []
                gt_ecef_coords = []
                for timemili, row in truepos.iterrows():
                    baseline_times.append(timemili)
                    latbl, lonbl, altbl = float(row['latDeg']),float(row['lngDeg']),float(row['heightAboveWgs84EllipsoidM'])
                    gt_ecef_coords.append(geodetic2ecef([latbl,lonbl,altbl]))
                    cur_measure['groundtrue']['times'] = np.array(baseline_times)
                    cur_measure['groundtrue']['values'] = local_transform(np.array(gt_ecef_coords))

            with open(path_folder+'/' + phone + '/' + phone +'.cache.pkl', 'wb') as f:
                pickle.dump(cur_measure, f, pickle.HIGHEST_PROTOCOL)

        if 'groundtrue' in cur_measure:
            ground_truth['times'] = cur_measure['groundtrue']['times']
            ground_truth['values'] = cur_measure['groundtrue']['values']

        if len(ground_truth) != 0:
            analyzeRigid(cur_measure, ground_truth, path_folder)
            #doAnylizeSatByGroundTruth(cur_measure, ground_truth)

        measures.append(cur_measure)
    def wgs_converter(pos):
        return ecef2geodetic(local_transform_inv(pos))


    #times, ecf_local = get_track(measures, path_folder, wgs_converter)
    #return times, local_transform_inv(ecf_local)