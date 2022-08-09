import numpy as np
import pandas as pd
import glob as gl
import random
import pickle
from os.path import exists
import os
from src.laika.lib.coordinates import ecef2geodetic, geodetic2ecef
from src.anylize_derivative import load_baseline, load_times
from scipy import ndimage, misc
import matplotlib.pyplot as plt
from src.utils.kml_writer import KMLWriter
from sklearn.linear_model import LinearRegression

def interp_3d(times, times_true, poses):
    if len(times) == len(poses):
        return poses
    bs = []
    for i in range(3):
        bs.append(np.interp(times, times_true, poses[:,i]))
    return np.array(bs).T

submission = pd.read_csv(r'D:\databases\smartphone-decimeter-2022\sample_submission.csv')
google = pd.read_csv(r'D:\databases\smartphone-decimeter-2022\google_baseline.csv')
times = submission['UnixTimeMillis'].to_numpy()
tripId = submission['tripId'].to_numpy()[1:]
time_dif = times[1:]-times[:-1]
print(time_dif[np.logical_and(time_dif>3000,time_dif<1000000)  ])
print(tripId[np.logical_and(time_dif>3000,time_dif<1000000)  ])

sh_tresh = 1.3

start_path = "*\\*\\"
paths = gl.glob("D:\\databases\\smartphone-decimeter-2022\\train\\"+start_path)
tracks = {}
phones = {}
err50 = []
err95 = []

ecef_coords = []
ecef_shifts = []
for i, dirname in enumerate(paths):
    drive, phone = dirname.split('\\')[-3:-1]
    result_name = '\\result0.pkl'
    if not exists(dirname+result_name):
        continue
    if phone not in phones:
        phones[phone] = []
    track_type = drive.split('-')[-2]
    if not track_type in tracks:
        tracks[track_type] = []

    with open(dirname+result_name, 'rb') as f:
        poses =  pickle.load(f)
    poses = geodetic2ecef(poses)
    if np.sum(np.isnan(poses)) > 0:
        print("error in poses")
        os.remove(dirname+'\\result0.pkl')
        continue

    gt_raw = pd.read_csv(f'{dirname}/ground_truth.csv')
    gt_np = gt_raw[['LatitudeDegrees','LongitudeDegrees','AltitudeMeters']].to_numpy()
    gt_np[np.isnan(gt_np)] = 0
    gt_np = geodetic2ecef(gt_np)

    if len(gt_np) != len(poses):
        times = gt_raw['UnixTimeMillis'].to_numpy()
        times_true =  load_times(f"\\train\{drive}\\{phone}")
        poses = interp_3d(times,times_true,poses)


    coords_start = np.median(gt_np[0:15], axis = 0)
    
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

    gt_np = local_transform(gt_np)
    poses = local_transform(poses)

    if np.sum(np.isnan(poses)) > 0:
        print(gt_np[:5])
        print("error in poses")
        continue

    shift = gt_np-poses
    shift[:,2] = 0
    if track_type != 'LAX':
        ecef_coords.extend(local_transform_inv(poses))
        ecef_shifts.extend(shift)
    pos_error = np.linalg.norm((gt_np-poses)[:,:2], axis = -1)
    pos_error = np.sort(pos_error)
    pos_error_abs = (pos_error[len(pos_error)//2] + pos_error[len(pos_error)*95//100])/2

    print(i,dirname, pos_error_abs)
    tracks[track_type].append(np.median(gt_np-poses, axis = 0))
    phones[phone].append(np.median(gt_np-poses, axis = 0))

for k,v in tracks.items():
    print(k,np.median(v, axis=0), len(v))
for k,v in phones.items():
    print(k,np.median(v, axis=0), len(v))
model = LinearRegression()
model.fit(ecef_coords,ecef_shifts)
r_sq = model.score(ecef_coords,ecef_shifts)
print(f"coefficient of determination: {r_sq}")

j = 0
err_by_trip = {}
err_by_phone = {}
for i, dirname in enumerate(paths):
    drive, phone = dirname.split('\\')[-3:-1]
    result_name = '\\result0.pkl'
    if not exists(dirname+result_name):
        continue
    j += 1
    track_type = drive.split('-')[-2]
    if not track_type in tracks:
        tracks[track_type] = []

    with open(dirname+result_name, 'rb') as f:
        poses =  pickle.load(f)

    baseline =  load_baseline(f"\\train\{drive}\\{phone}")


    poses = geodetic2ecef(poses)
    if np.sum(np.isnan(poses)) > 0:
        print("error in poses")
        os.remove(dirname+'\\result0.pkl')
        continue
    gt_raw = pd.read_csv(f'{dirname}/ground_truth.csv')
    gt_np = gt_raw[['LatitudeDegrees','LongitudeDegrees','AltitudeMeters']].to_numpy()
    gt_np[np.isnan(gt_np)] = 0
    gt_np = geodetic2ecef(gt_np)

    if len(gt_np) != len(poses) or len(gt_np) != len(baseline):
        times = gt_raw['UnixTimeMillis'].to_numpy()
        times_true =  load_times(f"\\train\{drive}\\{phone}")
        poses = interp_3d(times,times_true,poses)
        baseline = interp_3d(times,times_true,baseline)

    model_shifts = model.predict(poses)

    coords_start = np.median(gt_np[0:15], axis = 0)
    
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

    gt_np = local_transform(gt_np)
    poses = local_transform(poses)
    baseline = local_transform(baseline)
    sh = baseline-poses
    for i in range(1):
        sh = ndimage.uniform_filter(sh,size=(150,1))
    sh2 = np.median(sh, axis = 0)
    sh2[2] = 0
    if np.linalg.norm(sh2) > sh_tresh:
        poses += sh2

    if np.sum(np.isnan(poses)) > 0:
        print(gt_np[:5])
        print("error in poses")
        continue

    pos_error = np.linalg.norm((gt_np-poses-np.mean(tracks[track_type], axis = 0))[:,:2], axis = -1)
    pos_error_save = pos_error
    pos_error = np.sort(pos_error)
    pos_error_abs = (pos_error[len(pos_error)//2] + pos_error[len(pos_error)*95//100])/2
    err50.append(pos_error[len(pos_error)//2])
    err95.append(pos_error[len(pos_error)*95//100])
    sh2 = np.median((gt_np - poses-np.median(tracks[track_type], axis = 0)), axis = 0)
    pos_error = np.linalg.norm((gt_np - poses  - model_shifts)[:,:2], axis = -1)
    pos_error = np.sort(pos_error)
    pos_error_abs2 = (pos_error[len(pos_error)//2] + pos_error[len(pos_error)*95//100])/2
    if track_type != 'LAX':
        err50 = err50[:-1]
        err95 = err95[:-1]
        err50.append(pos_error[len(pos_error)//2])
        err95.append(pos_error[len(pos_error)*95//100])
        pos_error_abs = pos_error_abs2
    if True:#'Sams' in phone:
        print('>>>>', end='')
        if pos_error_abs > 2:
            print(j,dirname, pos_error_abs, pos_error_abs2,'**************', sh2[:2])
        else:
            print(j,dirname, pos_error_abs, pos_error_abs2, sh2[:2])
    if np.linalg.norm(sh2[:2]) > sh_tresh:
        print('***')
        # plt.clf()
        # #plt.plot( np.arange(len(sh)), -np.sort(sh[:,:2], axis = 0))
        # #plt.plot( np.arange(len(sh)), -sh[:,:2])
        # plt.plot( np.arange(len(sh)), (gt_np-poses-np.mean(tracks[track_type], axis = 0))[:,:2])
        # plt.plot( np.arange(len(sh)), (gt_np-baseline-np.mean(tracks[track_type], axis = 0))[:,:2])
        # plt.show()

    if track_type not in err_by_trip:
        err_by_trip[track_type] = []
    err_by_trip[track_type].append(pos_error_abs)
    
    if phone not in err_by_phone:
        err_by_phone[phone] = []
    err_by_phone[phone].append(pos_error_abs)

print("Mean error", (np.mean(np.array(err50))+np.mean(np.array(err95)))/2)
for k,v in err_by_trip.items():
    print(k, np.mean(v), len(v))
for k,v in err_by_phone.items():
    print(k, np.mean(v), len(v))
'''
j = 0
kml = KMLWriter(r'D:\databases\smartphone-decimeter-2022\my_submission.kml', "Submission")
destFile = open(r'D:\databases\smartphone-decimeter-2022\my_submission.csv', 'w')
print('tripId,UnixTimeMillis,LatitudeDegrees,LongitudeDegrees', file = destFile)
paths = gl.glob("D:\\databases\\smartphone-decimeter-2022\\test\\"+start_path)
for i, dirname in enumerate(paths):
    drive, phone = dirname.split('\\')[-3:-1]
    result_name = '\\result0.pkl'
    if phone == 'cors_obs':
        continue
    
    if not exists(dirname+result_name):
        print(drive, phone,result_name,'not exists')
        continue

    tripID = f"{drive}/{phone}"
    times = submission[tripID == submission['tripId']]['UnixTimeMillis'].to_numpy()

    times_true =  load_times(f"\\train\{drive}\\{phone}")
    baseline =  load_baseline(f"\\train\{drive}\\{phone}")

    j += 1
    track_type = drive.split('-')[-2]
    if not track_type in tracks:
        tracks[track_type] = tracks['MTV']

    with open(dirname+result_name, 'rb') as f:
        poses =  pickle.load(f)

    poses = geodetic2ecef(poses)
    if np.sum(np.isnan(poses)) > 0:
        print("error in poses")
        os.remove(dirname+'\\result0.pkl')
        continue
    coords_start = np.median(poses[0:15], axis = 0)
    
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




    poses = local_transform(poses)
    baseline = local_transform(baseline)

    googlepos = google[tripID == submission['tripId']][['LatitudeDegrees','LongitudeDegrees']].to_numpy()
    googlepos = np.concatenate([googlepos,np.zeros((len(googlepos),1)) ], axis = -1)
    googlepos = geodetic2ecef(googlepos)
    googlepos = local_transform(googlepos)


    if len(times) != len(times_true):
        print(tripID, 'aproximating positions')
        bs = []
        for i in range(3):
            bs.append(np.interp(times, times_true, baseline[:,i]))
        baseline = np.array(bs).T
        bs = []
        for i in range(3):
            bs.append(np.interp(times, times_true, poses[:,i]))
        poses = np.array(bs).T

        for i in range(730,790):
            poses[i] = (poses[730]*(times[790]-times[i])+poses[790]*(times[i]-times[730]))/(times[790] - times[730])
        for i in range(2200,2250):
            poses[i] = (poses[2200]*(times[2250]-times[i])+poses[2250]*(times[i]-times[2200]))/(times[2250] - times[2200])

    elif np.sum(np.abs(times_true - times)) > 0:
        print(tripID, 'Non zero time dif', np.sum(np.abs(times_true - times)))


    poses    += np.mean(tracks[track_type], axis = 0)
    baseline += np.mean(tracks[track_type], axis = 0)

    sh2 = googlepos-poses
    for i in range(1):
        #sh2 = ndimage.uniform_filter(sh2,size=(150,1))
        sh2 = ndimage.median_filter(sh2,size=(150,1))
    sh3 = baseline-poses
    for i in range(1):
        #sh2 = ndimage.uniform_filter(sh2,size=(150,1))
        sh3 = ndimage.median_filter(sh3,size=(150,1))
    indexes = []
    sh = np.linalg.norm(sh2[:,:2], axis = -1)
    sh_copy = sh.copy()
    for i in range(10):
        ind = np.argmax(sh)
        sh[max(ind-50,0):min(ind+50,len(sh))] = 0
        indexes.append(ind)
    #suspic = poses_copy[indexes]
    private_trips = [
    '2021-04-28-US-MTV-2/SamsungGalaxyS20Ultra',
    '2021-09-28-US-MTV-1/GooglePixel5',
    '2021-11-05-US-MTV-1/XiaomiMi8',
    '2021-11-30-US-MTV-1/GooglePixel5',
    '2022-01-18-US-SJC-2/GooglePixel5',
    '2022-03-22-US-MTV-1/SamsungGalaxyS20Ultra',
    '2022-03-31-US-LAX-1/GooglePixel5',
    '2022-04-25-US-OAK-1/GooglePixel5',
    ]
    if False and tripID in private_trips:
        plt.clf()
        #plt.plot( np.arange(len(sh)), -np.sort(sh[:,:2], axis = 0))
        #plt.plot( np.arange(len(sh)), -sh[:,:2])
        speed = poses[1:]-poses[:-1]
        acs = speed[1:]-speed[:-1]
        plt.plot( np.arange(len(sh)), sh3[:,:2])
        plt.plot( np.arange(len(sh)), sh2[:,:2])
        plt.show()
        plt.clf()
        plt.plot( np.arange(len(speed)), speed[:,:2]/10)
        plt.show()
        plt.clf()
        plt.plot( np.arange(len(acs)), acs[:,:2])
        plt.show()
    # if tripID == '2022-02-23-US-LAX-3/XiaomiMi8':
    #     poses += [20,20,20]    
        

    poses = local_transform_inv(poses)
    baseline = local_transform_inv(baseline)
    googlepos = local_transform_inv(googlepos)
    if len(times) > len(poses):
        np.append(poses,[poses[-1]],axis=0)
        print("wrong number of poses", tripID, len(times), len(poses))

    kml.addFolder(tripID)    
    kml.addTrack('google','FFFF0000', ecef2geodetic(googlepos), False)
    kml.addTrack('baseline','FF0000FF', ecef2geodetic(baseline), False)
    kml.addTrack('predicted','FF00FF00', ecef2geodetic(poses), False)
    name = None
    if exists(dirname+'result0.pkl.00'):
        name = 'result0.pkl.00'

    if name != None:
        with open(dirname+name, 'rb') as f:
            poses1 =  local_transform(geodetic2ecef(pickle.load(f)))+np.mean(tracks[track_type], axis = 0)
            kml.addTrack('before','FF00AAAA', ecef2geodetic(local_transform_inv(poses1)), False)
        if len(poses1) != len(poses):
            kml.addPoints(ecef2geodetic(poses[indexes]))
            kml.addPoints(ecef2geodetic(baseline[indexes]))
            kml.addPoints(ecef2geodetic(googlepos[indexes]))
        else:
            dr = poses1[1:]-poses1[:-1]
            dr = dr/(np.linalg.norm(dr[:,:2], axis = -1, keepdims=True)+0.1)
            #790-850 2022-04-01-US-LAX-3\XiaomiMi8
            if tripID == '2022-04-01-US-LAX-3/XiaomiMi8':
                print('*'*10)
                print('replacing xiaomi')
                poses[790:850] = local_transform_inv(poses1[790:850]) 
                print('*'*10)
            sh = (poses1-local_transform(poses))[1:,:2]
            sh = np.abs(dr[:,0]*sh[:,1] - dr[:,1]*sh[:,0])
            sh_copy = sh.copy()
            indexes = []
            for i in range(10):
                ind = np.argmax(sh)
                sh[max(ind-50,0):min(ind+50,len(sh))] = 0
                indexes.append(ind)
            kml.addPoints(ecef2geodetic(poses[indexes]))
            kml.addPoints(ecef2geodetic(local_transform_inv(poses1[indexes])))

    else:
        kml.addPoints(ecef2geodetic(poses[indexes]))
        kml.addPoints(ecef2geodetic(baseline[indexes]))
        kml.addPoints(ecef2geodetic(googlepos[indexes]))

    kml.closeFolder()    
    print(tripID, sh_copy[indexes], indexes)

    poses = ecef2geodetic(poses)
    for i in range(len(poses)):
        p = poses[i]
        print(tripID,times[i],p[0],p[1], file = destFile, sep = ',')

destFile.close()
kml.finish()
'''