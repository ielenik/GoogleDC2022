import numpy as np
from tqdm import tqdm
from src.laika.gps_time import GPSTime
from matplotlib import pyplot
import math
from .predict_distance import predict_distance


def doAnylizeSatByGroundTruth(cur_measure, ground_truth):
    NaN = float("NaN")
    
    ground_times = ground_truth['times']
    ground_values = ground_truth['values']

    num_epoch = len(cur_measure['epoch_times'])
    num_sat = cur_measure['sat_psevdodist'].shape[1]

    psevdo = np.ones((num_sat,num_epoch))*NaN
    delta = np.ones((num_sat,num_epoch))*NaN
    truth = np.ones((num_sat,num_epoch))*NaN

    ecef0 = ground_values[0]
    for i in tqdm(range(num_epoch)):
        tmd = cur_measure['epoch_times'][i]
        tmd = tmd/1000000
        indx = np.searchsorted(ground_times, tmd)
        if indx == 0 or indx == len(ground_times): # ground true contains full time or just time of the day
            tmd = tmd/1000
            tmd = tmd % (3600*24)
            indx = np.searchsorted(ground_times, tmd)
            if indx == 0 or indx == len(ground_times):
                continue


        ecef = ground_values[indx]
        #x0, _ = calc_pos_fix(cur_measure['sat_positions'][i], cur_measure['sat_psevdodist'][i], cur_measure['sat_psevdoweights'][i],cur_measure['sat_psevdovalid'][i] > 0, cur_measure['sat_types'], x0)
        sat_type_bias = [[],[],[],[],[],[],[],[],[],[],[],[],[],]
        for j in range(num_sat):
            if cur_measure['sat_psevdovalid'][i][j] == 0:
                continue
            vect = cur_measure['sat_positions'][i][j] - ecef
            dist_true = np.linalg.norm(vect)
            dist_pred = cur_measure['sat_psevdodist'][i][j]
            dif = dist_pred - dist_true
            sat_type_bias[cur_measure['sat_types'][j]].append(dif)

        for j in range(8):
            if len(sat_type_bias[j]) > 0:
                sat_type_bias[j] = np.median(sat_type_bias[j])
            else:
                sat_type_bias[j] = 0

        for j in range(num_sat):
            vect = cur_measure['sat_positions'][i][j] - ecef
            vect0 = cur_measure['sat_positions'][i][j] - ecef0
            dist_true = np.linalg.norm(vect) - np.linalg.norm(vect0)
            truth[j,i] = dist_true

            if cur_measure['sat_psevdovalid'][i][j] == 0:
                continue

            dist_pred = cur_measure['sat_psevdodist'][i][j] - sat_type_bias[cur_measure['sat_types'][j]] - np.linalg.norm(vect0)
            psevdo[j,i] = dist_pred

        for j in range(num_sat):
            if cur_measure['sat_deltavalid'][i][j] == 0:
                continue
            delta[j,i] = cur_measure['sat_deltarange'][i][j]

    delta = delta[:,1:] - delta[:,:-1]
    psevdo = psevdo[:,1:]
    truth = truth[:,1:]

    predict_distance(psevdo, delta, truth)

    mat = np.ones(cur_measure['sat_psevdodist'].shape)*NaN
    for i in tqdm(range(num_epoch)):
        tmd = cur_measure['epoch_times'][i]
        tmd = tmd/1000000
        indx = np.searchsorted(ground_times, tmd)
        if indx == 0 or indx == len(ground_times): # ground true contains full time or just time of the day
            tmd = tmd/1000
            tmd = tmd % (3600*24)
            indx = np.searchsorted(ground_times, tmd)
            if indx == 0 or indx == len(ground_times):
                continue

        ecef = ground_values[indx]
        #x0, _ = calc_pos_fix(cur_measure['sat_positions'][i], cur_measure['sat_psevdodist'][i], cur_measure['sat_psevdoweights'][i],cur_measure['sat_psevdovalid'][i] > 0, cur_measure['sat_types'], x0)
        sat_type_bias = [[],[],[],[],[],[],[],[],[],[],[],[],[],]
        for j in range(len(cur_measure['sat_psevdovalid'][i])):
            if cur_measure['sat_psevdovalid'][i][j] == 0:
                continue
            vect = cur_measure['sat_positions'][i][j] - ecef
            dist_true = np.linalg.norm(vect)
            dist_pred = cur_measure['sat_psevdodist'][i][j]
            dif = dist_pred - dist_true
            sat_type_bias[cur_measure['sat_types'][j]].append(dif)
        for j in range(8):
            if len(sat_type_bias[j]) > 0:
                sat_type_bias[j] = np.median(sat_type_bias[j])
            else:
                sat_type_bias[j] = 0

        for j in range(len(cur_measure['sat_psevdovalid'][i])):
            if cur_measure['sat_psevdovalid'][i][j] == 0:
                continue
            vect = cur_measure['sat_positions'][i][j] - ecef
            dist_true = np.linalg.norm(vect)
            dist_pred = cur_measure['sat_psevdodist'][i][j]
            dif = dist_pred - dist_true
            dif -= sat_type_bias[cur_measure['sat_types'][j]]
            if abs(dif) > 1000:
                continue
            mat[i,j] = dif #+ j * 10

    times = cur_measure['epoch_times']
    for i in range(mat.shape[1]):
        pyplot.plot( times, mat[:,i])
    pyplot.show()        

    mat = np.ones(cur_measure['sat_deltavalid'].shape)*NaN
    for i in tqdm(range(num_epoch)):
        tmd = cur_measure['epoch_times'][i]
        tmd = tmd/1000000
        indx = np.searchsorted(ground_times, tmd)
        if indx == 0 or indx == len(ground_times): # ground true contains full time or just time of the day
            tmd = tmd/1000
            tmd = tmd % (3600*24)
            indx = np.searchsorted(ground_times, tmd)
            if indx == 0 or indx == len(ground_times):
                continue


        ecef = ground_values[indx]
        #x0, _ = calc_pos_fix(cur_measure['sat_positions'][i], cur_measure['sat_psevdodist'][i], cur_measure['sat_psevdoweights'][i],cur_measure['sat_psevdovalid'][i] > 0, cur_measure['sat_types'], x0)
        for j in range(len(cur_measure['sat_deltavalid'][i])):
            if cur_measure['sat_deltavalid'][i][j] == 0:
                continue
            vect = cur_measure['sat_positions'][i][j] - ecef
            dist_true = np.linalg.norm(vect)
            dist_pred = cur_measure['sat_deltarange'][i][j]
            dif = dist_pred - dist_true
            mat[i,j] = dif + j * 1

    mat = mat[1:,:] - mat[:-1,:]
    mat -= np.nanmedian(mat,axis=-1,keepdims=True)
    sum1 = mat.shape[1] - np.sum(np.isnan(mat), axis = -1, keepdims=True)
    mat[abs(mat) > 0.5] = NaN
    sum2 = mat.shape[1] - np.sum(np.isnan(mat), axis = -1, keepdims=True)
    mat = np.concatenate([mat, sum1, sum2], axis = -1)
    times = cur_measure['epoch_times'][1:]
    for i in range(mat.shape[1]):
        pyplot.plot( times, mat[:,i])
    pyplot.show()        
