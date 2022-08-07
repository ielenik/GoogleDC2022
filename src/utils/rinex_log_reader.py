from .loader import myLoadRinexIndexed
import numpy as np
from .coords_tools import getValuesAtTime
from .coords_tools import rotate_sat
from src.laika.gps_time import GPSTime
from tqdm import tqdm

def load_rinex(path, base, dog):

    '''
        'sat_psevdovalid':sat_psevdovalid,
        'sat_psevdodist':sat_psevdodist,
        'sat_psevdoweights':sat_psevdoweights,

        'sat_deltarange':sat_deltarange,
        'sat_deltavalid':sat_deltavalid,

        'sat_positions':sat_positions,
        'epoch_times':epoch_times,
        'sat_types': np.array(sat_types),
    '''
    LIGHTSPEED = 2.99792458e8
    NaN = float("NaN")

    values, _, satregistry = myLoadRinexIndexed(path)
    
    epoch_times    = np.array([r[0] for r in values])
    sat_psevdodist = [r[1] for r in values]
    sat_deltarange = [r[2] for r in values]
    
    min_epoch = 0#17700
    max_epoch = -1#min_epoch + 500
    epoch_times = epoch_times[min_epoch:max_epoch]
    sat_psevdodist = sat_psevdodist[min_epoch:max_epoch]
    sat_deltarange = sat_deltarange[min_epoch:max_epoch]

    sat_psevdovalid = []
    sat_psevdoweights = []
    sat_deltavalid = []

    num_used_satelites = total_sats = len(satregistry)
    inv_satregistry = {v: k for k, v in satregistry.items()}
    sat_names = []
    sat_types = []
    sig_types = []
    sat_types_reg = {}
    base_local_values = np.ones((len(base['times']),num_used_satelites))*NaN
    sat_positions = np.ones((len(epoch_times),num_used_satelites,3))

    sat_clock_bias = np.zeros((256))

    for i in range(total_sats):
        sat_name = inv_satregistry[i]
        sat_names.append(sat_name[:3])
        sat_type = sat_name[0]+sat_name[-1]
        sig_types.append('C'+sat_name[-1])
        
        if sat_type not in sat_types_reg:
            sat_types_reg[sat_type] = len(sat_types_reg)
        sat_types.append(sat_types_reg[sat_type])
        if sat_name in base['sat_registry'] and base['sat_registry'][sat_name] < 256:
            base_local_values[:,i] = base['values'][:,base['sat_registry'][sat_name]]

    def getCorrections(time_nanos, rsat):
        psevdobase = getValuesAtTime(base['times'], base_local_values, time_nanos)
        dist_to_sat = np.linalg.norm(rsat - base['coords'], axis = -1)
        return dist_to_sat - psevdobase

    for i in tqdm(range(len(epoch_times))):
        epoch_number = i
        sat_psevdodist[i] = np.array(sat_psevdodist[i][:total_sats])
        sat_deltarange[i] = np.array(sat_deltarange[i][:total_sats])
        sat_psevdovalid.append(np.zeros(total_sats))
        sat_deltavalid.append(np.zeros(total_sats))
        sat_psevdoweights.append([1.]*total_sats)


        time_nanos = epoch_times[i]


        #if epoch_number == 800:
        #    deltarange      = 123
        #if epoch_number == 66:
        #    break

        for j in range(num_used_satelites):
            sat_index = j
            sat_name = sat_names[j]
            ReceivedSvTimeNanos = time_nanos - sat_psevdodist[i][j]/LIGHTSPEED  - sat_clock_bias[sat_index]*1000000000
            if np.isnan(ReceivedSvTimeNanos):
                continue
            
            week = int(ReceivedSvTimeNanos/(7*24*60*60*1000000000))
            tow = ReceivedSvTimeNanos/1000000000 - week*7*24*60*60
            timegp = GPSTime(week,tow)
            obj = dog.get_sat_info(sat_name, timegp)
            #obj = (0,0,0),0,0,0 #
            if obj is None:
                continue

            freq = dog.get_frequency(sat_name, timegp, sig_types[j])
            sat_deltarange[i][j] = sat_deltarange[i][j]*(LIGHTSPEED/freq)

            sat_pos, sat_vel, sat_clock_err, sat_clock_drift = obj

            sat_clock_bias[j] = sat_clock_err
            sat_positions[i,j] = sat_pos

        sat_positions[epoch_number] = rotate_sat(sat_positions[epoch_number], sat_psevdodist[epoch_number])
        corr = getCorrections(time_nanos,sat_positions[epoch_number])
        sat_psevdodist[epoch_number] += corr
        sat_psevdovalid[epoch_number][~np.isnan(sat_psevdodist[epoch_number])] = 1.
        sat_deltavalid[epoch_number][~np.isnan(sat_deltarange[epoch_number])] = 1.
        sat_deltavalid[epoch_number][sat_deltarange[epoch_number] == 0] = 0.

    sat_psevdovalid = np.array(sat_psevdovalid)
    sat_psevdodist = np.array(sat_psevdodist)
    sat_psevdoweights =  np.array(sat_psevdoweights)

    sat_deltarange = np.array(sat_deltarange)
    sat_deltavalid = np.array(sat_deltavalid)

    sat_positions = np.array(sat_positions)
    epoch_times = np.array(epoch_times)
    sat_types = np.array(sat_types)
        
    sat_psevdovalid[np.isnan(sat_psevdodist)] = 0
    sat_psevdodist[sat_psevdovalid == 0] = 0
    sat_deltarange[sat_deltavalid == 0] = 0

    return { 
        'sat_psevdovalid':np.array(sat_psevdovalid),
        'sat_psevdodist':np.array(sat_psevdodist),
        'sat_psevdoweights':np.array(sat_psevdoweights),

        'sat_deltarange':np.array(sat_deltarange),
        'sat_deltavalid':np.array(sat_deltavalid),

        'sat_positions':np.array(sat_positions),
        'epoch_times':np.array(epoch_times),
        'sat_types': np.array(sat_types),
    }
