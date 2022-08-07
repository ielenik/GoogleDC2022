import numpy as np
from tqdm import tqdm
from src.laika.gps_time import GPSTime
from matplotlib import pyplot
import math

def rotate_sat(sat, dist):
    res = np.zeros((3,))
    tm = dist/299792458
    ang = math.pi/(12*60*60)*tm
    res[2] = sat[2]
    res[0] = np.cos(ang)*sat[0]+np.sin(ang)*sat[1]
    res[1] = -np.sin(ang)*sat[0]+np.cos(ang)*sat[1]
    return res


'''    basestation = { 
        'times': np.array([r[0] for r in basestation]), 
        'values':np.array([r[1] for r in basestation]),
        'coords':coords,
        'sat_registry':satregistry,
        }
        '''
def doAnylizeBasestation(basestation, dog):
    times = basestation['times']
    values = basestation['values']    
    ecef_coords = basestation['coords']    
    sat_registry = basestation['sat_registry']    
    NaN = float("NaN")
    LIGHTSPEED = 299792458

    sat_names = []
    for i in range(len(sat_registry)):
        sat_names.append("")
    for k in sat_registry:
        sat_names[sat_registry[k]] = k.split('_')[0]



    mat = np.ones(values.shape)*NaN
    sat_clock_error = np.ones((256,))*NaN
    for i in tqdm(range(len(times))):
        t = times[i]
        v = values[i]

        week = int(t/(7*24*60*60*1000000000))
        tow = t/1000000000 - week*7*24*60*60

        for j in range(len(v)):
            if np.isnan(v[j]):
                continue

            if np.isnan(sat_clock_error[j]):
                obj = dog.get_sat_info(sat_names[j], GPSTime(week,tow  - v[j]/LIGHTSPEED))    
                if obj is None:
                    continue
                sat_pos, _, sat_clock_error[j], _ = obj

            vj = v[j] + sat_clock_error[j] * LIGHTSPEED
            obj = dog.get_sat_info(sat_names[j], GPSTime(week,tow - vj/LIGHTSPEED))    
            if obj is None:
                continue

            sat_pos, _, sat_clock_error[j], _ = obj
            sat_pos = rotate_sat(sat_pos, vj)
            dist = np.linalg.norm(sat_pos - ecef_coords)
            mat[i,j] = vj - dist

    for i in range(mat.shape[1]):
        pyplot.plot( times, mat[:,i])
    pyplot.show()        
