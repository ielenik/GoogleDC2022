import numpy as np
import pandas as pd
from .coords_tools import getValuesAtTime
from .coords_tools import rotate_sat
from tqdm import tqdm
from src.laika.gps_time import GPSTime

def process_gnss_log(df_raw, base, dog):
    LIGHTSPEED = 2.99792458e8
    NaN = float("NaN")

    df_raw['L15'] = '1'
    df_raw.loc[df_raw['CarrierFrequencyHz'] < 1575420030,'L15'] = '5'

    df_raw['Svid_str'] = df_raw['Svid'].apply(str)
    df_raw.loc[df_raw['Svid_str'].str.len() == 1, 'Svid_str'] = '0' + df_raw['Svid_str']
    
    df_raw['Constellation'] ='U'
    df_raw.loc[df_raw['ConstellationType'] == 1, 'Constellation'] = 'G'
    df_raw.loc[df_raw['ConstellationType'] == 3, 'Constellation'] = 'R'
    df_raw.loc[df_raw['ConstellationType'] == 5, 'Constellation'] = 'C'
    df_raw.loc[df_raw['ConstellationType'] == 6, 'Constellation'] = 'E'    
    df_raw = df_raw[df_raw['Constellation'] != 'U'].copy(deep=True) 
    
    df_raw['SvName'] = df_raw['Constellation'] + df_raw['Svid_str']
    df_raw['SvNameType'] = df_raw['Constellation'] + df_raw['Svid_str'] + '_' + df_raw['L15']

    df_raw['NanosSinceGpsEpoch'] = df_raw['TimeNanos'] - df_raw['FullBiasNanos']
    df_raw['PrNanos'] = df_raw['NanosSinceGpsEpoch'] - df_raw['ReceivedSvTimeNanos']
    df_raw['PrNanos'] -= np.floor(df_raw['PrNanos']*1e-9 + 0.02).astype(np.int64) * 1000000000
    df_raw['ReceivedSvTimeNanos'] = df_raw['NanosSinceGpsEpoch'] + df_raw['PrNanos'] # fix sat time
    df_raw['PrM'] = LIGHTSPEED * df_raw['PrNanos'] * 1e-9
    df_raw['PrSigmaM'] = LIGHTSPEED * 1e-9 * df_raw['ReceivedSvTimeUncertaintyNanos']
    df_raw['SAT_FULL_INDEX'] = pd.factorize(df_raw['SvNameType'].tolist())[0]    
    df_raw['ISBRM_INDEX'] = pd.factorize((df_raw['Constellation']+df_raw['L15']).tolist())[0]    


    delta_millis = df_raw['PrNanos'] / 1e6
    where_good_signals = (delta_millis > -20) & (delta_millis < 300)
    df_invalide = df_raw[~where_good_signals]
    print("Bad: ", np.sum(~where_good_signals))
    print("Good: ", np.sum(where_good_signals))
    #df_raw = df_raw[where_good_signals].copy()
    single_path = df_raw['MultipathIndicator'] != 1
    print("Multipath: ", np.sum(~single_path))
    print("excluding multipath")
    df_raw = df_raw[single_path].copy()
    
    good_time_bias = df_raw['BiasUncertaintyNanos'] < 100
    print("Bad bias: ", np.sum(~good_time_bias))
    #df_raw = df_raw[good_time_bias].copy()

    df_raw['Epoch'] = 0
    df_raw.loc[df_raw['NanosSinceGpsEpoch'] - df_raw['NanosSinceGpsEpoch'].shift() > 10*1e6, 'Epoch'] = 1
    df_raw['Epoch'] = df_raw['Epoch'].cumsum()

    
    num_used_satelites = df_raw['SAT_FULL_INDEX'].max() + 1
    num_epochs         = df_raw['Epoch'].max() + 1
    
    base_local_values = np.ones((len(base['times']),num_used_satelites))*NaN
    base_delta_values = np.ones((len(base['times']),num_used_satelites))*NaN
    sat_names = []
    sat_types = []
    sat_uniq = df_raw.drop_duplicates(['SAT_FULL_INDEX'],keep='last')
    sat_uniq = sat_uniq.sort_values(['SAT_FULL_INDEX'])
    for _, df in sat_uniq.iterrows():
        sat_num = df['SAT_FULL_INDEX']
        sat_name = df['SvNameType']
        while sat_num > len(sat_names):
            sat_names.append('dummy')
            sat_types.append(7)

        sat_names.append(df['SvName'])
        sat_types.append(df['ISBRM_INDEX'])
        if sat_name in base['sat_registry']:
            base_local_values[:,sat_num] = base['values'][:,base['sat_registry'][sat_name]]
            base_delta_values[:,sat_num] = base['deltas'][:,base['sat_registry'][sat_name]]


    def getCorrections(time_nanos, rsat):
        psevdobase = getValuesAtTime(base['times'], base_local_values, time_nanos)
        dist_to_sat = np.linalg.norm(rsat - base['coords'], axis = -1)
        return dist_to_sat - psevdobase
    def getBaseDelta(time_nanos, rsat):
        return getValuesAtTime(base['times'], base_delta_values, time_nanos)

    sat_positions     = np.ones((num_epochs, num_used_satelites, 3))
    sat_velosities     = np.ones((num_epochs, num_used_satelites, 3))
    sat_frequencis     = np.ones((num_epochs, num_used_satelites))*NaN

    sat_psevdodist    = np.zeros((num_epochs, num_used_satelites))
    sat_psevdovalid   = np.zeros((num_epochs, num_used_satelites))
    sat_psevdoweights = np.zeros((num_epochs, num_used_satelites))

    sat_basedeltarange   = np.zeros((num_epochs, num_used_satelites))
    sat_deltarange       = np.zeros((num_epochs, num_used_satelites))
    sat_deltarangeuncert = np.ones((num_epochs, num_used_satelites))*1e5
    sat_deltastate        = np.zeros((num_epochs, num_used_satelites)).astype(np.int64)
    
    sat_deltaspeed = np.zeros((num_epochs, num_used_satelites))
    sat_deltaspeeduncert = np.ones((num_epochs, num_used_satelites))*1e5
    
    epoch_times = np.zeros((num_epochs)).astype(np.int64)
    utcTimeMillis = np.zeros((num_epochs)).astype(np.int64)
    sat_clock_bias = np.zeros((num_used_satelites))

    for epoch_number, epoch in tqdm(df_raw.groupby(['Epoch'])):
        time_nanos = epoch["NanosSinceGpsEpoch"].to_numpy()
        time_nanos = np.sort(time_nanos)
        time_nanos = time_nanos[len(time_nanos)//2]

        epoch_times[epoch_number] = time_nanos
        
        time_millis = epoch["utcTimeMillis"].to_numpy()
        time_millis = np.sort(time_millis)
        time_millis = time_millis[len(time_millis)//2]

        utcTimeMillis[epoch_number] = time_millis

        #if epoch_number == 800:
        #    deltarange      = 123
        #if epoch_number == 66:
        #    break

        for _,r in epoch.iterrows():
            sat_index = r['SAT_FULL_INDEX']
            sat_name  = r['SvName']
            if sat_names[sat_index] != sat_name:
                print("Error in satelite registry")
            

            ReceivedSvTimeNanos = r['ReceivedSvTimeNanos'] - int(sat_clock_bias[sat_index]*1000000000)
            week = int(ReceivedSvTimeNanos/(7*24*60*60*1000000000))
            tow = ReceivedSvTimeNanos/1000000000 - week*7*24*60*60
            timegp = GPSTime(week,tow)
            obj = dog.get_sat_info(sat_names[sat_index], timegp)
            #obj = (0,0,0),0,0,0 #
            if obj is None:
                continue

            sat_pos, sat_vel, sat_clock_err, sat_clock_drift = obj

            sat_clock_bias[sat_index] = sat_clock_err
            sat_positions[epoch_number,sat_index] = sat_pos
            sat_velosities[epoch_number,sat_index] = sat_vel
            sat_psevdodist[epoch_number,sat_index] = r['PrM']
            sat_psevdovalid[epoch_number,sat_index] = 1
            sat_psevdoweights[epoch_number,sat_index] = 1/r['PrSigmaM']

            sat_deltarange[epoch_number,sat_index] = r['AccumulatedDeltaRangeMeters']
            if sat_deltarange[epoch_number,sat_index] != 0:
                sat_deltarangeuncert[epoch_number,sat_index] = r['AccumulatedDeltaRangeUncertaintyMeters']
                sat_deltastate[epoch_number,sat_index] = r['AccumulatedDeltaRangeState']

            sat_deltaspeed[epoch_number,sat_index] = r['PseudorangeRateMetersPerSecond']
            sat_deltaspeeduncert[epoch_number,sat_index] = r['PseudorangeRateUncertaintyMetersPerSecond']
            sat_frequencis[epoch_number,sat_index] = r['CarrierFrequencyHz']


        sat_positions[epoch_number] = rotate_sat(sat_positions[epoch_number], -(sat_psevdodist[epoch_number] + sat_clock_bias[sat_index]*LIGHTSPEED)) 
        corr = getCorrections(time_nanos,sat_positions[epoch_number])
        sat_psevdodist[epoch_number] += corr
        sat_basedeltarange[epoch_number] = getBaseDelta(time_nanos,sat_positions[epoch_number])

    sat_psevdovalid[np.isnan(sat_psevdodist)] = 0
    sat_psevdodist[sat_psevdovalid == 0] = 0

    i = 1
    utcTimeMillisOld = utcTimeMillis.copy()
    newtimes = []
    while i < len(utcTimeMillis):
        if utcTimeMillis[i] - utcTimeMillis[i-1] > 1700:
            utcTimeMillis = np.insert(utcTimeMillis, i, utcTimeMillis[i-1]+min(1000,(utcTimeMillis[i] - utcTimeMillis[i-1])//2), axis = 0)
            newtimes.append(i)
        i += 1

    if len(newtimes) > 0:
        def interp_nd(epochtimes, valtimes, valcum):
            dim = []
            shape = list(valcum.shape)
            valcum = valcum.reshape((len(valcum),-1))
            for i in range(len(valcum[0])):
                dim.append(np.interp(epochtimes, valtimes, valcum[:,i]))
            shape[0] = len(epochtimes)
            valcum = np.array(dim).T.reshape(shape)
            return valcum 

        sat_positions        = interp_nd(utcTimeMillis, utcTimeMillisOld, sat_positions)
        sat_velosities       = interp_nd(utcTimeMillis, utcTimeMillisOld, sat_velosities)
        sat_deltarange       = interp_nd(utcTimeMillis, utcTimeMillisOld, sat_deltarange)
        sat_deltastate       = interp_nd(utcTimeMillis, utcTimeMillisOld, sat_deltastate)
        sat_deltarangeuncert = interp_nd(utcTimeMillis, utcTimeMillisOld, sat_deltarangeuncert)
        sat_deltaspeed       = interp_nd(utcTimeMillis, utcTimeMillisOld, sat_deltaspeed)
        sat_deltaspeeduncert = interp_nd(utcTimeMillis, utcTimeMillisOld, sat_deltaspeeduncert)
        sat_basedeltarange   = interp_nd(utcTimeMillis, utcTimeMillisOld, sat_basedeltarange)
        sat_frequencis       = interp_nd(utcTimeMillis, utcTimeMillisOld, sat_frequencis)
        sat_psevdovalid      = interp_nd(utcTimeMillis, utcTimeMillisOld, sat_psevdovalid)
        sat_psevdodist       = interp_nd(utcTimeMillis, utcTimeMillisOld, sat_psevdodist)
        sat_psevdoweights    = interp_nd(utcTimeMillis, utcTimeMillisOld, sat_psevdoweights)
        epoch_times          = np.interp(utcTimeMillis, utcTimeMillisOld, epoch_times)

        newtimes = np.array(newtimes)
        sat_deltastate[newtimes] = 0
        sat_deltarangeuncert[newtimes] = 1e10
        sat_deltaspeeduncert[newtimes] = 1e10
        sat_psevdovalid[newtimes] = 0
        sat_psevdoweights[newtimes] = 0

    return { 
        'sat_psevdovalid':sat_psevdovalid,
        'sat_psevdodist':sat_psevdodist,
        'sat_psevdoweights':sat_psevdoweights,

        'sat_deltarange':sat_deltarange,
        'sat_basedeltarange':sat_basedeltarange,
        'sat_deltarangeuncert':sat_deltarangeuncert,
        'sat_deltastate':sat_deltastate,

        'sat_deltaspeed':sat_deltaspeed,
        'sat_deltaspeeduncert':sat_deltaspeeduncert,

        'sat_positions':sat_positions,
        'sat_velosities':sat_velosities,
        'epoch_times':epoch_times,
        'utcTimeMillis':utcTimeMillis,
        'sat_frequencis':sat_frequencis,
        'sat_types': np.array(sat_types),
        'sat_names': np.array(sat_names),
    }
