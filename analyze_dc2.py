from src.anylize_derivative import analyze_derivative, get_num_good_speeds, get_num_good_shifts, calc_track_speed
from src.anylize_derivative1 import analyze_derivative1 

import numpy as np
import pandas as pd
import glob as gl
import random
from os.path import exists


#start_path = "2021-08-04-US-SJC-1\\s*\\"
#start_path = "2020-12-10-US-SJC-1\\*\\"
#start_path = "2021-12-28-US-MTV-1\\*\\"
start_path = "2021-12-15-US-MTV-1\\XiaomiMi8\\"
start_path = "2021-12-09-US-LAX-2\\GooglePixel5\\"
start_path = "2021-12-07-US-LAX-1\\SamsungGalaxyS20Ultra\\"
start_path = "2021-12-09-US-LAX-2\\XiaomiMi8\\"
start_path = "2021-07-19-US-MTV-1\\XiaomiMi8\\"
start_path = "2021-07-01-US-MTV-1\\SamsungGalaxyS20Ultra\\"
start_path = "2020-12-10-US-SJC-2\\XiaomiMi8\\" #BAD GT in begin and end
start_path = "2021-04-26-US-SVL-2\\SamsungGalaxyS20Ultra\\" #2.5010
start_path = "2020-05-29-US-MTV-1\\GooglePixel4\\" #<0.8
start_path = "2021-12-09-US-LAX-2\\XiaomiMi8\\" #2.00
start_path = "2021-04-26-US-SVL-2\\XiaomiMi8\\" #1.82
#start_path = "2020-07-24-US-MTV-1\\GooglePixel5\\"



#start_path = "*\\*\\"
paths = gl.glob("D:\\databases\\smartphone-decimeter-2022\\train\\"+start_path)
#random.shuffle(paths)
existent = []


for i, dirname in enumerate(paths):
    #dirname = "D:\\databases\\smartphone-decimeter-2022\\test\\"+dirname+"\\"
    drive, phone = dirname.split('\\')[-3:-1]
    if phone == 'cors_obs':
        continue
    tripID = f"\\train\\{drive}\\{phone}"
    # if i == 1:
    #     continue

    # if exists(dirname+'\\result0.pkl'):
    #     existent.append(dirname)
    #     print("Exists", i, tripID)
    #     continue

    print("Reading", i, tripID)
    calc_track_speed(tripID)
print('*'*50)    
print('*'*50)    
print('*'*50)    
for i, dirname in enumerate(existent):
    drive, phone = dirname.split('\\')[-3:-1]
    if phone == 'cors_obs':
        continue

    tripID = f"\\train\{drive}\\{phone}"
    print("Reading", i, tripID)
    calc_track_speed(tripID)
    
for i, dirname in enumerate(sorted(gl.glob(r"D:\databases\smartphone-decimeter-2022\test\2022-04-25-US-OAK-2/*/"))):
    drive, phone = dirname.split('\\')[-3:-1]
    tripID = f"\\test\{drive}\\{phone}"
    print("Reading", i, tripID)
    g,t = get_num_good_shifts(tripID)
    print("Good", g, "total", t, "percent", g*100/t)



base_path = r'D:\databases\smartphone-decimeter-2022'
gsr = 0
#for i, dirname in enumerate(sorted(gl.glob(r"D:\databases\smartphone-decimeter-2022\test\*\*/"))):
for i in range(0):
    drive, phone = dirname.split('\\')[-3:-1]
    if phone == 'cors_obs':
        continue

    tripID = f"\\train\{drive}\\{phone}"
    gnss_df = pd.read_csv(f"{dirname}/device_gnss.csv")

    gnss_df["adr_valid"] = (gnss_df["AccumulatedDeltaRangeState"] & 2**0) != 0
    gnss_df["adr_reset"] = (gnss_df["AccumulatedDeltaRangeState"] & 2**1) != 0
    gnss_df["adr_slip"]  = (gnss_df["AccumulatedDeltaRangeState"] & 2**2) != 0
    sr = (~gnss_df["adr_valid"]).mean()
    if sr > gsr:
        gsr = sr
        print("* ", end = '')
    print( i, tripID, "\tslip rate", sr)
public_trips = [
    '2021-06-22-US-MTV-1\\XiaomiMi8',
    '2021-09-07-US-MTV-1\\SamsungGalaxyS20Ultra',
    '2021-09-20-US-MTV-1\\XiaomiMi8',
    '2022-01-04-US-MTV-1\\SamsungGalaxyS20Ultra',
    '2022-01-26-US-MTV-1\\XiaomiMi8',
    '2022-02-01-US-SJC-1\\XiaomiMi8',
    '2022-02-08-US-SJC-1\\XiaomiMi8',
    '2022-02-23-US-LAX-3\\XiaomiMi8',
    '2022-02-23-US-LAX-5\\XiaomiMi8',
    '2022-02-24-US-LAX-1\\SamsungGalaxyS20Ultra',
    '2022-02-24-US-LAX-3\\XiaomiMi8',
    '2022-02-24-US-LAX-5\\SamsungGalaxyS20Ultra',
    '2022-03-31-US-LAX-3\\SamsungGalaxyS20Ultra',
    '2022-04-01-US-LAX-1\\SamsungGalaxyS20Ultra',
    '2022-04-01-US-LAX-3\\XiaomiMi8',
    '2022-04-22-US-OAK-2\\XiaomiMi8',
    '2021-08-12-US-MTV-1\\GooglePixel4',
    '2021-08-17-US-MTV-1\\GooglePixel5',
    '2021-08-24-US-SVL-2\\GooglePixel5',
    '2021-09-14-US-MTV-1\\GooglePixel5',
    '2021-09-20-US-MTV-2\\GooglePixel4',
    '2022-01-11-US-MTV-1\\GooglePixel6Pro',
    '2022-02-15-US-SJC-1\\GooglePixel5',
    '2022-02-23-US-LAX-1\\GooglePixel5',
    '2022-03-14-US-MTV-1\\GooglePixel5',
    '2022-03-17-US-SJC-1\\GooglePixel5',
    '2022-04-22-US-OAK-1\\GooglePixel5',
    '2022-04-25-US-OAK-2\\GooglePixel4'
]
private_trips = [
    '2021-04-28-US-MTV-2\\SamsungGalaxyS20Ultra',
    '2021-09-28-US-MTV-1\\GooglePixel5',
    '2021-11-05-US-MTV-1\\XiaomiMi8',
    '2021-11-30-US-MTV-1\\GooglePixel5',
    '2022-01-18-US-SJC-2\\GooglePixel5',
    '2022-03-22-US-MTV-1\\SamsungGalaxyS20Ultra',
    '2022-03-31-US-LAX-1\\GooglePixel5',
    '2022-04-25-US-OAK-1\\GooglePixel5',
]
