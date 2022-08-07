from src.android_gnss_log import align_gnss_log
from src.laika.lib.coordinates import ecef2geodetic
from src.utils.kml_writer import KMLWriter
from src.utils.slac import loadSlac
from dateutil.parser import parse
import os
from datetime import datetime
from src.laika.gps_time import GPSTime
from src.laika.downloader import download_cors_station
from src.laika.rinex_file import RINEXFile


os.environ['NASA_USERNAME'] = 'ilye@yandex.ru'
os.environ['NASA_PASSWORD'] = 'Pass4nasa'

#align_gnss_log('helgilab/2020-05-14-US-MTV-1', 'helgilab/2020-05-14-US-MTV-1/basestation.20o')
#align_gnss_log('helgilab/20210913', 'helgilab/20210913/basestation.21o')
#align_gnss_log('helgilab/20210918', 'helgilab/20210918/basestation.21o')
folder = 'helgilab/20210921.oleg'
folder = 'helgilab/20211004.yurec'
folder = 'helgilab/20211022_testArma'
folder = 'helgilab/20211125'
folder = 'helgilab/20220121'


folder = "D:/databases/google-smartphone-decimeter-challenge/train/2021-04-22-US-SJC-1"

time = GPSTime.from_datetime(datetime(2021, 4, 22))
slac_rinex_obs_file = download_cors_station(time, 'slac', "D:/databases/slac")
times, ecef_poses = align_gnss_log(folder, slac_rinex_obs_file)
wgs_poses = ecef2geodetic(ecef_poses)
#align_gnss_log('helgilab/20210921.yurec', 'helgilab/20210921.yurec/basestation.21o')


kml = KMLWriter(folder+ "/predicted.kml", "predicted")
kml.addTrack('predicted','FF00FF00', wgs_poses)
kml.finish()

