from src.sat_data_analyzer import analize_sat_data
from src.laika.lib.coordinates import ecef2geodetic
from src.utils.kml_writer import KMLWriter
import os

os.environ['NASA_USERNAME'] = 'ilye@yandex.ru'
os.environ['NASA_PASSWORD'] = 'Pass4nasa'

folder = "D:/databases/google-smartphone-decimeter-challenge/train/2021-04-22-US-SJC-1"
analize_sat_data(folder)
