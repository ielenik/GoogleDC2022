import argparse
import sys
from src.utils.kml_writer import KMLWriter
import re




def convertCsv(inp, out):
    with open(inp) as file:
        lines = file.readlines()[1:]
        skip = 0

        coords = []
        for l in lines:
            skip += 1
            if skip % 5 != 0:
                continue

            bt = l.split(',')
            lat = float(bt[1])
            lon = float(bt[2])
            hgh = float(bt[3])

            coords.append([lat,lon,hgh])
    
    kml = KMLWriter(out, "From CSV")
    kml.addTrack('CSV from Emlid','FFFF0000', coords)
    kml.finish()

def convertKml(inp, out):
    with open(inp) as file:
        lines = file.readlines()[1:]
        skip = 0

        coords = []
        for l in lines:

            bt = re.split('<|>|,|\n| |\t',l)
            bt = bt[3:]
            if bt[0] != 'coordinates':
                continue

            skip += 1
            if skip % 1 != 0:
                continue

            lat = float(bt[2])
            lon = float(bt[1])
            hgh = float(bt[3])

            coords.append([lat,lon,hgh])

    kml = KMLWriter(out, "From Topcon")
    kml.addTrack('Topcon track','FF0000FF', coords)
    kml.finish()



inpname = sys.argv[1]
if inpname.endswith('.csv'):
    convertCsv(sys.argv[1], sys.argv[2])
elif inpname.endswith('.kml'):
    convertKml(sys.argv[1], sys.argv[2])
else:
     print("Unknown input format")

