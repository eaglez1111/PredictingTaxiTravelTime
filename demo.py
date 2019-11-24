#from __future__ import print_function
#import torch

import numpy as np
import pandas as pd

import sys
sys.path.append('./Tools')
import HolidayLoader
import WeatherLoader
import TaxiZoneLoader
from HolidayLoader import Holidays
from WeatherLoader import Weather_dict
from TaxiZoneLoader import ZoneBorough, ZoneCoordinates, ZoneVertices, EuclideanDistance, TravelDistance
from TaxiDataLoader import TaxiDataLoader




TaxiZoneLoader.presenter(ZoneBorough, ZoneCoordinates, ZoneVertices)
# WeatherLoader.presenter(Weather_dict)
# HolidayLoader.presenter(Holidays)

df = TaxiDataLoader(range(0,3))

N = len(df)
for i in range(N):
    pass

print(df)



#
