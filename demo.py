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
from TaxiZoneLoader import zoneBorough, zoneCoordinates, zoneVertices
from TaxiDataLoader import TaxiDataLoader



# TaxiZoneLoader.presenter(zoneBorough, zoneCoordinates, zoneVertices)
# WeatherLoader.presenter(Weather_dict)
# HolidayLoader.presenter(Holidays)

df = TaxiDataLoader(range(0,3))

print(df)



#
