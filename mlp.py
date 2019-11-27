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

import torch
import torch.nn as nn
import torch.nn.functional as F




class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim):
        super(MultiLayerPerceptron, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, 50)
        self.out = nn.Linear(50, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.out(x)
        return x

def prepare_input(df,travel_distances, euclid_distances, zone_coordinates):
    # pu coords
    loc0 = df['loc0'].to_numpy()
    loc1 = df['loc1'].to_numpy()
    loc0_coords = zone_coordinates[loc0]
    loc1_coords = zone_coordinates[loc1]
    dists = travel_distances[loc0, loc1]
    print(dists.shape)
    # do coords
    # travel distance
    # time in seconds
    dt = df['dt']
    dt2secs = lambda dt: dt.seconds + 86400*dt.days 
    dt_secs = dt.apply(dt2secs)
    targets = dt_secs.to_numpy()
    print(targets.shape)



def train():
    travel_distances = np.load('./FeatureData_processed/TravelDistance.npy')
    euclid_distances = np.load('./FeatureData_processed/EuclideanDistance.npy')
    zone_coordinates = np.load('./FeatureData_processed/ZoneCoordinates.npy')
    df = TaxiDataLoader(range(0,4))

    input = prepare_input(df, travel_distances, euclid_distances, zone_coordinates)
   
    

if __name__ == '__main__':
    train()
