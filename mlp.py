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
from tqdm import tqdm




class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim):
        super(MultiLayerPerceptron, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 20)
        self.out = nn.Linear(20, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.out(x)
        return x

def prepare_inputs(df, travel_distances, zone_coordinates):
    # pu coords
    loc0 = df['loc0'].to_numpy()
    loc0_coords = zone_coordinates[loc0]
    loc0_lat, loc0_lon = np.hsplit(loc0_coords, 2)

    # pu coords
    loc1 = df['loc1'].to_numpy()
    loc1_coords = zone_coordinates[loc1]
    loc1_lat, loc1_lon = np.hsplit(loc1_coords, 2)

    # travel distances
    dists = travel_distances[loc0, loc1]
    dists = dists.reshape(-1,1)
    #dists = np.clamp(dists, a_min=min(travel_distances)/2,a_max=None)

    t0 = df['t0']
    t2secs = lambda t: 3600*t.hour + 60*t.minute + t.second
    t0 = t0.apply(t2secs)
    t0 = t0.to_numpy().reshape(-1,1)

    input = np.hstack((loc0_lat, loc0_lon, loc1_lat, loc1_lon, dists, t0))
    input = torch.FloatTensor(input)
    return input

    
def prepare_targets(df):
    # time in seconds
    dt = df['dt']
    dt2secs = lambda dt: dt.seconds + 86400*dt.days 
    dt_secs = dt.apply(dt2secs)
    targets = dt_secs.to_numpy()
    targets = torch.FloatTensor(targets)
    return targets



def train():
    travel_distances = np.load('./FeatureData_processed/TravelDistance.npy')
    #euclid_distances = np.load('./FeatureData_processed/EuclideanDistance.npy')
    zone_coordinates = np.load('./FeatureData_processed/ZoneCoordinates.npy')
    df = TaxiDataLoader(range(0,2))


    mlp = MultiLayerPerceptron(input_dim=6)
    optimizer = torch.optim.SGD(mlp.parameters(), lr=1e-12)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    criterion = nn.MSELoss()

    inputs = prepare_inputs(df, travel_distances, zone_coordinates)
    targets = prepare_targets(df)


    #for idx in tqdm(range(len(input))):
    total_loss = 0
    for idx in tqdm(range(100000)):
        optimizer.zero_grad()
        input = inputs[idx:idx+1]
        output = mlp(input)
        target = targets[idx:idx+1]
        loss = criterion(output, target)
        loss.backward()
        total_loss += loss.item()
        avg_loss = total_loss / (idx+1)
        scheduler.step(avg_loss)
        if idx % 10000 == 0:
            tqdm.write(f'index {idx} \t train loss = {avg_loss}')

    

if __name__ == '__main__':
    train()
