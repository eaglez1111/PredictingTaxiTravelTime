#from __future__ import print_function
#import torch

import numpy as np
import pandas as pd

import os
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

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


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
    normalizer = np.amax(input, axis=0)
    input = input / normalizer
    input = torch.FloatTensor(input)
    input = input.to(device)
    return input

    
def prepare_targets(df):
    # time in seconds
    dt = df['dt']
    dt2secs = lambda dt: dt.seconds + 86400*dt.days 
    dt_secs = dt.apply(dt2secs)
    targets = dt_secs.to_numpy()
    targets = targets.reshape(-1,1)
    targets = torch.FloatTensor(targets)
    targets = targets.to(device)
    return targets



def train(file_batch_size=10, input_batch_size=20, num_epochs=20):
    travel_distances = np.load('./FeatureData_processed/TravelDistance.npy')
    euclid_distances = np.load('./FeatureData_processed/EuclideanDistance.npy')
    zone_coordinates = np.load('./FeatureData_processed/ZoneCoordinates.npy')

    mlp = MultiLayerPerceptron(input_dim=6)
    mlp = mlp.to(device)
    optimizer = torch.optim.SGD(mlp.parameters(), lr=1e-12)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    criterion = nn.MSELoss()
    val_criterion = nn.L1Loss()

    num_files = 634
    num_train = 200
    num_val = num_train // 20

    val_idxs = np.arange(num_files)
    np.random.shuffle(val_idxs)
    val_idxs = val_idxs[:num_val]
    val_loss = 0

    train_idxs = np.asarray([idx for idx in range(num_files) if idx not in val_idxs])
    for epoch_idx in tqdm(range(num_epochs)):
        np.random.shuffle(train_idxs)

        # train over batch
        mlp.train()
        train_file_idxs = train_idxs[:num_train]
        train_file_idxs = train_file_idxs.reshape(-1,file_batch_size)
        train_loss = 0
        tqdm.write(f'epoch {epoch_idx}')
        for batch_num, train_file_batch in enumerate(train_file_idxs):
            df = TaxiDataLoader(train_file_batch)
            #inputs = prepare_inputs(df, travel_distances, zone_coordinates)
            inputs = prepare_inputs(df, euclid_distances, zone_coordinates)
            targets = prepare_targets(df)
            batch_idxs = np.arange(0,inputs.shape[0],input_batch_size)
            np.random.shuffle(batch_idxs)
            batch_idxs = batch_idxs.reshape(-1,input_batch_size)
            file_batch_loss = 0
            for batch_idx in range(batch_idxs.shape[0]):
                optimizer.zero_grad()
                idxs = torch.LongTensor(batch_idxs[batch_idx]).to(device)
                input = torch.index_select(inputs, 0, idxs)
                target = torch.index_select(targets, 0, idxs)
                output = mlp(input)
                loss = criterion(output, target)
                loss.backward()
                file_batch_loss += loss.item()
                scheduler.step(val_loss)
            file_batch_loss /= inputs.shape[0]
            if batch_num %5 == 0:
                tqdm.write(f'\tbatch {batch_num} loss {file_batch_loss:.2f}')
            train_loss += file_batch_loss
        train_loss /= train_file_idxs.shape[0]

        # compute validation loss
        mlp.eval()
        df = TaxiDataLoader(val_idxs)
        #inputs = prepare_inputs(df, travel_distances, zone_coordinates)
        inputs = prepare_inputs(df, euclid_distances, zone_coordinates)
        targets = prepare_targets(df)
        batch_idxs = np.arange(0,inputs.shape[0],input_batch_size)
        np.random.shuffle(batch_idxs)
        batch_idxs = batch_idxs.reshape(-1,input_batch_size)
        batch_loss = 0
        val_l1loss = list()
        for batch_idx in range(batch_idxs.shape[0]):
            idxs = torch.LongTensor(batch_idxs[batch_idx]).to(device)
            input = torch.index_select(inputs, 0, idxs)
            target = torch.index_select(targets, 0, idxs)
            output = mlp(input)
            l1loss = np.abs(target-output)
            val_l1loss.append(l1loss)
            loss = criterion(output, target)
            batch_loss += loss.item()
        batch_loss /= inputs.shape[0]
        val_loss = batch_loss
   
        if epoch_idx%1==0:
            tqdm.write(f'\ttrain_loss = {train_loss:.2f} val_loss = {val_loss:.2f}\n')
    num_models = len(os.listdir('models'))
    torch.save(mlp.state_dict(), f'models/mlp_{num_models}')


if __name__ == '__main__':
    train()
