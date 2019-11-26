import sys
sys.path.append('./Tools')
from TaxiDataLoader import TaxiDataLoader
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype



holidays = np.load("FeatureData_processed/Holidays.npy", allow_pickle=True)
weather = np.load("FeatureData_processed/Weather_dict.npy", allow_pickle=True)[0]
zoneBorough = np.load("FeatureData_processed/ZoneBorough.npy", allow_pickle=True)
euclidianDistance = np.load("FeatureData_processed/EuclideanDistance.npy", allow_pickle=True)
euclidianDistance /= np.max(euclidianDistance)
travelDistance = np.load("FeatureData_processed/TravelDistance.npy", allow_pickle=True)
travelDistance /= np.max(travelDistance)

total_rows = 0
for i in range(674):
    print("{}/647".format(i+1))
    df = TaxiDataLoader([i])
    filter = pd.Series(df['t0'].dt.year == 2017)
    df = df[filter.values]
    total_rows += len(df)
    df['euc_dist'] = euclidianDistance[df['loc0'], df['loc1']]
    df['real_dist'] = travelDistance[df['loc0'], df['loc1']]
    df['bor0'] = zoneBorough[df['loc0']]
    df['bor1'] = zoneBorough[df['loc1']]
    # print(df['t0'])
    hours = df['t0'].values.astype('datetime64[h]')
    # print(hours)
    df['hour'] = pd.to_datetime(hours).hour
    hoursArray = np.asarray(df['hour'])
    df['morn'] = np.logical_and(hoursArray >= 4, hoursArray < 12).astype(int)
    df['aftnoon'] = np.logical_and(hoursArray >= 12, hoursArray < 20).astype(int)
    df['night'] = np.logical_or(hoursArray >= 20, hoursArray < 4).astype(int)
    # df['hour'] = df['hour'].astype(CategoricalDtype(categories=range(24)))
    weatherLists = [weather[time] for time in hours]
    df['humidity'] = [row[0] for row in weatherLists]
    df['windspeed'] = [row[1] for row in weatherLists]
    df['vis'] = [row[2] for row in weatherLists]
    df['temp'] = [row[3] for row in weatherLists]
    df['haze'] = [row[5] for row in weatherLists]
    df['fog'] = [row[6] for row in weatherLists]
    df['rain'] = [row[7] for row in weatherLists]
    df['snow'] = [row[8] for row in weatherLists]
    days = df['t0'].values.astype('datetime64[D]')
    df['hday'] = [d in holidays for d in days]
    df['wkday'] = df['wkday'].astype(CategoricalDtype(categories=range(7)))
    df = pd.get_dummies(df, columns=['bor0', 'bor1'])
    if i == 0:
        df.to_csv('PreProcessedData/data.csv', header='column_names')
    else:
        df.to_csv('PreProcessedData/data.csv', mode='a', header=False)
    df.to_pickle('PreProcessedData/df_{}.pkl'.format(i+1))


# print(weather)
# print(len(weather[0].keys()))