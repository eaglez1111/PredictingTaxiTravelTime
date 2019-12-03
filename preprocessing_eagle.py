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

t2secs = lambda t: 3600*t.hour + 60*t.minute + t.second
dt2secs = lambda dt: dt.seconds + 86400*dt.days
getHour = lambda t: t.hour

for idx in range(674):
    print("{}/674".format(idx+1))
    df = TaxiDataLoader([idx])

    # handles those weird rows
    filter = pd.Series(df['t0'].dt.year == 2017)
    df = df[filter.values]



    df['start_time'] = df['t0'].apply(t2secs)
    df['travel_time'] = df['dt'].apply(dt2secs)

    df = df[df['travel_time'] > 30]
    df = df[df['travel_time'] < 3600*10]

    df['euc_dist'] = euclidianDistance[df['loc0'], df['loc1']]
    df['real_dist'] = travelDistance[df['loc0'], df['loc1']]
    df['bor0'] = zoneBorough[df['loc0']]
    df['bor1'] = zoneBorough[df['loc1']]

    hours = df['t0'].values.astype('datetime64[h]')

    hoursArray = np.asarray(pd.to_datetime(hours).hour)
    # df['morn'] = np.logical_and(hoursArray >= 4, hoursArray < 12).astype(int)
    # df['aftnoon'] = np.logical_and(hoursArray >= 12, hoursArray < 20).astype(int)
    # df['night'] = np.logical_or(hoursArray >= 20, hoursArray < 4).astype(int)
    # df['hour'] = df['hour'].astype(CategoricalDtype(categories=range(24)))
    df['hour'] = df['t0'].apply(getHour)

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
    df['hday'] = [(d in holidays or (d-1) in holidays) for d in days]

    for i in range(2,4+1):
        df['wkday^{}'.format(i)] = df['wkday']**i
    for i in range(2,6+1):
        df['hour^{}'.format(i)] = df['hour']**i

    # df['hour'] = df['hour'].astype(CategoricalDtype(categories=range(24)))
    # df['wkday'] = df['wkday'].astype(CategoricalDtype(categories=range(7)))

    df['bor0'] = df['bor0'].astype(CategoricalDtype(categories=range(6)))
    df['bor1'] = df['bor1'].astype(CategoricalDtype(categories=range(6)))

    df = pd.get_dummies(df, columns=['bor0', 'bor1']) #, 'wkday', 'hour'])

    for i in range(6):
        for j in range(6):
            df['bor{}to{}'.format(i, j)] = df['bor0_{}'.format(i)] * df['bor1_{}'.format(j)]

    for col in ['t0','t1','loc0','loc1','loc_pair','dt','bor0_0', 'bor0_1', 'bor0_2', 'bor0_3', 'bor0_4', 'bor0_5', 'bor1_0', 'bor1_1', 'bor1_2', 'bor1_3', 'bor1_4', 'bor1_5']:
        del df[col]


    df.to_pickle('PreProcessedData_eagle/df_{}.pkl'.format(idx+1))
