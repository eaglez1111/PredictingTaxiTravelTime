#from __future__ import print_function
#import torch
import numpy as np
import pandas as pd



_Prefix = './TaxiData/'
_NumOfLoc = 265
_OldColName = ["VendorID","tpep_pickup_datetime","tpep_dropoff_datetime","passenger_count","PULocationID","DOLocationID","payment_type"]
_NewColName = ['vid','t0','t1','ps','loc0','loc1','pt']
_DataType = ['int8','datetime64[s]','datetime64[s]','int8','int16','int16','int8']



''' Load data, Rename columns '''  # Basic Pandas operation: https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html
File = np.loadtxt('./TaxiData/manifest.txt',dtype="str")[0:3] # Only loading first several files, can't do all at once
for i in range(len(File)):
    print('Loading:', File[i])
    df_temp = pd.read_table(_Prefix+File[i], sep=',')
    if i==0: df = df_temp
    else:    df = df.append(df_temp, ignore_index=True)
for i in range(7):
    df.iloc[:,i] = df.iloc[:,i].astype(_DataType[i])
    df = df.rename( columns = {_OldColName[i]:_NewColName[i]})



''' Generate useful features '''
df['dt'] = df['t1'] - df['t0'] # delta T , duration
df['loc'] = (df['loc0'].astype('int32')*_NumOfLoc+df['loc1']) # Location pair ID, one ID refers to one unique start-end location pair
df['wkday'] = df['t1'].dt.weekday # weekday, Mon=0 Sun=6



''' Notes to the team '''
# Weather: downloaded in "./Otherdata" , from https://rp5.ru/Weather_archive_in_New_York,_Central_Park_(AWS),_METAR
# Public holiday? what about this one? https://gist.github.com/shivaas/4758439
# Also, is this cheating? https://developers.google.com/maps/documentation/distance-matrix/intro



''' View the dataframe '''
print(df)
