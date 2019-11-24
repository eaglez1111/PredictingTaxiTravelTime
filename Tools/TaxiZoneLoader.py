import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import sys


N = 266
BOROUGH = { 'EWR':0, 'Bronx':1, 'Brooklyn':2, 'Manhattan':3, 'Queens':4, 'Staten Island':5 }

def presenter(zoneBorough,zoneCoordinates,zoneVertices):
    print("\n\n\n\nPresenting TaxiZone:\n")
    ZoneColor = { 0:'c', 1:'r', 2:'y', 3:'b', 4:'g', 5:'m'}
    Boroughcolor = { 0:(0.65,1,1), 1:(1,.65,.65), 2:(1,1,.65), 3:(.65,.65,1), 4:(.65,1,.65), 5:(1,.65,1) }
    for id in range(N):
        if zoneBorough[id]==-1: continue
        print(id,zoneCoordinates[id][0],zoneCoordinates[id][1],zoneBorough[id])
        plt.text(zoneCoordinates[id][0],zoneCoordinates[id][1],str(id),label='')
        plt.plot(zoneCoordinates[id][0],zoneCoordinates[id][1],'*'+ZoneColor[zoneBorough[id]] )
        plt.fill(zoneVertices[id].T[0],zoneVertices[id].T[1],color=Boroughcolor[zoneBorough[id]])
        # n = len(zoneVertices[id])
        # for i in range(n):
        #     j = (i+1)%n
        #     plt.plot(zoneVertices[id].T[0][[i,j]], zoneVertices[id].T[1][[i,j]],'-k',label='')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('')
    #plt.legend(loc='bottom right')
    plt.show()
    print("\n\n\n")


def getDistance(x, y):
    lat1 = x[0]
    lon1 = x[1]
    lat2 = y[0]
    lon2 = y[1]
    R = 6371e3
    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    dphi = np.deg2rad(lat2-lat1)
    dlam = np.deg2rad(lon2-lon1)
    a = np.sin(dphi/2)*np.sin(dphi/2) + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)*np.sin(dlam/2)
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = R*c
    return d

if __name__ != "__main__" :

    ''' Load TaxiZone data '''
    zoneBorough = np.load('./FeatureData_processed/zoneBorough.npy',allow_pickle = True)
    zoneCoordinates = np.load('./FeatureData_processed/zoneCoordinates.npy',allow_pickle = True)
    [zoneVertices] = np.load('./FeatureData_processed/zoneVertices.npy',allow_pickle = True)
    print("Taxi Zone Data loaded successfully!")


else :

    ''' Load TaxiZone Data and save '''

    with open('./FeatureData_raw/zones.geojson') as f:
        data = json.load(f)
    zoneCoordinates = np.zeros([N,2])
    zoneBorough = np.ones(N,dtype='int8')*(-1)
    zoneVertices = {}

    for zone in data['features']:
        id = int(zone['properties']['location_id'])
        zoneBorough[id] = BOROUGH[zone['properties']['borough']]
        vertices = []
        for PG in zone['geometry']['coordinates']:
            for pg in PG:
                if len(vertices)==0:
                    vertices = np.array(pg)
                else:
                    np.append( vertices , np.array(pg) , axis=0 )
        X, Y = vertices.T[0], vertices.T[1]
        zoneCoordinates[id] = [np.mean([np.min(X),np.max(X)]),np.mean([np.min(Y),np.max(Y)])]
        zoneVertices[id]=vertices



    ''' Catch the missing zones '''

    zoneBorough[57], zoneCoordinates[57], zoneVertices[57] = zoneBorough[56], zoneCoordinates[56], zoneVertices[56]
    zoneBorough[104], zoneCoordinates[104], zoneVertices[104] = zoneBorough[103], zoneCoordinates[103], zoneVertices[103]
    zoneBorough[105], zoneCoordinates[105], zoneVertices[105] = zoneBorough[103], zoneCoordinates[103], zoneVertices[103]
    meanCoordinates = [ np.mean(zoneCoordinates.T[0][1:264]), np.mean(zoneCoordinates.T[1][1:264]) ]
    zoneCoordinates[264] = meanCoordinates
    zoneCoordinates[265] = meanCoordinates


    ''' Build Distance Matrix'''

    zoneDistances = np.zeros((266, 266))
    for i in range(266):
        for j in range(266):
            zoneDistances[i][j] = getDistance(zoneCoordinates[i], zoneCoordinates[j])
    zoneDistances /= np.max(zoneDistances)

    ''' Save as .npy to future easy reload '''

    np.save('./FeatureData_processed/zoneBorough.npy',zoneBorough)
    np.save('./FeatureData_processed/zoneCoordinates.npy',zoneCoordinates)
    np.save('./FeatureData_processed/zoneVertices.npy',[zoneVertices])
    np.save('./FeatureData_processed/zoneDistances.npy', zoneDistances)

    ''' Plot '''
    presenter(zoneBorough, zoneCoordinates, zoneVertices)
