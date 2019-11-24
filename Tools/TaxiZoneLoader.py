import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import sys


_NumOfLoc = 266
N = _NumOfLoc
BOROUGH = { 'EWR':0, 'Bronx':1, 'Brooklyn':2, 'Manhattan':3, 'Queens':4, 'Staten Island':5 }

def presenter(ZoneBorough,ZoneCoordinates,ZoneVertices):
    print("\n\n\n\nPresenting TaxiZone:\n")
    ZoneColor = { 0:'c', 1:'r', 2:'y', 3:'b', 4:'g', 5:'m'}
    Boroughcolor = { 0:(0.65,1,1), 1:(1,.65,.65), 2:(1,1,.65), 3:(.65,.65,1), 4:(.65,1,.65), 5:(1,.65,1) }
    for id in range(N):
        if ZoneBorough[id]==-1: continue
        print(id,ZoneCoordinates[id][0],ZoneCoordinates[id][1],ZoneBorough[id],ZoneName[id])
        plt.text(ZoneCoordinates[id][0],ZoneCoordinates[id][1],str(id),label='')
        plt.plot(ZoneCoordinates[id][0],ZoneCoordinates[id][1],'*'+ZoneColor[ZoneBorough[id]] )
        plt.fill(ZoneVertices[id].T[0],ZoneVertices[id].T[1],color=Boroughcolor[ZoneBorough[id]])
        # n = len(ZoneVertices[id])
        # for i in range(n):
        #     j = (i+1)%n
        #     plt.plot(ZoneVertices[id].T[0][[i,j]], ZoneVertices[id].T[1][[i,j]],'-k',label='')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('')
    #plt.legend(loc='bottom right')
    plt.show()

    id = range(N)
    plt.plot(id,EuclideanDistance[1][:]*136.7985008644862,'r.-')
    plt.plot(id,TravelDistance[1][:],'b.-')
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
    ZoneBorough = np.load('./FeatureData_processed/ZoneBorough.npy',allow_pickle = True)
    ZoneCoordinates = np.load('./FeatureData_processed/ZoneCoordinates.npy',allow_pickle = True)
    [ZoneName] = np.load('./FeatureData_processed/ZoneName.npy',allow_pickle = True)
    [ZoneVertices] = np.load('./FeatureData_processed/ZoneVertices.npy',allow_pickle = True)
    EuclideanDistance = np.load('./FeatureData_processed/EuclideanDistance.npy',allow_pickle = True)
    TravelDistance = np.load('./FeatureData_processed/TravelDistance.npy',allow_pickle = True)

    print("Taxi Zone Data loaded successfully!")


else :

    ''' Load TaxiZone Data and save '''

    with open('./FeatureData_raw/zones.geojson') as f:
        data = json.load(f)
    ZoneCoordinates = np.zeros([N,2])
    ZoneName = {}
    ZoneBorough = np.ones(N,dtype='int8')*(-1)
    ZoneVertices = {}

    for zone in data['features']:
        id = int(zone['properties']['location_id'])
        ZoneName[id] = zone['properties']['zone']
        ZoneBorough[id] = BOROUGH[zone['properties']['borough']]
        vertices = []
        for PG in zone['geometry']['coordinates']:
            for pg in PG:
                if len(vertices)==0:
                    vertices = np.array(pg)
                else:
                    np.append( vertices , np.array(pg) , axis=0 )
        X, Y = vertices.T[0], vertices.T[1]
<<<<<<< HEAD
        ZoneCoordinates[id] = [np.mean([np.min(X),np.max(X)]),np.mean([np.min(Y),np.max(Y)])]
        ZoneVertices[id]=vertices



    ''' Making up data for the missing zones '''
    ID0 = [57,104,105,264,265,0]
    ID1 = [56,103,103,163,163,163]
    for i in range(6):
        ZoneName[ID0[i]], ZoneBorough[ID0[i]], ZoneCoordinates[ID0[i]], ZoneVertices[ID0[i]] = ZoneName[ID1[i]], ZoneBorough[ID1[i]], ZoneCoordinates[ID1[i]], ZoneVertices[ID1[i]]



    ''' Accquire Distance '''
    import requests
    import xml.etree.ElementTree as ET
    linkFix = ['http://dev.virtualearth.net/REST/V1/Routes/Driving?o=xml&wp.0=', '&wp.1=', '&key=Alkkq8JkvZ9kCYdIZXS2GzuS98HgB3dm7OqDH126lKA1dyHBb8Zu3zS7pm_qS7Th']
    EuclideanDistance, TravelDistance = np.zeros([2,N,N])
    for i in range(N):
        for j in range(N):
            pairID = i*N + j
            EuclideanDistance[i][j] = np.linalg.norm(ZoneCoordinates[i]-ZoneCoordinates[j])
            #url = linkFix[0]+ZoneName[i]+','+list(BOROUGH.keys())[ZoneBorough[i]]+linkFix[1]+ZoneName[j]+','+list(BOROUGH.keys())[ZoneBorough[j]]+linkFix[2]
            url = linkFix[0]+ str(ZoneCoordinates[i][1]) +','+ str(ZoneCoordinates[i][0]) +linkFix[1] \
                + str(ZoneCoordinates[j][1]) +','+ str(ZoneCoordinates[j][0]) +linkFix[2]
            response = requests.get(url)
            print(i,j,url)
            if response.status_code != 200:
                print(response.status_code)
                print('\n\n\nRequest failed at (',i,',',j,').\n\n\n')
            else:
                root = ET.fromstring(response.text)
                TravelDistance[i][j] = float(root[6][0][1][0][4].text)
                print(TravelDistance[i][j])

    ''' Save as .npy to future easy reload '''

    np.save('./FeatureData_processed/ZoneBorough.npy',ZoneBorough)
    np.save('./FeatureData_processed/ZoneCoordinates.npy',ZoneCoordinates)
    np.save('./FeatureData_processed/ZoneVertices.npy',[ZoneVertices])
    np.save('./FeatureData_processed/ZoneName.npy',[ZoneName])
    np.save('./FeatureData_processed/EuclideanDistance.npy',EuclideanDistance)
    np.save('./FeatureData_processed/TravelDistance.npy',TravelDistance)

=======
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
>>>>>>> 5d50529a7c8c780c3d1e87b213349ed298253e7e

    ''' Plot '''
    presenter(ZoneBorough, ZoneCoordinates, ZoneVertices)
