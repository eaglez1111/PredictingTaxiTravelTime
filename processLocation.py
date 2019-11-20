import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt


N = 266

BOROUGH = { 'EWR':0, 'Bronx':1, 'Brooklyn':2, 'Manhattan':3, 'Queens':4, 'Staten Island':5 }
with open('./OtherData/zones.geojson') as f:
    data = json.load(f)
zoneCoordinates = np.zeros([N,2])
zoneBorough = np.ones(N)*(-1)
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



''' Plot '''
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
