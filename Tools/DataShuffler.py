import numpy as np
import sys




_Prefix = './TaxiData/'
File = np.loadtxt('./TaxiData/manifest.txt',dtype="str")#[FileIndices] # Only loading first several files, can't do all at once



def readLines(path):
    Lines = []
    with open(path) as f:
        line = f.readline()
        while line:
            Lines.append(line)
            line = f.readline()
    return Lines

def writeLines(path,Lines,cmd='w'):
    f = open(path, cmd)
    for i in range(len(Lines)):
        f.write(Lines[i])
    f.close()
    return 1

''' Remove data from non-2017 years (only exist in file3(File[2]) )'''
f3Lines = readLines(_Prefix+File[2])
fNLines = readLines(_Prefix+File[-1])

f3 = open(_Prefix+File[2], "w")
f3.write(f3Lines[0])
for i in range(1,len(f3Lines)):
    if (f3Lines[i][5:9]=='2017'):
        line = f3Lines[i]
    else:
        line = fNLines.pop()
    f3.write(line)
f3.close()

writeLines(_Prefix+File[-1],fNLines)



'''  '''

n_lines = 100000-1
N_files = len(File) # 674
N_lines = n_lines*(N_files-1)+(2863-1)
_NewPrefix = './TaxiData_shuffled/'
ShuffledLineNum = np.random.permutation(N_lines)


line0 = '"VendorID","tpep_pickup_datetime","tpep_dropoff_datetime","passenger_count","PULocationID","DOLocationID","payment_type"'
for i in range(N_files):
    writeLines(_NewPrefix+File[i],[line0,'\n'])

for i in range(N_files):
    Id = np.array(ShuffledLineNum[i*n_lines:(i+1)*n_lines])
    fileId, lineId = (Id/n_lines).astype('int16'), (Id%n_lines)+1
    #lineId = lineId[np.argsort(fileId)]
    #fileId = np.sort(fileId)
    print(i,':')
    Lines = []
    for j in range(N_files):
        print('--',j)
        Lines_fJ = np.array( readLines(_Prefix+File[j]) )
        Lines_fJ = Lines_fJ [ lineId[np.argwhere(fileId==j).reshape(-1)] ]
        Lines = np.append(Lines,Lines_fJ)
    Lines = Lines[np.random.permutation(len(Lines))]
    writeLines(_NewPrefix+File[i],Lines,"a")










#
