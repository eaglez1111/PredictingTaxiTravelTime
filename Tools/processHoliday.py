import numpy as np
import sys


if len(sys.argv)>1 and sys.argv[1]=='--demo':
    ''' Demo - how to load data '''
    holidays = np.load('./FeatureData_processed/holidays.npy',allow_pickle = True)

else:
    holidays = []
    holidays_str = np.array( ['2017-01-02','2017-01-16','2017-02-20','2017-05-29','2017-07-04','2017-09-04','2017-10-09','2017-11-10','2017-11-23','2017-12-25','2017-03-17'] , dtype='datetime64[D]' )
    for st in holidays_str:
        holidays.append(np.datetime64(st,'D'))
    holidays = np.array(holidays)
    np.save('./FeatureData_processed/holidays',holidays)


print(type(holidays))
print(type(holidays[0]))
print(holidays)
