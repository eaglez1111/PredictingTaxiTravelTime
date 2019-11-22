import numpy as np
import sys


def presenter(Holidays):
    print("\n\n\n\nPresenting Holidays:\n")
    print(type(Holidays))
    print(type(Holidays[0]))
    print(Holidays)


if __name__ != "__main__" :

    ''' Demo - how to load data '''
    Holidays = np.load('./FeatureData_processed/Holidays.npy',allow_pickle = True)
    print("Holidays loaded successfully!")



else:
    ''' Save the data as .npy '''
    Holidays = []
    Holidays_str = np.array( ['2017-01-02','2017-01-16','2017-02-20','2017-05-29','2017-07-04','2017-09-04','2017-10-09','2017-11-10','2017-11-23','2017-12-25','2017-03-17'] , dtype='datetime64[D]' )
    for st in Holidays_str:
        Holidays.append(np.datetime64(st,'D'))
    Holidays = np.array(Holidays)
    np.save('./FeatureData_processed/Holidays',Holidays)

    presenter(Holidays)
