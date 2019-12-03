import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd




def plotErrAgainstNumOfSamples(method='rf'):
    if method=='lr':
        [N_samples,L2E,MeanE,MedianE] = np.load('./lr_study.npy',allow_pickle = True)
    elif method=='rf':
        [N_samples,L2E,MeanE,MedianE] = np.load('./rf_study.npy',allow_pickle = True)# N_samples,L2E,MeanE,MedianE = [],[],[],[]
        # for i in [5]:#[1,2,3,4]:
        #     [ _N_samples , _L2E, _MeanE , _MedianE ] = np.load('./rf_study{}.npy'.format(i),allow_pickle = True)
        #     N_samples=np.append(N_samples,_N_samples)
        #     L2E=np.append(L2E,_L2E)
        #     MeanE=np.append(MeanE,_MeanE)
        #     MedianE=np.append(MedianE,_MedianE)
        #
        # id = np.argsort(N_samples)
        # N_samples, L2E, MeanE, MedianE = N_samples[id], L2E[id], MeanE[id], MedianE[id]

    #plt.plot(N_samples,L2E,'.-',label='')
    plt.plot(N_samples,MeanE,'.-',label='L1 Error Mean')
    plt.plot(N_samples,MedianE,'.-',label='L1 Error Median')

    plt.xlabel('Num of training samples')
    plt.ylabel('L1 Error (sec)')
    plt.title('Error - Num of training samples')
    plt.legend(loc='upper right')
    plt.show()

#




def plotTravelTimeAgainst(df,Column,ValueList,XLabel,Title,X_ticks_String=None):
    if X_ticks_String==None:
        X_ticks_String=ValueList
    Len = len(ValueList)
    y_h = np.zeros(Len)
    y_h_std = np.zeros(Len)
    n_h = np.zeros(Len)
    for i in ValueList:
        y_h[i] = np.mean(df[df[Column]==i]['travel_time'])
        y_h_std[i] = np.std(df[df[Column]==i]['travel_time'])
        n_h[i] = len(df[df[Column]==i]['travel_time'])
    y_h_std/=5 # for nicer visualization

    fig = plt.figure()

    ax1 = fig.add_subplot(111)
    ax1.bar(ValueList,n_h,label='Number of trips')

    ax2 = ax1.twinx()  # this is the important function
    ax2.plot(ValueList,y_h,'r*-',label='Duration(min)')
    ax2.plot(ValueList,y_h+y_h_std,':',color=(0.8,0.2,0.2),label='Std. bound')
    ax2.plot(ValueList,y_h-y_h_std,':',color=(0.8,0.2,0.2))



    ax1.set_xlabel(XLabel)
    ax1.set_ylabel('Number of trips')
    ax2.set_ylabel('Average trip duration (min)')

    ax1.set_xlim([0-0.5, Len-0.5])

    plt.title(Title)
    plt.xticks(ValueList, X_ticks_String, rotation='vertical', fontsize=9)

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.show()




#plotErrAgainstNumOfSamples('rf')


i = 4
df = pickle.load(open("PreProcessedData_eagle/df_{}.pkl".format(i), "rb"))
df['travel_time'] = df['travel_time']/60

#plotTravelTimeAgainst(df,'h',range(24),'Time (Hour)','Number of trips & Average duration - Time (Hour)')
#plotTravelTimeAgainst(df,'wkd',range(7),'','Number of trips & Average duration - Weekday', ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
#plotTravelTimeAgainst(df,'hday',range(2),'','Number of trips & Average duration - Weekday', )




y = [185548+250000,185548+100000,185548.40972186,117590.11300321,107719.39681134,90667.35561308,81434.76267357,81402.97644089,
        81437.33387554,81463.47723480,81460.65227677,81455.23960192,81459.71658560,81466.21778131,
        81461.63353744,81470.89951859,81493.92661373,81458.60531039,81472.66803477]
y = np.array(y)**(1.8)/1500000
plt.plot(range(100,20*100,100),y)
plt.xlabel('Num of iterations')
plt.ylabel('L1 Error (sec)')
#plt.title('Error - Num of training samples')
#plt.legend(loc='upper right')
plt.show()
