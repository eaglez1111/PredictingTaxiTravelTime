import numpy as np
import sys


Mist = { 'Haze':1, 'Mist':2, 'Unknown precipitation':3 }
Fog = { 'Fog':1, 'Freezing (supercooled) fog':2 }
Rain = { 'Light rain':1, 'Rain':2, 'Heavy rain':3, 'Light freezing (supercooled) rain':4, 'Freezing (supercooled) rain':5 }
Snow = { 'Light snow':1, 'Snow':2, 'Heavy snow':3 }
WeatherTypeDict = [Mist, Fog, Rain, Snow]
weather_init = [0,0,0,0,0,0,0,0,0] # [ Humidity, WindSpeed, Visibility, Temperature, Temperature - Mean(Temp.), Mist(&Haze), Fog, Rain, Snow ]
Col = [4,6,11,1]
Col_SpecialCondition = [5,6,7,8]

n_forMeanTemp = 3
Date = np.arange(np.datetime64('2017','D')-3, np.datetime64('2018','D')+3, dtype='datetime64[D]')




def presenter(Weather_dict):
    print("\n\n\n\nPresenting Weather:\n")
    print(type(Weather_dict))
    Time_Example = [ np.datetime64('2017-01-01 00','h'),
            (np.datetime64('2017-07-24 09:11:54','m')+30).astype('datetime64[h]'),
            np.datetime64('2017-12-31 23','h')]
    for time in Time_Example:
        print('\nThe weather at',time,':')
        print(Weather_dict[time])
        print('[ Humidity, WindSpeed, Visibility, Temp., Temp_Diff, Mist&Haze, Fog, Rain, Snow ]')




if __name__ != "__main__" :

    ''' Demo - how to load data '''
    [Weather_dict] = np.load('./FeatureData_processed/Weather_dict.npy',allow_pickle = True)
    print("Weather Data loaded successfully!")

else :
    data = np.loadtxt('./FeatureData_raw/Weather.NYC.2017.csv',delimiter=';', dtype='str',skiprows=7)
    N = len(data)
    Weather = [i[:] for i in [[weather_init] * 24] * (365+n_forMeanTemp*2)]

    lastNum = [0,0,0,0]
    for line in data:
        wth = weather_init.copy()
        for i in [0,1,2,3]: #[ Temperature, Humidity, WindSpeed, Visibility ]
            num = line[Col[i]][1:-1]
            if num=='':
                wth[i] = lastNum[i]
            else:
                wth[i] = float(num)
                lastNum[i] = wth[i]
        for st in line[8][1:-1].split(','):
            if len(st)>1 and st[0]==' ': st=st[1:]
            st = st.capitalize()
            for i in range(len(WeatherTypeDict)):
                if st in WeatherTypeDict[i].keys():
                    wth[Col_SpecialCondition[i]] = WeatherTypeDict[i][st]

        # Date and Hour:
        t = line[0][1:-1]
        t = t[6:10]+'-'+t[3:5]+'-'+t[0:2]+' '+t[11:]
        t = np.datetime64(t,'m')+30
        _date = t.astype('datetime64[D]')
        _hour = t.astype(object).hour
        _ind = np.argwhere(Date==_date)
        if len(_ind)!=0:
            Weather[_ind[0][0]][_hour] = wth
    # np.save('./FeatureData_processed/weather',[weather])

    w = np.array(Weather)
    for j in range(24):
        for i in range(n_forMeanTemp,len(w)-n_forMeanTemp):
            w[i][j][4] = w[i][j][3]-np.mean(w[i-n_forMeanTemp:i+n_forMeanTemp,j][3])
            Weather[i][j][4] = w[i][j][4]

    Weather = Weather[n_forMeanTemp:-n_forMeanTemp]
    Date = Date[n_forMeanTemp:-n_forMeanTemp]
    Weather_dict = {}
    for i in range(365):
        day_h = np.datetime64(Date[i],'h')
        for h in range(24):
            Weather_dict[day_h+h] = Weather[i][h]

    np.save('./FeatureData_processed/Weather_dict',[Weather_dict])



    ''' Print Examples '''
    presenter(Weather_dict)
