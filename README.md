# PredictingTaxiTravelTime
<Large Data: Predicting Taxi Travel Time>


1.
Download the TaxiData from https://drive.google.com/file/d/1Wi1HZItFGBwaCq8cTwjxk_GvUP0p9mR-/view?usp=sharing
Extract it at the project folder, so that, for example, a csv file's path is: '.../PredictingTaxiTravelTime/TaxiData/assignment1_data-1.csv'
File ".gitinore" is set, so we can just keep TaxiData in your local folder, and these data won't be committed or pushed.


2.
For the script, make sure Pandas is installed on your computer, and FYI, I have been using python3 (3.6):
$ pip3 install pandas


3.
To see taxi zone data reload: under the folder "......./PredictingTaxiTravelTime", run:
...../PredictingTaxiTravelTime $ python3 Tools/processLocation.py --demo

Read Tools/processLocation.py to see how to access taxi zone data.


4.
To see Weather data reload: under the folder "......./PredictingTaxiTravelTime", run:
...../PredictingTaxiTravelTime $ python3 -i Tools/processWeather.py --demo

You should see:

<class 'dict'>

The weather at 2017-01-01T00 :
[0, 0, 0, 0, 0.0, 0, 0, 0, 0]
[ Humidity, WindSpeed, Visibility, Temp., Temp_Diff, Mist&Haze, Fog, Rain, Snow ]

The weather at 2017-07-24T09 :
[94.0, 5.0, 6.4, 19.0, 4.844444444444443, 2, 0, 1, 0]
[ Humidity, WindSpeed, Visibility, Temp., Temp_Diff, Mist&Haze, Fog, Rain, Snow ]

The weather at 2017-12-31T23 :
[52.0, 5.0, 16.0, -12.0, -18.77777777777778, 0, 0, 0, 0]
[ Humidity, WindSpeed, Visibility, Temp., Temp_Diff, Mist&Haze, Fog, Rain, Snow ]
