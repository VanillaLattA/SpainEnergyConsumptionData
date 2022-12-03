import pandas as pd
import numpy as np

Weather = pd.read_csv('../../data/weather_features.csv', skipinitialspace=True)
EnergySub = pd.read_csv('../../data/energy_dataset.csv')

EnergySub = EnergySub[EnergySub['total load actual'].notna()]

WV = {
    'tempV': np.array(Weather.loc[Weather['city_name'] == 'Valencia', 'temp']),
    'humV': np.array(Weather.loc[Weather['city_name'] == 'Valencia', 'humidity']),
    'cldV': np.array(Weather.loc[Weather['city_name'] == 'Valencia', 'clouds_all']),
    'wndV': np.array(Weather.loc[Weather['city_name'] == 'Valencia', 'wind_speed']),
    'timeWV': np.array(Weather.loc[Weather['city_name'] == 'Valencia', 'dt_iso']),
}
WM = {
    'tempM': np.array(Weather.loc[Weather['city_name'] == 'Madrid', 'temp']),
    'humM': np.array(Weather.loc[Weather['city_name'] == 'Madrid', 'humidity']),
    'cldM': np.array(Weather.loc[Weather['city_name'] == 'Madrid', 'clouds_all']),
    'wndM': np.array(Weather.loc[Weather['city_name'] == 'Madrid', 'wind_speed']),
    'timeWM': np.array(Weather.loc[Weather['city_name'] == 'Madrid', 'dt_iso']),
}
WBi = {
    'tempBi': np.array(Weather.loc[Weather['city_name'] == 'Bilbao', 'temp']),
    'humBi': np.array(Weather.loc[Weather['city_name'] == 'Bilbao', 'humidity']),
    'cldBi': np.array(Weather.loc[Weather['city_name'] == 'Bilbao', 'clouds_all']),
    'wndBi': np.array(Weather.loc[Weather['city_name'] == 'Bilbao', 'wind_speed']),
    'timeWBi': np.array(Weather.loc[Weather['city_name'] == 'Bilbao', 'dt_iso']),
}
WBa = {
    'tempBa': np.array(Weather.loc[Weather['city_name'] == 'Barcelona', 'temp']),
    'humBa': np.array(Weather.loc[Weather['city_name'] == 'Barcelona', 'humidity']),
    'cldBa': np.array(Weather.loc[Weather['city_name'] == 'Barcelona', 'clouds_all']),
    'wndBa': np.array(Weather.loc[Weather['city_name'] == 'Barcelona', 'wind_speed']),
    'timeWBa': np.array(Weather.loc[Weather['city_name'] == 'Barcelona', 'dt_iso']),
}
WS = {
    'tempS': np.array(Weather.loc[Weather['city_name'] == 'Seville', 'temp']),
    'humS': np.array(Weather.loc[Weather['city_name'] == 'Seville', 'humidity']),
    'cldS': np.array(Weather.loc[Weather['city_name'] == 'Seville', 'clouds_all']),
    'wndS': np.array(Weather.loc[Weather['city_name'] == 'Seville', 'wind_speed']),
    'timeWS': np.array(Weather.loc[Weather['city_name'] == 'Seville', 'dt_iso']),
}
WV = pd.DataFrame(WV)
WM = pd.DataFrame(WM)
WBi = pd.DataFrame(WBi)
WBa = pd.DataFrame(WBa)
WS = pd.DataFrame(WS)

for i in range(len(EnergySub)):
    if (WV.iloc[i]['timeWV'] != EnergySub.iloc[i]['time']):
        WV.drop(i, inplace=True, axis=0)
        i -= 1
for i in range(len(EnergySub)):
    if (WM.iloc[i]['timeWM'] != EnergySub.iloc[i]['time']):
        WM.drop(i, inplace=True, axis=0)
        i -= 1
for i in range(len(EnergySub)):
    if (WBi.iloc[i]['timeWBi'] != EnergySub.iloc[i]['time']):
        WBi.drop(i, inplace=True, axis=0)
        i -= 1
for i in range(len(EnergySub)):
    if (WBa.iloc[i]['timeWBa'] != EnergySub.iloc[i]['time']):
        WBa.drop(i, inplace=True, axis=0)
        i -= 1
for i in range(len(EnergySub)):
    if (WS.iloc[i]['timeWS'] != EnergySub.iloc[i]['time']):
        WS.drop(i, inplace=True, axis=0)
        i -= 1

WeatherSub = pd.DataFrame ({
    'time': list(WV['timeWV']),
    'tempV': list(WV['tempV']),
    'tempM': list(WM['tempM']),
    'tempBi': list(WBi['tempBi']),
    'tempBa': list(WBa['tempBa']),
    'tempS': list(WS['tempS']),
    'humV': list(WV['humV']),
    'humM': list(WM['humM']),
    'humBi': list(WBi['humBi']),
    'humBa': list(WBa['humBa']),
    'humS': list(WS['humS']),
    'cldV': list(WV['cldV']),
    'cldM': list(WM['cldM']),
    'cldBi': list(WBi['cldBi']),
    'cldBa': list(WBa['cldBa']),
    'cldS': list(WS['cldS']),
    'wndV': list(WV['wndV']),
    'wndM': list(WM['wndM']),
    'wndBi': list(WBi['wndBi']),
    'wndBa': list(WBa['wndBa']),
    'wndS': list(WS['wndS']),
})

WeatherSub.to_csv('AllTemps.csv')