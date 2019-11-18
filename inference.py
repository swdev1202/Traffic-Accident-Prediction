import json
import requests
import pickle
import datetime
import argparse
import os

import pandas as pd
import numpy as np

classifier_path = './../models/'
filename = 'rf_classifier.sav'

weather_api_path = './../api/darksky_api.txt'

cluster_info = './../data/cluster_info.csv'

def get_api_key(api_key_loc):
    f = open(api_key_loc, "r")
    key = f.readline()
    f.close()

    return key

def get_weather_info(lat, lng, year, month, day, hr, key):
    dt = datetime.date(year, month, day)
    str_hr = ""
    if(hr < 10):
        str_hr = "0"+str(hr)
    else:
        str_hr = str(hr)
    tm = dt.strftime("%Y-%m-%d") + "T" + str_hr + ":00:01"  # get the exact hour's forecast

    url = "https://api.darksky.net/forecast/" + key + \
    "/" + str(lat) + "," + str(lng) + "," + tm + \
    "?exclude=minutely,flags,daily,alerts,hourly"
    
    response = requests.get(url)
    data = response.json()
    return data

def calculate_distance(src_lat, src_lng, cluster):
    cluster['Distance'] = cluster.apply(lambda row: \
                                        np.sqrt(np.square(row.Latitude - src_lat) + np.square(row.Longitude - src_lng)),\
                                        axis=1)
    
    return cluster

def single_mode_predict(lat, lng):
    rf_model = pickle.load(open(os.path.join(classifier_path, filename), 'rb'))
    curr_time = datetime.datetime.now()
    weather_key = get_api_key(weather_api_path)
    curr_weather = get_weather_info(lat, lng, curr_time.year, curr_time.month, curr_time.day, curr_time.hour, weather_key)

    features = ['Year', 'Month', 'Day', 'Day_of_Year', 'Weekday', 'Hour',\
            'Humidity(%)', 'Temperature(F)', 'Visibility(mi)', 'Wind_Speed(mph)']

    _year = curr_time.year
    _month = curr_time.month
    _day = curr_time.day
    _day_of_year = curr_time.timetuple().tm_yday
    _weekday = curr_time.weekday()
    _hour = curr_time.hour

    _humidity = curr_weather['currently']['humidity']
    _temperature = curr_weather['currently']['temperature']
    _visibility = curr_weather['currently']['visibility']
    _windspeed = curr_weather['currently']['windSpeed']

    X = pd.DataFrame([[_year, _month, _day, _day_of_year, _weekday, _hour,\
                    _humidity, _temperature, _visibility, _windspeed]],\
                    columns = features)

    return (rf_model.predict(X) == 1.0)

def find_nearest_hotspot(lat, lng):
    hotspots = pd.read_csv(cluster_info)
    hotspots = calculate_distance(lat, lng, hotspots)
    hotspots = hotspots.sort_values(by=['Distance'])

    return hotspots.iloc[0:1][['Latitude', 'Longitude']]

def multi_cluster_predict(lat, lng):
    hotspots = pd.read_csv(cluster_info)
    hotspots = calculate_distance(lat, lng, hotspots)
    
    activated_hotspots = pd.DataFrame(columns=hotspots.columns)

    for i in range(0, hotspots.shape[0]):
        curr_hotspot = hotspots.iloc[i]
        
        filepath = './../models/classifier_by_cluster/'
        filename = 'rf_classf_cluster' + str(curr_hotspot['Cluster'].astype(int)) + '.sav'
        path = os.path.join(filepath, filename)
        rf_model = pickle.load(open(path, 'rb'))
        
        result = rf_model.predict(X)[0]
        if(result == True):
            act = pd.DataFrame([[curr_hotspot.Cluster.astype(int), curr_hotspot.Longitude,\
                                curr_hotspot.Latitude, curr_hotspot.Distance]],\
                                columns = hotspots.columns)
            activated_hotspots = activated_hotspots.append(act, ignore_index = True)
    
    return activated_hotspots



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Accident Prediction')
    parser.add_argument('--single_mode', type=bool, default=True)
    parser.add_argument('--lat', type=float, default=37.336431)
    parser.add_argument('--lng', type=float, default=-121.883980)

    args = parser.parse_args()

    if(args.single_mode):
        result = single_mode_predict(args.lat, args.lng)
        print("Will an accident happen? ", result)

        if(result):
            hotspot = find_nearest_hotspot(args.lat, args.lng)
            print("Nearest Hotspot from my location = ", hotspot)
    else:
        activated = multi_cluster_predict(args.lat, args.lng)
        activated = calculate_distance(args.lat, args.lng, activated)
        activated = activated.sort_values(by=['Distance'])

        print("Out of all activated clusters, the closest location is = ", activated.iloc[0:1][['Latitude', 'Longitude']])