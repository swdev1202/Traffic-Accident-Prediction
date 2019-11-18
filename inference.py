import json
import requests
import pickle
import datetime
import pandas as pd
import numpy as np
import gmaps
import os

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Random Forest Classifier Training')
    parser.add_argument('--filepath', type=str, default=file_loc)
    parser.add_argument('--train_by_cluster', type=bool, default=False)
    args = parser.parse_args()