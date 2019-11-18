import pandas as pd
import numpy as np
import os
import datetime
import json
import requests
import argparse
import random

datapath = './../data/'
filename = 'PositiveTrainingData.csv'
file_loc = os.path.join(datapath, filename)

apipath = './../api'
apiname = 'darksky_api.txt'
api_loc = os.path.join(apipath, apiname)

def get_api_key():
    # Weather API Key Retrieval

    f = open(api_loc, "r")
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

def get_exact_month_day(year, day_of_year):
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    leap_days = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    leap_year = (year % 4 == 0)
    
    if(day_of_year < 1 or (leap_year and day_of_year > 366) or ((leap_year==False) and day_of_year > 365)): return 0, 0
    
    month = 1
    
    if(day_of_year <= 31): 
        return month, day_of_year
    else:
        if(leap_year != True):
            while(True):
                if(day_of_year > month_days[month-1]):
                    day_of_year -= month_days[month-1]
                    month += 1
                else:
                    break
        else:
            while(True):
                if(day_of_year > leap_days[month-1]):
                    day_of_year -= leap_days[month-1]
                    month += 1
                else:
                    break
    
    return month, day_of_year

def negative_sample_generation(data):
    key = get_api_key()

    accident_features = ['Temperature(F)', 'Humidity(%)', 'Visibility(mi)', 'Wind_Speed(mph)',\
                     'Year', 'Month', 'Day', 'Day_of_Year', 'Hour',\
                     'Cluster', 'Cluster_Lat', 'Cluster_Lng']

    non_accident = pd.DataFrame(columns=accident_features)

    random.seed() #system time as a seed
    min_yr = data['Year'].min()
    max_yr = data['Year'].max()

    cluster_by_size = data['Cluster'].value_counts()

    cluster_num = 0
    while(cluster_num < cluster_by_size.shape[0]):
        # generate smaples cluster by cluster
        cluster_size = cluster_by_size[cluster_num] # e.g.) cluster 0's size = cluster_by_size[0]
        i = 1
        cluster_lat = data[data['Cluster'] == cluster_num]['Cluster_Lat'].unique()[0]
        cluster_lng = data[data['Cluster'] == cluster_num]['Cluster_Lng'].unique()[0]
        print("Cluster # = ", cluster_num)
        print("Cluster Density =", cluster_size)
        print("Cluster Lat = ", cluster_lat)
        print("Cluster Lng = ", cluster_lng)
        
        while (i < cluster_size*2): # generate x3 negative samples for positive samples
            _day = 0
            _hour = random.randint(0,23)
            _year = random.randint(min_yr, max_yr)
            if(_year == 2016):
                _day = random.randint(83,366) # leap year
            elif(_year == 2019):
                _day = random.randint(1, 89)
            else:
                _day = random.randint(1,365)
            
            is_accident = data.loc[(data['Cluster'] == cluster_num) &\
                                (data['Day_of_Year'] == _day) &\
                                (data['Hour'] == _hour) &\
                                (data['Year'] == _year)]
            
            if (is_accident.empty): # no duplice in a cluster at a given a specific time
                _month, _m_day = get_exact_month_day(_year, _day)
                
                print('randomly generated date = ', _year, _month, _m_day, _hour)
                weather_data = get_weather_info(cluster_lat, cluster_lng, _year, _month, _m_day, _hour, key)
                
                try:
                    _temperature = weather_data['currently']['temperature']
                    _humidity = weather_data['currently']['humidity'] * 100
                    _visibility = weather_data['currently']['visibility']
                    _windspeed = weather_data['currently']['windSpeed']
                except KeyError:
                    pass
                
                no_accident = pd.DataFrame([[_temperature, _humidity, _visibility, _windspeed,\
                                        _year, _month, _m_day, _day, _hour,\
                                        cluster_num, cluster_lat, cluster_lng]],\
                                        columns = accident_features)
                non_accident = non_accident.append(no_accident, ignore_index = True)
                i += 1
            else:
                print("duplicate found in cluster:", cluster_num, "on ", _day, "th day at ", _hour, "th hour")
        
        print("Cluster # = ", cluster_num, " finished")
        cluster_num += 1 # go to the next cluster

    return non_accident


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Negative Sample Generation')
    parser.add_argument('--filepath', type=str, default=file_loc)
    parser.add_argument('--apipath', type=str, default=api_loc)
    args = parser.parse_args()

    data = pd.read_csv(args.filepath)
    non_accident = negative_sample_generation(data)

    data['Accident'] = np.ones(data.shape[0]).tolist()
    non_accident['Accident'] = np.zeros(non_accident.shape[0]).tolist()
    data = data.append(non_accident)

    training = 'training.csv'
    save_path = os.path.join(datapath, training)
    print('Whole Training Data Generation Completed!')
    print('Saving training data at', save_path)
    data.to_csv(save_path, index = None, header=True)