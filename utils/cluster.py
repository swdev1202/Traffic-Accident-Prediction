import os
import argparse
import datetime
import pandas as pd
import numpy as np

from sklearn.cluster import DBSCAN

datapath = './../data/'
filename = 'San Jose_CA.csv'
file_loc = os.path.join(datapath, filename)

def parse_time(data):
    pd.set_option('mode.chained_assignment', None)
    data['Year'] = data.apply(lambda row: int(row.Start_Time.split(' ')[0].split('-')[0]), axis = 1)
    data['Month'] = data.apply(lambda row: int(row.Start_Time.split(' ')[0].split('-')[1]), axis = 1)
    data['Day'] = data.apply(lambda row: int(row.Start_Time.split(' ')[0].split('-')[2]), axis = 1)
    data['Hour'] = data.apply(lambda row: int(row.Start_Time.split(' ')[1].split(':')[0]), axis = 1)
    data['Weekday'] = data.apply(lambda row: datetime.date(row.Year, row.Month, row.Day).weekday(), axis = 1)
    data['Day_of_Year'] = data.apply(lambda row: datetime.date(row.Year, row.Month, row.Day).timetuple().tm_yday, axis=1)
    return data

def cleanup(data):
    pd.set_option('mode.chained_assignment', None)
    data['Zipcode'] = data.apply(lambda row: row.Zipcode.split('-')[0], axis=1)

    # fill the temperature with the month's average
    temp_monthly_avg = data.groupby(['Month']).mean()['Temperature(F)']
    data['Temperature(F)'] = data.apply(lambda row: \
                                        temp_monthly_avg.iloc[row.Month-1] \
                                        if np.isnan(row['Temperature(F)']) \
                                        else row['Temperature(F)'], axis=1)

    # fill the humidity with the month's average
    humidity_monthly_avg = data.groupby(['Month']).mean()['Humidity(%)']
    data['Humidity(%)'] = data.apply(lambda row: \
                                    humidity_monthly_avg.iloc[row.Month-1] \
                                    if np.isnan(row['Humidity(%)']) \
                                    else row['Humidity(%)'], axis=1)

    # fill the visibility with the month's average
    visibility_monthly_avg = data.groupby(['Month']).mean()['Visibility(mi)']
    data['Visibility(mi)'] = data.apply(lambda row: \
                                        visibility_monthly_avg.iloc[row.Month-1] \
                                        if np.isnan(row['Visibility(mi)']) \
                                        else row['Visibility(mi)'], axis=1)
    
    # fill the wind speed with the month's average
    windspeed_monthly_avg = data.groupby(['Month']).mean()['Wind_Speed(mph)']
    data['Visibility(mi)'] = data.apply(lambda row: \
                                        windspeed_monthly_avg.iloc[row.Month-1] \
                                        if np.isnan(row['Visibility(mi)']) \
                                        else row['Visibility(mi)'], axis=1)

    # drop NULL weather conditions
    data = data.dropna(subset=['Weather_Condition'])

    return data

def create_cluster(data, cluster_d, cluster_samples):
    location = ['Start_Lat', 'Start_Lng']
    loc_data = data[location]

    db = DBSCAN(eps=cluster_d, min_samples=cluster_samples).fit(loc_data)

    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print('Estimated number of clusters: %d' % n_clusters_)
    n_noise_ = list(labels).count(-1)
    print('Estimated number of noise points: %d' % n_noise_)

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    cluster_loc = []
    cluster_mean = {}

    unique_labels = set(labels)
    for k in unique_labels:
        if k != -1:
            class_member_mask = (labels == k)
            xy = loc_data[class_member_mask & core_samples_mask]
            cluster_loc.append((xy.iloc[:, 1].mean(), xy.iloc[:, 0].mean(), xy.shape[0]))
            cluster_mean[k] = [xy.iloc[:, 1].mean(), xy.iloc[:, 0].mean()]

    return cluster_mean, db.labels_
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cluster accidents using DBSCAN and generate positive samples for training')
    parser.add_argument('--filepath', type=str, default=file_loc)
    parser.add_argument('--cluster_distance', type=float, default=0.001)
    parser.add_argument('--cluster_samples', type=int, default=20)
    args = parser.parse_args()

    data = pd.read_csv(args.filepath)
    features = ['Start_Time', 'Start_Lat', 'Start_Lng', 'Zipcode', 'Temperature(F)', 'Humidity(%)',\
            'Visibility(mi)', 'Wind_Speed(mph)', 'Weather_Condition']
    selected_data = data[features]
    selected_data = parse_time(selected_data)
    selected_data = cleanup(selected_data)

    clust_mean, labels = create_cluster(selected_data, args.cluster_distance, args.cluster_samples)
    selected_data['Cluster'] = labels.tolist()
    selected_data.drop(selected_data[selected_data['Cluster'] == -1].index , inplace=True)

    # store cluster_lat, cluster_lng from cluster_mean
    selected_data['Cluster_Lat'] = selected_data.apply(lambda row: clust_mean[row.Cluster][1], axis=1)
    selected_data['Cluster_Lng'] = selected_data.apply(lambda row: clust_mean[row.Cluster][0], axis=1)
    
    pos_filename = 'PositiveTrainingData.csv'
    save_path = os.path.join(datapath, pos_filename)
    print('Cluster Generation Completed!')
    print('Saving positive training samples at', save_path)
    selected_data.to_csv(save_path, index = None, header=True)