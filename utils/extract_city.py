import os
import argparse
import pandas as pd

datapath = './../data/'
filename = 'US_Accidents_May19.csv'
file_loc = os.path.join(datapath, filename)

def extract_city(city, state, filepath = file_loc):
    print("Reading a CSV File from ... ", filepath)
    print("It may take some time if the file is very large.")
    data = pd.read_csv(file_loc)
    print(filepath, " Loading completed")

    city_data = data.loc[data['State'] == state]
    city_data = city_data.loc[data['City'] == city]
    cityname = city + '_' + state + '.csv'
    citypath = os.path.join(datapath, cityname)
    print("Saving file into ... ", citypath)
    city_data.to_csv(citypath, index=None, header=True)
    print("Extraction Completed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract All Accidents of a Certain City in US')
    parser.add_argument('--city', type=str, default='San Jose')
    parser.add_argument('--state', type=str, default='CA')
    parser.add_argument('--filepath', type=str, default=file_loc)
    args = parser.parse_args()

    extract_city(args.city, args.state, args.filepath)