import random
import datetime
import argparse
import os
import pickle

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

datapath = './../data/'
filename = 'training.csv'
file_loc = os.path.join(datapath, filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cluster accidents using DBSCAN and generate positive samples for training')
    parser.add_argument('--filepath', type=str, default=file_loc)
    parser.add_argument('--cluster_distance', type=float, default=0.001)
    parser.add_argument('--cluster_samples', type=int, default=20)
    args = parser.parse_args()

    data = pd.read_csv(args.filepath)