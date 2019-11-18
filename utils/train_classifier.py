import random
import datetime
import argparse
import os
import pickle

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score


datapath = './../data/'
filename = 'training.csv'
file_loc = os.path.join(datapath, filename)

modelpath = './../models/'
percluster = 'classifier_by_cluster/'

def train_per_cluster(data, features):
    # generating samples for each cluster
    clf_container = []

    cluster_max = data['Cluster'].max()

    for i in range(0, cluster_max+1):
        X = data[data['Cluster'] == i][features]
        y = data[data['Cluster'] == i].Accident
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        clf=RandomForestClassifier(n_estimators=100, n_jobs = 6)
        clf.fit(X_train,y_train)
        clf_container.append(clf)
        
        y_pred = clf.predict(X_test)
        
        print("Cluster = ", i)
        print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
        print("ROC score",roc_auc_score(y_test, y_pred))

        _name = 'rf_classf_cluster' + str(i) + '.sav'
        path = os.path.join(modelpath, percluster, _name)
        pickle.dump(clf_container[i], open(path, 'wb'))


def train_entire(data, features):
    X=data[features]  # Features
    y=data.Accident

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    #Create a Gaussian Classifier
    clf=RandomForestClassifier(n_estimators=100, n_jobs=6)

    #Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train,y_train)

    y_pred=clf.predict(X_test)

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    print("ROC score",roc_auc_score(y_test, y_pred))
    print("F1 score",f1_score(y_test, y_pred))

    _name = 'rf_classifier_entire.sav'
    path = os.path.join(modelpath, _name)
    pickle.dump(clf, open(path, 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Random Forest Classifier Training')
    parser.add_argument('--filepath', type=str, default=file_loc)
    parser.add_argument('--train_by_cluster', type=bool, default=False)
    args = parser.parse_args()

    data = pd.read_csv(args.filepath)
    data['Weekday'] = data.apply(lambda row: datetime.date(row.Year, row.Month, row.Day).weekday(), axis = 1)
    data['Start_Lat'] = data.apply(lambda row: row.Cluster_Lat \
                               if np.isnan(row.Start_Lat) \
                               else row.Start_Lat, axis=1)

    data['Start_Lng'] = data.apply(lambda row: row.Cluster_Lng \
                               if np.isnan(row.Start_Lng) \
                               else row.Start_Lng, axis=1)

    features = ['Month', 'Day', 'Day_of_Year', 'Weekday', 'Hour', 'Humidity(%)', 'Temperature(F)', 'Visibility(mi)', 'Wind_Speed(mph)']

    if(args.train_by_cluster):
        train_per_cluster(data, features)
    else:
        train_entire(data, features)

    print("Training Completed!")