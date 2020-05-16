"""Class to pre-process the input data"""
import sys
import numpy as np
import random
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
import math

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

sys.path.append('../../')


def pre_process(filename):
    df = pd.read_csv(filename)
    # df = df.drop_duplicates(df.iloc[:, :-1].columns, keep='last')
    df.fillna(df.mean(), inplace=True)

    X = df.iloc[:, 1:-1].values
    y = np.asarray(df.iloc[:, -1].values)

    df.insert(11, 'distance',df.apply(lambda row: find_distance(row.drop_lon, row.drop_lat, row.pick_lon, row.pick_lat),axis=1))

    # df['distance'] = df.apply(lambda row: find_distance(row.drop_lon, row.drop_lat, row.pick_lon, row.pick_lat),axis=1)

    df.to_csv("train_distance.csv")

    scaler = StandardScaler()
    X = scaler.fit_transform(X)


    return X, y

def pre_processTest(filename):
    df2 = pd.read_csv(filename)
    X1 = df2.iloc[:, 1:].values
    index = np.asarray(df2.iloc[:, 0].values)

    df2['distance'] = df2.apply(lambda row: find_distance(row.drop_lon, row.drop_lat, row.pick_lon, row.pick_lat),axis=1)

    df2.to_csv("test_distance.csv")
    # print (df2.head())
    scaler = MinMaxScaler()
    X1 = scaler.fit_transform(X1)

    return X1, index

def pre_process_initially(filename):
    df = pd.read_csv(filename)
    # df = df.drop_duplicates(df.iloc[:, :-1].columns, keep='last')
    df.fillna(df.mean(), inplace=True)

    X = df.iloc[:, 1:-1].values
    y = np.asarray(df.iloc[:, -1].values)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    # print(X[0:20])

    return X, y

def pre_process_init_test(filename):
    df2 = pd.read_csv(filename)
    X1 = df2.iloc[:, 1:].values
    index = np.asarray(df2.iloc[:, 0].values)
    scaler = MinMaxScaler()
    X1 = scaler.fit_transform(X1)
    return X1, index

def find_distance(lon1, lat1, lon2, lat2):
    R = 6371000  # radius of Earth in meters
    phi_1 = math.radians(lat1)
    phi_2 = math.radians(lat2)

    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + \
        math.cos(phi_1) * math.cos(phi_2) * \
        math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # in KM
    m = (R * c)/1000

    return m


# # pre_process("../Data/train_org.csv")
# pre_processTest("../Data/

import datetime
from time import mktime
import time
def find_time(timestamp):
    # print(timestamp)
    ts = pd.to_datetime(timestamp).hour

    if(ts>=0 and ts<=6):
        return 0
    elif(ts>6 and ts<=10):
        return 1
    elif (ts>10 and ts<=12):
        return 2
    elif(ts>12 and ts<=16):
        return 3
    elif(ts>16 and ts<=21):
        return 4
    elif (ts>21 and ts <=24):
        return 5
    else:
        return 6
    # return ts

def find_day(timestamp):
    # print(timestamp)
    datetime = pd.to_datetime(timestamp)
    return datetime.weekday()

def preProcess_date(filename):

    df = pd.read_csv(filename)
    df.fillna(df.mean(), inplace=True)

    X = df.iloc[:, 1:-1].values
    y = np.asarray(df.iloc[:, -1].values)


    df['time'] = df.apply(lambda row: find_time(row.pickup_time), axis=1)
    # df['day'] = df.apply(lambda row: find_day(row.pickup_time), axis=1)

    df.to_csv("interim.csv")

# find_day('11/1/2019 0:20')
# preProcess_date("../Data/test_date.csv")

def processOneHotTrain (filename):
    df = pd.read_csv(filename)
    # df = df.drop_duplicates(df.iloc[:, :-1].columns, keep='last')
    df.fillna(df.mean(), inplace=True)

    X = df.iloc[:, 1:-1].values
    y = np.asarray(df.iloc[:, -1].values)

    # columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(drop='first'), [5,6])], remainder='passthrough')
    # X = np.array(columnTransformer.fit_transform(X), dtype=np.str)

    # columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(drop='first'), [6])], remainder='passthrough')
    # X = np.array(columnTransformer.fit_transform(X), dtype=np.str)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    return X, y

def processOneHotTest(filename):
    df2 = pd.read_csv(filename)
    X1 = df2.iloc[:, 1:].values
    index = np.asarray(df2.iloc[:, 0].values)

    # columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(drop='first'), [5,6])], remainder='passthrough')
    # X1 = np.array(columnTransformer.fit_transform(X1), dtype=np.str)

    # columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(drop='first'), [6])], remainder='passthrough')
    # X1 = np.array(columnTransformer.fit_transform(X1), dtype=np.str)

    scaler = MinMaxScaler()
    X1 = scaler.fit_transform(X1)
    return X1, index