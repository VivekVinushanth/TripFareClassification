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

    df.insert(11, 'distance',
              df.apply(lambda row: find_distance(row.drop_lon, row.drop_lat, row.pick_lon, row.pick_lat), axis=1))

    # df['distance'] = df.apply(lambda row: find_distance(row.drop_lon, row.drop_lat, row.pick_lon, row.pick_lat),axis=1)

    df.to_csv("train_distance.csv")

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y


def pre_processTest(filename):
    df2 = pd.read_csv(filename)
    X1 = df2.iloc[:, 1:].values
    index = np.asarray(df2.iloc[:, 0].values)

    df2['distance'] = df2.apply(lambda row: find_distance(row.drop_lon, row.drop_lat, row.pick_lon, row.pick_lat),
                                axis=1)

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


from math import radians, sin, cos, asin, sqrt


def find_distance(lon1, lat1, lon2, lat2):
    R = 6371  # radius of Earth in Kilometers
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))

    # in KM
    m = R * c

    return m


# # pre_process("../Data/train_org.csv")
# pre_processTest("../Data/

import datetime
from time import mktime
import time


def find_time(timestamp):
    # print(timestamp)
    ts = pd.to_datetime(timestamp).hour
    # # print(ts)
    # return ts

    if (ts >= 0 and ts <= 6):
        return 0
    elif (ts > 6 and ts <= 12):
        return 1
    elif (ts > 12 and ts <= 18):
        return 3
    elif (ts > 18 and ts <= 24):
        return 4
    else:
        return 6
    return ts


def find_day(timestamp):
    # print(timestamp)
    datetime = pd.to_datetime(timestamp)
    # print(datetime.weekday())
    return datetime.weekday()


def preProcess_DTDi(filename):
    df = pd.read_csv(filename)
    # df.fillna(df.mean(), inplace=True)
    df.dropna()

    # df['time'] = df.apply(lambda row: find_time(row.pickup_time), axis=1)
    # df['day'] = df.apply(lambda row: find_day(row.pickup_time), axis=1)
    df['distance'] = df.apply(lambda row: find_distance(row.drop_lon, row.drop_lat, row.pick_lon, row.pick_lat), axis=1)

    df.to_csv("temp_test.csv")


# find_day('18/1/2019 0:20')
# find_time('11/1/2019 23:20')
# preProcess_DTDi("../Data/test.csv")


def processOneHotTrain(filename):
    df = pd.read_csv(filename)
    # df = df.drop_duplicates(df.iloc[:, :-1].columns, keep='last')

    # df.fillna(df.mean(), inplace=True)

    X = df.iloc[:, 1:-1].values
    y = np.asarray(df.iloc[:, -1].values)

    df.drop(['tripid'], axis=1, inplace=True)
    df.dropna(axis=0, how='any', inplace=True)
    # df.drop(['time'], axis=1, inplace=True)
    # df.drop(['day'], axis=1, inplace=True)

    print(df.columns)

    # encode for time
    # columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(drop='first'), [5])], remainder='passthrough')
    # X = np.array(columnTransformer.fit_transform(X), dtype=np.str)

    # scaler = MinMaxScaler()
    # X = scaler.fit_transform(X)

    return X, y


def processOneHotTest(filename):
    df2 = pd.read_csv(filename)
    X1 = df2.iloc[:, 1:].values
    index = np.asarray(df2.iloc[:, 0].values)

    df2.drop(['tripid'], axis=1, inplace=True)
    df2.dropna(axis=0, how='any', inplace=True)

    # df2.drop(['time'], axis=1, inplace=True)
    # df2.drop(['day'], axis=1, inplace=True)

    # # encode for time
    # columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(drop='first'), [5])], remainder='passthrough')
    # X1 = np.array(columnTransformer.fit_transform(X1), dtype=np.str)

    # scaler = MinMaxScaler()
    # X1 = scaler.fit_transform(X1)
    print(df2.columns)
    return X1, index
