"""Class to pre-process the input data"""
import sys
import numpy as np
import random
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

sys.path.append('../../')

def pre_process(filename):
    df = pd.read_csv(filename)
    # df = df.drop_duplicates(df.iloc[:, :-1].columns, keep='last')
    df.fillna(df.mean(), inplace=True)

    X = np.asarray(df.iloc[:, :-1].values)
    y = np.asarray(df.iloc[:, -1].values)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    return X, y

