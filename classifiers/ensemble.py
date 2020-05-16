"""Class to compare performance with different classifiers"""
import sys

from classifiers.oldGSmote import OldGeometricSMOTE

sys.path.append('../../')
import numpy as np

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
import xgboost as xgb
from classifiers.preprocessing import pre_process as pp
from sklearn.model_selection import GridSearchCV


def evaluate(classifier, index, y_predict):
    # Write output to csv file
    # y_predict.resize(8576,1)
    index.resize(8576, 1)
    data = np.column_stack([index, y_predict])
    label = ["tripid", "prediction"]
    frame = pd.DataFrame(data, columns=label)
    # export_csv = frame.to_csv(r'output/eva_lj.csv', header=True)
    file_path = "../output/evalEnsemble2.csv"
    with open(file_path, mode='w', newline='\n') as f:
        frame.to_csv(f, float_format='%.2f',index=False,header=True)



def KNN():
    # Fitting Simple Linear Regression to the Training set
    # create a dictionary of all values we want to test for n_neighbors
    knn = KNeighborsClassifier()
    # fit model to training data
    knn.fit(X_train, y_train)
    return knn;

def logistic_training():
    # Fitting Simple Logistic Regression to the Training set
    regressor = LogisticRegression()
    regressor.fit(X_train, y_train)

    return regressor


def gradient_boosting():
    # Fitting Gradient boosting
    gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.01, max_depth=3)
    gbc.fit(X_train, y_train)
    return gbc


def XGBoost():
    # Fitting X-Gradient boosting
    gbc = xgb.XGBClassifier(objective="binary:logistic", random_state=42,scale_pos_weight=0.2)
    gbc.fit(X_train, y_train)

    return gbc


def decision_tree():
    # Fitting Simple Linear Regression to the Training set
    regressor = DecisionTreeClassifier()
    regressor.fit(X_train, y_train)

    return regressor;

def getTest(filename):
    df2 = pd.read_csv(filename)
    X1 = df2.iloc[:, 1:].values
    index = np.asarray(df2.iloc[:, 0].values)
    scaler = MinMaxScaler()
    X1 = scaler.fit_transform(X1)
    return X1, index

# data transformation if necessary.
X_t, y_t = pp("../Data/train.csv")
X_test,index =  getTest("../Data/test.csv")

# For SMOTE because of data imbalance
sm = SMOTE(sampling_strategy='auto', k_neighbors=3, random_state=42)
# sm = OldGeometricSMOTE()
X_train, y_train = sm.fit_resample(X_t, y_t)
# X_train, y_train = X_t, y_t

# Trying various classifiers
xgboost= XGBoost()
knn = KNN()
dt = decision_tree()
gbc = gradient_boosting()
lr = logistic_training()

#create a dictionary of our models
estimators=[('knn', knn), ('dt', dt),('gbc',gbc),('lr',lr),('xgboost',xgboost)]
#create our voting classifier, inputting our models
ensemble = VotingClassifier(estimators, voting='hard')

#fit model to training data
ensemble.fit(X_train, y_train)
#test our model on the test data
y_pred = ensemble.predict(X_test)

evaluate("ensemble",index,y_pred)
