"""Class to compare performance with different classifiers"""
import sys

sys.path.append('../../')
import numpy as np

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from classifiers.preprocessing import pre_process as pp


def evaluate(classifier, index, y_predict):
    # Write output to csv file
    # y_predict.resize(8576,1)
    index.resize(8576, 1)
    data = np.column_stack([index, y_predict])
    label = ["tripid", "prediction"]
    frame = pd.DataFrame(data, columns=label)
    # export_csv = frame.to_csv(r'output/eva_lj.csv', header=True)
    file_path = "../output/eval3.csv"
    with open(file_path, mode='w', newline='\n') as f:
        frame.to_csv(f, float_format='%.2f',index=False,header=True)

def logistic_training():
    # Fitting Simple Logistic Regression to the Training set
    regressor = LogisticRegression()
    regressor.fit(X_train, y_train)

    # Predicting the Test set results
    y_predict = regressor.predict(X_test)
    y_pred = np.where(y_predict > 0.5, 1, 0)

    return evaluate("Logistic Regression", y_pred)


def gradient_boosting():
    # Fitting Gradient boosting
    gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.01, max_depth=3)
    gbc.fit(X_train, y_train)

    # Predicting the Test set results
    y_predict = gbc.predict(X_test)
    y_pred = np.where(y_predict.astype(int) > 0.5, 1, 0)

    return evaluate("Gradient Boosting", y_pred)


def XGBoost():
    # Fitting X-Gradient boosting
    gbc = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
    gbc.fit(X_train, y_train)

    # Predicting the Test set results
    y_predict = gbc.predict(X_test)
    y_pred = np.where(y_predict.astype(int) > 0.5, 1, 0)

    evaluate("XGBoost", index, y_pred)


def KNN():
    # Fitting Simple Linear Regression to the Training set
    classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test).astype(int)

    return evaluate("KNN", y_pred)


def decision_tree():
    # Fitting Simple Linear Regression to the Training set
    regressor = DecisionTreeRegressor()
    regressor.fit(X_train, y_train)

    # Predicting the Test set results
    y_predict = regressor.predict(X_test)
    y_pred = np.where(y_predict > 0.5, 1, 0)

    return evaluate("Decision Tree", y_pred)


# def GaussianMixture_model():
#     from sklearn.mixture import GaussianMixture
#     gmm = GaussianMixture(n_components=1)
#     gmm.fit(X_train[y_train == 0])
#
#     OKscore = gmm.score_samples(X_train[y_train == 0])
#     threshold = OKscore.mean() - 1 * OKscore.std()
#
#     score = gmm.score_samples(X_test)
#
#     # majority_correct = len(score[(y_test == 1) & (score > thred)])
#     y_pred = np.where(score < threshold, 1, 0)
#     return evaluate("GaussianMixture_model", y_pred)

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
X_train, y_train = sm.fit_resample(X_t, y_t)

# Trying various classifiers
XGBoost()
# GaussianMixture_model()
# KNN()
# decision_tree()
# gradient_boosting()
# logistic_training()
