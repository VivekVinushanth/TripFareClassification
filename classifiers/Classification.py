"""Class to compare performance with different classifiers"""
import sys

# from classifiers.eg_smote import EGSmote
from classifiers.gsomClassifier import GSOMClassifier
from classifiers.oldGSmote import OldGeometricSMOTE

sys.path.append('../../')
import numpy as np

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from classifiers.preprocessing import pre_process, pre_processTest, pre_process_initially, pre_process_init_test, \
    processOneHotTest, processOneHotTrain
from sklearn.model_selection import GridSearchCV


def evaluate(classifier, index, y_predict):
    # Write output to csv file
    # y_predict.resize(8576,1)
    index.resize(8576, 1)
    data = np.column_stack([index, y_predict])
    label = ["tripid", "prediction"]
    frame = pd.DataFrame(data, columns=label)
    # export_csv = frame.to_csv(r'output/eva_lj.csv', header=True)
    file_path = "../output/xg-new-dis-runtime-fare_no_scale.csv"
    with open(file_path, mode='w', newline='\n') as f:
        frame.to_csv(f, float_format='%.2f', index=False, header=True)


def XGBoost():
    # Fitting X-Gradient boosting
    gbc = xgb.XGBClassifier(objective="binary:logistic", seed=27, learning_rate= 0.01, n_estimators=5000,
                            max_depth=4, min_child_weight=6, gamma=0, subsample=0.8, colsample_bylevel=0.8,
                            reg_alpha=0.0005, scale_pos_weight=1,thread=4)

    # parameters = {
    #     'scale_pos_weight': [0.1, 0.5, 0.9, 1.0], 'max_depth': range(3, 10, 2), 'min_child_weight': range(1, 6, 2),
    #     'learning_rate': [0.0001, 0.001, 0.01, 0.1], 'n_estimators': [100, 200, 300, 400, 500]
    # }

    parameters = {
        'max_depth': range(3, 10, 2), 'min_child_weight': range(1, 6, 2), 'learning_rate': [0.0001, 0.001, 0.01, 0.1],
        'n_estimators': [100, 200, 300, 400, 500]
    }

    # gbc = GridSearchCV(gbc, parameters, cv=5)
    # fit model to training data
    gbc.fit(X_train, y_train)

    # save best model
    # gbcModel = gbc.best_estimator_

    # Predicting the Test set results
    y_predict = gbc.predict(X_test)
    # y_pred = np.where(y_predict.astype(int) > 0.5, 1, 0)

    evaluate("XGBoost", index, y_predict)

def xgboost_ensemble():

    gbc1 = xgb.XGBClassifier(objective="binary:logistic", seed=27, learning_rate= 0.01, n_estimators=500,
                            max_depth=4, min_child_weight=6, gamma=0, subsample=0.8, colsample_bylevel=0.8,
                            reg_alpha=0.0005, scale_pos_weight=1,thread=4)

    gbc2 = xgb.XGBClassifier(objective="binary:logistic", seed=27, learning_rate= 0.01, n_estimators=5000,
                            max_depth=4, min_child_weight=6, gamma=0, subsample=0.8, colsample_bylevel=0.8,
                            reg_alpha=0.0005, scale_pos_weight=1,thread=4)

    gbc3 = xgb.XGBClassifier(objective="binary:logistic", seed=27, learning_rate= 0.01, n_estimators=5000,
                            max_depth=4, min_child_weight=6, gamma=0, subsample=0.8, colsample_bylevel=0.8,
                            reg_alpha=0.0005, scale_pos_weight=1,thread=4)

    ens = VotingClassifier(estimators=[('xg1', gbc1), ('xg2', gbc2), ('xg3', gbc3)],voting='hard')

    ens.fit(X_train,y_train)

    y_pred = ens.predict(X_test)
    evaluate("ens", index, y_pred)


def GSOM_Classifier():
    gsom = GSOMClassifier()
    gsom.fit(X_train, y_train)
    y_pred = gsom.predict(X_test)
    evaluate("GSOM", index, y_pred)


# data transformation if necessary.
X_t, y_t = processOneHotTrain("../Data/modified_train.csv")
X_test, index = processOneHotTest("../Data/modified_test.csv")

# For SMOTE because of data imbalance
# sm = SMOTE(sampling_strategy='auto', k_neighbors=3, random_state=42)
# sm = OldGeometricSMOTE()
# sm = EGSmote()
# X_train, y_train = sm.fit_resample(X_t, y_t)
X_train, y_train = X_t, y_t

# Trying various classifiers
XGBoost()
# GSOM_Classifiersifier()
