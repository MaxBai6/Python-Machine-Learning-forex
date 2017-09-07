# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 20:39:35 2017

@author: 白家鹏
"""

from sklearn.linear_model import LinearRegression, RidgeCV
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import neighbors
from sklearn import datasets
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

ForexData = pd.read_csv('AUDUSD_30days.csv')

# Get the features X and target/response y
X =ForexData.iloc[:, 2:]
y = ForexData.iloc[:, 1]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

"""
OLS

""" 
lm = LinearRegression()

lm.fit(X_train, y_train)

preds_ols = lm.predict(X_test)

print("OLS: {0}".format(mean_squared_error(y_test, preds_ols)/2))

"""
RIDGE

""" 
alpha_range = 10.**np.arange(-2, 3)

rregcv = RidgeCV(normalize=True, scoring='neg_mean_squared_error', alphas=alpha_range)

rregcv.fit(X_train, y_train)

preds_ridge = rregcv.predict(X_test)
print("RIDGE: {0}".format(mean_squared_error(y_test, preds_ridge)/2))

"""
LassoCV

"""
from sklearn.linear_model import LassoCV, ElasticNetCV

lascv = LassoCV(normalize=True, tol=0.01)
lascv.fit(X_train, y_train)

preds_lassocv = lascv.predict(X_test)
print("LASSO: {0}".format(mean_squared_error(y_test, preds_lassocv)/2))
print("LASSO Lambda: {0}".format(lascv.alpha_))
"""
ElasticNetCV

"""
elascv = ElasticNetCV(normalize=True, tol=0.01)
elascv.fit(X_train, y_train)

preds_elascv = elascv.predict(X_test)
print("Elastic Net: {0}".format(mean_squared_error(y_test, preds_elascv)/2))
print("Elastic Net Lambda: {0}".format(elascv.alpha_))


"""
b

"""
m=245
#obtain features of LeBron_James and Kevin_Durant
row = X.iloc[m, :]
Test_predictions_LM = lm.predict(row)
Test_predictions_RISGE = rregcv.predict(row)
Test_predictions_LASSO = lascv.predict(row)
Test_predictions_ELA = elascv.predict(row)
Test_realValue=y.iloc[m]