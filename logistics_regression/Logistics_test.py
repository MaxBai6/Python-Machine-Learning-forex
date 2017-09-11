# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 21:25:39 2017

@author: 白家鹏
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split

"""
First, input the date which you hope to predict
"""
ID_of_Date=99

"""
Then, begin our project
"""
df = pd.read_csv('AUDUSD_logistics2.csv')
csv2 = df[:-31]

X1 = df[['Rate_high']][30:]
X2 = df[['Rate_low']][30:]
Y=df.iloc[:,2:7].values
position=np.zeros((len(csv2),1))
"""
Deal y
"""

for i in range(1, 31):
    csv2['Before_{0}Days_Open'.format(i)] = Y[i:-31+i,2]
    csv2['Before_{0}Days_Highest'.format(i)] = Y[i:-31+i,2]
    csv2['Before_{0}Days_Lowest'.format(i)] = Y[i:-31+i,2]
    csv2['Before_{0}Days_Adjust'.format(i)] = Y[i:-31+i,3]
    csv2['Before_{0}Days_Volume'.format(i)] = Y[i:-31+i,4]
"""

"""
x =csv2.iloc[:, 9:]
y = csv2.iloc[:, 8]
y=np.asarray(y)

for i in range(0, len(csv2)-1):
    if float(y[i]) > 0.05:
        position[i]=5
    if float(y[i])<0.05:
        position[i]=4
    if float(y[i])<0.04:
        position[i]=3
    if float(y[i])<0.03:
        position[i]=2
    if float(y[i])<0.02:
        position[i]=1
    if float(y[i])<0.01:
        position[i]=0
    if float(y[i])<-0.01:
        position[i]=-1
    if float(y[i])<-0.02:
        position[i]=-2
    if float(y[i])<-0.03:
        position[i]=-3
    if float(y[i])<-0.04:
        position[i]=-4
    if float(y[i])<-0.05:
        position[i]=-5
    
csv2['Position'] = position


X_train, X_val, y_train, y_val = train_test_split(x, csv2['Position'], test_size=0.3)

log_res = LogisticRegression()

log_res.fit(X_train, y_train)



from sklearn.metrics import confusion_matrix

pred_log = log_res.predict(X_val)

print(confusion_matrix(pred_log, y_val))

from sklearn.metrics import classification_report  

print(classification_report(y_val, pred_log, digits=3))

print("The first table finished")

"""

"""
x =csv2.iloc[:, 9:]
y = csv2.iloc[:, 7]
y=np.asarray(y)
for i in range(0, len(csv2)-1):
    if float(y[i]) > 0.05:
        position[i]=5
    if float(y[i])<0.05:
        position[i]=4
    if float(y[i])<0.04:
        position[i]=3
    if float(y[i])<0.03:
        position[i]=2
    if float(y[i])<0.02:
        position[i]=1
    if float(y[i])<0.01:
        position[i]=0
    if float(y[i])<-0.01:
        position[i]=-1
    if float(y[i])<-0.02:
        position[i]=-2
    if float(y[i])<-0.03:
        position[i]=-3
    if float(y[i])<-0.04:
        position[i]=-4
    if float(y[i])<-0.05:
        position[i]=-5
    
csv2['Position'] = position

X_train2, X_val2, y_train2, y_val2 = train_test_split(x, csv2['Position'], test_size=0.3)

log_res2 = LogisticRegression()

log_res2.fit(X_train2, y_train2)



from sklearn.metrics import confusion_matrix

pred_log2 = log_res2.predict(X_val2)

print(confusion_matrix(pred_log2, y_val2))

from sklearn.metrics import classification_report  

print(classification_report(y_val2, pred_log2, digits=3))

print("The first table finished")
#
prob = log_res2.predict_proba(x.iloc[99, :])
"""
calculate the best plan
"""
ID_of_Date=99
prob_nagative = log_res.predict_proba(x.iloc[ID_of_Date, :-1])
prob_positive = log_res2.predict_proba(x.iloc[ID_of_Date, :])

profit= np.zeros((25,1))

prob_positive1=prob_positive[0]
prob_nagative1=prob_nagative[0]

Max=sum(list(prob_nagative1[2:]))

num=0
for i in range(1, 6):
    for j in range(1, 6):
        profit[num]=sum(list(prob_positive1[i:]))*i+sum(list(prob_nagative1[j:]))*(-j)
        num=num+1
        
max_positive_profit=max(profit)
max_nagative_profit=min(profit)

profit_all = np.reshape(profit,(5,5))
profit_level=0
loss_stop=0

   
for profit_level in range(1,6):
    for loss_stop in range(1,6):
        if profit_all[profit_level-1,loss_stop-1] == max_positive_profit:   
            print("----------------------------------------------------------------------------------")
            print("the best positive strategy is that positive level at{0}%, and loss stop at {1}%     --".format(profit_level,loss_stop))
            print("the expected profit of the plan is {0}".format(max_positive_profit))
            print("----------------------------------------------------------------------------------")
        if profit_all[profit_level-1,loss_stop-1] == max_nagative_profit:
            print("----------------------------------------------------------------------------------")
            print("the best nagative strategy is that positive level at{0}%, and loss stop at {1}%     --".format(profit_level,loss_stop))
            print("the expected profit of the plan is {0}".format(max_nagative_profit))
            print("----------------------------------------------------------------------------------")

