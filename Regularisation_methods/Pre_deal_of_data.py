# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 19:57:33 2017

@author: 白家鹏
"""

import pandas as pd
import numpy as np

csv1 = pd.read_csv('AUDUSD2.csv')
csv2 = csv1[:-31]
Date=csv1['Date']
Price=csv1['Price']
Date_value=Date.values
X=csv1['Date'].values
Y=csv1['Price'].values
#x = np.reshape(X, (len(X), 1))
#x = np.column_stack(x)
y = np.reshape(Y, (len(Y), 1))
y= np.asmatrix(Y)


for i in range(1, 31):
    csv2['Before_{0}Days'.format(i)] = Y[i:-31+i]
    
df = pd.DataFrame(csv2)
filename = 'AUDUSD_30days.csv'
df.to_csv(filename, index=False, encoding='utf-8')

