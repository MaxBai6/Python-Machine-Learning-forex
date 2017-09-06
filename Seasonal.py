# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 20:37:35 2017

@author: 白家鹏
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from statsmodels.tsa.arima_model import ARIMA
from datetime import datetime

Vehicle1 = pd.read_csv("AUDUSD.csv", parse_dates=[0],index_col=0)
#date = elec['Date']
#elec = elec.iloc[::-1]
data= Vehicle1['Price'][-36:]
plt.figure()
plt.plot(Vehicle1['Price'])
plt.title("PriceOfForex")
Vehicle = Vehicle1.iloc[::-1]

decomp_obj = sm.tsa.seasonal_decompose(Vehicle1['Price'])
decomp_obj.plot()

decomp_obj = sm.tsa.seasonal_decompose(Vehicle1['Price'][:48])
decomp_obj.plot()

decomp_obj = sm.tsa.seasonal_decompose(Vehicle1['Price'][-60 3
                                       2:])
decomp_obj.plot()
