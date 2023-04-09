# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 15:03:08 2023

@author: eid
"""

import pandas as pd
import matplotlib as mat
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
train_data=pd.read_csv("housing.csv")
train_data.dropna(inplace=True)
for x in train_data:
       if train_data[x].dtype =='float64' and abs(train_data[x].skew())>1 :
           train_data[x]=np.log(train_data[x]+1)
train_data=train_data.join(pd.get_dummies(train_data.ocean_proximity))
train_data=train_data.drop('ocean_proximity',axis=1)
x=train_data.drop('median_house_value',axis=1)
y=train_data['median_house_value']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
model=RandomForestRegressor()
model.fit(x_train,y_train)
a=model.score(x_test, y_test)
print(a)
