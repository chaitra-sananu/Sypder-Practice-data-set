# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 09:52:51 2019

@author: Chaitra B S
"""

#multiple linear resgression

#Importing libraries

import numpy as np
import matplotlib as plt
import pandas as pd

#importing data set
dataset=pd.read_csv('C:\\Users\\Chaitra B S\\Desktop\\python class\\imp\\dataset\\50_Startups.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,4].values


#encoding categorical data
#encoding the independent variable

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labelencoder_x=LabelEncoder()
x[:,3]=labelencoder_x.fit_transform(x[:,3])
onehotencoder=OneHotEncoder(categorical_features=[3])
x=onehotencoder.fit_transform(x).toarray()

#splitting the dataset into the training set and testing sets

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#featur Scaling

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)

#fitting Multiple Linear Regression to the linear regression

from sklearn.linear_model import LinearRegression
regression=LinearRegression()
regression.fit(x_train,y_train)

#predict the test set restults

y_pred=regression.predict(x_test)

#multiple linear regression

import statsmodels.formula.api as sm
x=np.append(arr=np.ones((50,1)).astype(int),values=x, axis=1)
x_opt=x[:,[0,1,2,3,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()

#x4 is more than 5 hence remove it from the x_opt

import statsmodels.formula.api as sm
x=np.append(arr=np.ones((50,1)).astype(int),values=x, axis=1)
x_opt=x[:,[0,1,2,3,5]]
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()

#now x4 is 5, it is more than 5 hence remove it from the line

import statsmodels.formula.api as sm
x=np.append(arr=np.ones((50,1)).astype(int),values=x, axis=1)
x_opt=x[:,[0,1,2,3]]
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()




