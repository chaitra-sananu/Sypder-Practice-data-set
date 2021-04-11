# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 18:39:58 2019

@author: Chaitra B S
"""

import pandas as pd
data=pd.read_csv('C:\\Users\\Chaitra B S\\Desktop\\python class\\imp\\dataset\\Position_salaries.csv')

x=data.iloc[:,1:2].values

y=data.iloc[:,2].values

#Feature Scaling
import numpy as np
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = np.squeeze(sc_y.fit_transform(y.reshape(-1,1)))

#fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x,y)

#y_pred = regressor.predict(sc_x.transform(6.5))

y_pred = regressor.predict(sc_x.transform(np.array([[6.5]])))

#visulaising the SVR results
import matplotlib.pyplot as plt
plt.scatter(x,y,color='red')
plt.plot(x,regressor.predict(x),color='blue')
plt.title('Truth or Bluff(SVR Model)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

y_pred = sc_y.inverse_transform(regressor.predict(sc_x.transform(np.array([[6.5]]))))
