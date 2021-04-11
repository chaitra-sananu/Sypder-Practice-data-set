# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 12:30:03 2019

@author: Chaitra B S
"""

#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing the dataset
data=pd.read_csv('C:\\Users\\Chaitra B S\\Desktop\\python class\\imp\\dataset\\Position_salaries.csv')
x=data.iloc[:,1:2].values
y=data.iloc[:,2].values

#Fitting Random Forest Regression to the dataset

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(x,y)


#predicting the new result
y_pred = regressor.predict(np.array([[6.5]]))


#visulising the Decision Tree Regressor results
plt.scatter(x,y,color='red')
plt.plot(x,regressor.predict(x),color='blue')
plt.title('Truth or Bluff(Decision Tree Regressor)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#reshaping the curve  (higher and smoother curve)

x_grid = np.arange(min(x),max(x),0.01)
x_grid = x_grid.reshape((len(x_grid),1))

#visulising the decision tree regressor result (higher and smoother curve)
#import matplotlib.pyplot as plt
plt.scatter(x,y,color='red')
plt.plot(x_grid,regressor.predict(x_grid),color ='blue')
plt.title('Truth or Bluff(Decision Tree Regressor)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


