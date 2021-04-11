# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 11:26:51 2019

@author: Chaitra B S
"""

import pandas as pd
data=pd.read_csv('C:\\Users\\Chaitra B S\\Desktop\\python class\\imp\\dataset\\Salary_Data.csv')
i=data.iloc[:,:-1].values

d=data.iloc[:,-1].values


#onehotencoder
'''
from sklearn.preprocessing import OneHotEncoder
onehotencoder=OneHotEncoder(categorical_features=[0])
i=onehotencoder.fit_transform(i).toarray()
i
'''
#splitting the datset into the traning set & Testing set

from sklearn.model_selection import train_test_split
i_train,i_test,d_train,d_test=train_test_split(i,d,test_size=1/3,random_state=0)

#fitting simple linear regressionto the training set

from sklearn.linear_model import LinearRegression
regression= LinearRegression()
regression.fit(i_train,d_train)

#predict the test set result

d_pred=regression.predict(i_test)

#visulising the training set results

import matplotlib.pyplot as plt

plt.scatter(i_train,d_train,color='red')

#plotting the regression line on the traning sets

plt.plot(i_train,regression.predict(i_train),color='blue')
#labeling the x-axis and y-axis

plt.title('salary vs experince(Training set)')
plt.xlabel('Years of experince')
plt.ylabel('salary')
plt.show()


#'''visulaising the test set results
'''plt.scatter(i_test,d_test,color='red')
plt.plot(i_train,regression.predict(i_train),colour='blue')
plt.title('Salary vs experience(test set)')
plt.xlabel('Years of Experince')
plt.ylabel('Salary')
plt.show()'''




'''import numpy as np
a=np.array(15).reshape((1,-1))

new_salary=regressi.predict(a)'''

#fitting polynomial regression to the datasets

from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2)
i_poly=poly_reg.fit_transform(i)

#we need to fit these i_poly into linear regression
from sklearn.linear_model import LinearRegression
regression_2= LinearRegression()
regression_2.fit(i_poly,d)

#visualising the linear regression results

import matplotlib.pyplot as plt

plt.scatter(i_train,d_train,color='red')

#plotting the regression line on the traning sets

plt.plot(i_train,regression.predict(i_train),color='blue')
#labeling the x-axis and y-axis

plt.title('salary vs experince(Training set)')
plt.xlabel('Years of experince')
plt.ylabel('salary')
plt.show()

#visualising the polynomial regression results
import matplotlib.pyplot as plt
plt.scatter(i_train,d_train,color='red')
plt.plot(i_train,regression_2.predict(poly_reg.fit_transform(i_train)),color='blue')
plt.title('salary vs experince(Training set)')
plt.xlabel('Years of experince')
plt.ylabel('salary')
plt.show()

#change the degrees of the polynomial and check 
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=3)
i_poly=poly_reg.fit_transform(i)
poly_reg.fit_transform(i_poly,d)
regression_2=LinearRegression()
regression_2.fit(i_poly,d)

#visualising the polynomial regression results
import matplotlib.pyplot as plt
plt.scatter(i_train,d_train,color='red')
plt.plot(i,regression_2.predict(poly_reg.fit_transform(i)),color='blue')
plt.title('salary vs experince(Training set)')
plt.xlabel('Years of experince')
plt.ylabel('salary')
plt.show()

#change the degrees of the polynomial and check 
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
i_poly=poly_reg.fit_transform(i)
poly_reg.fit_transform(i_poly,d)
regression_2=LinearRegression()
regression_2.fit(i_poly,d)

#visualising the polynomial regression results
import matplotlib.pyplot as plt
plt.scatter(i_train,d_train,color='red')
plt.plot(i,regression_2.predict(poly_reg.fit_transform(i)),color='blue')
plt.title('salary vs experince(Training set)')
plt.xlabel('Years of experince')
plt.ylabel('salary')
plt.show()

#reshaping the curve

#visualising the polynomial regression results

i_grid=np.arange(min(i),max(i),0,1)
i_grid=i_grid.reshape((len(i_grid),1))

import matplotlib.pyplot as plt
plt.scatter(i_train,d_train,color='red')
plt.plot(i_grid,regression_2.predict(poly_reg.fit_transform(i_grid)),color='blue')
plt.title('salary vs experince(Training set)')
plt.xlabel('Years of experince')
plt.ylabel('salary')
plt.show()

#for predicting the salary of epecifi years of expreience
#new_salary=regression.predict(8)

import numpy as np
m=np.array(8).reshape((1,-1))
new_salary=regression.predict(m)

