# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 11:33:40 2020

@author: Chaitra B S
"""

#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import the mall dataset with pandas
dataset = pd.read_csv('C:\\Users\\Chaitra B S\\Desktop\\python class\\imp\\Mall_Customers.csv')
x = dataset.iloc[:, [3,4]].values

#Using The Dendrogram to find the optimal number of cluster

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage (x , method = 'ward' ))
plt.title( 'Dendrogram')
plt.xlabel ('Customers')
plt.show()

#Fitting HC to the dataset

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5 , affinity = 'euclidean' , linkage = 'ward')
y_hc = hc.fit_predict(x)

#Visualization

plt.scatter(x[y_hc == 0,0], x[y_hc == 0,1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(x[y_hc == 1,0], x[y_hc == 1,1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(x[y_hc == 2,0], x[y_hc == 2,1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(x[y_hc == 3,0], x[y_hc == 3,1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(x[y_hc == 4,0], x[y_hc == 4,1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('clusters of clients')
plt.xlabel ('Annual Income(K$)')
plt.ylabel ('Spending Scores(1-100)')
plt.legend()
plt.show()