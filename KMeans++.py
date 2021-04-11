# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 11:48:05 2020

@author: Chaitra B S
"""

#importing the libraries
import pandas as pd
import matplotlib.pyplot as plt

#import the mall dataset with pandas
dataset = pd.read_csv('C:\\Users\\Chaitra B S\\Desktop\\python class\\imp\\Mall_Customers.csv')
x = dataset.iloc[:, [3,4]].values

#using ELBOW Method to find the optimal number

from sklearn.cluster import KMeans
wcss = []
for i in range (1 , 11):
        kmeans = KMeans (n_clusters = i, init = 'k-means++' , max_iter = 300, n_init = 10, random_state=0)
        kmeans.fit(x)
        wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of cluster')
plt.ylabel('wcss')
plt.show()

#Applying k-means to the dataset
KMeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_KMeans = KMeans.fit_predict(x)

#Visualization

plt.scatter(x[y_KMeans == 0,0], x[y_KMeans == 0,1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(x[y_KMeans == 1,0], x[y_KMeans == 1,1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(x[y_KMeans == 2,0], x[y_KMeans == 2,1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(x[y_KMeans == 3,0], x[y_KMeans == 3,1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(x[y_KMeans == 4,0], x[y_KMeans == 4,1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow',
            label = 'Centroids')
plt.title('clusters of clients')
plt.xlabel ('Annual Income(K$)')
plt.ylabel ('Spending Scores(1-100)')
plt.legend()
plt.show()



