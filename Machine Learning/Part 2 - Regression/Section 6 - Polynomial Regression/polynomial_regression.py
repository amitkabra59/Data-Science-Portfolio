# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 20:22:48 2017

@author: Amit
"""

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) """

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#Fitting Linear Regression to dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

#Fitting Polynomial regression to dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg2= LinearRegression()
lin_reg2.fit(X_poly,y)

#Visualiizing linear regression results
plt.scatter(X,y, color='red')
plt.plot(X,lin_reg.predict(X),color='blue')
plt.tittle('Truth or Bluff (LR)')
plt.xlabel('Position level')
plt.ylabel('salary')
plt.show()

#Visualizing Polynomial regression results
X_grid= np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y, color='green')
plt.plot(X_grid,lin_reg2.predict(poly_reg.fit_transform(X_grid)),color='yellow')
plt.tittle('Truth or Bluff (PR)')
plt.xlabel('Position level')
plt.ylabel('salary')
plt.show()

#Predicting new results with LR
lin_reg.predict(6.5)

#predicting with PR

lin_reg2.predict(poly_reg.fit_transform(6.5))