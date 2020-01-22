# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 22:15:33 2019

@author: Faster-PC
"""

import pandas as pd

base = pd.read_csv('plano-saude2.csv')

X = base.iloc[:, 0:1].values
y = base.iloc[:, 1].values

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 2) #Expoentes
X_poly = poly.fit_transform(X)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_poly, y)
score = regressor.score(X_poly, y)

#Previsao requer valor normal e polinomiais
regressor.predict(poly.transform([[40]]))

import matplotlib.pyplot as plt
plt.scatter(X,y)
plt.plot(X, regressor.predict(poly.fit_transform(X)), color = 'red')
plt.title('Regressao polinomial')
plt.xlabel('Idade')
plt.ylabel('Custo')