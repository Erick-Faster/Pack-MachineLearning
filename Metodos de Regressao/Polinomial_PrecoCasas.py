# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 22:41:30 2019

@author: Faster-PC
"""

import pandas as pd

'''
Pre-Processamento
'''
base = pd.read_csv('house-prices.csv')

X = base.iloc[:,3:19].values #sqft_living
y = base.iloc[:,2].values #Price

from sklearn.model_selection import train_test_split
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y,
                                                                  test_size = 0.3,
                                                                  random_state = 0)
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 2)
X_treinamento_poly = poly.fit_transform(X_treinamento)
X_teste_poly = poly.transform(X_teste)

'''
Processamento
'''

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_treinamento_poly, y_treinamento)
score = regressor.score(X_treinamento_poly, y_treinamento)

previsoes = regressor.predict(X_teste_poly)

