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

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X_treinamento, y_treinamento)
score = regressor.score(X_treinamento, y_treinamento)

previsoes = regressor.predict(X_teste)

from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(y_teste, previsoes)
mse = mean_squared_error(y_teste, previsoes)

