# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 20:19:18 2019

@author: Faster-PC
"""

import pandas as pd

base = pd.read_csv('plano-saude.csv')

X = base.iloc[:,0].values
y = base.iloc[:,1].values

import numpy as np
correlacao = np.corrcoef(X,y) #Qto + proximo de 1, melhor, + correlacionado

X = X.reshape(-1,1) #Transforma lista em matriz. -1 pq n quero mexer nas linhas, 1 pq quero add uma coluna

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X,y)

#y = b0 + b1*x
regressor.intercept_ #b0
regressor.coef_ #b1

import matplotlib.pyplot as plt
plt.scatter(X, y)
plt.plot(X, regressor.predict(X), color = 'red')
plt.title("Regressao Linear Simples")
plt.xlabel('Idade')
plt.ylabel('Custo')

previsao1 = regressor.predict([40])