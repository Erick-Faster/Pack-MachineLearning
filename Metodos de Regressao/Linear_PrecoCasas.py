# -*- coding: utf-8 -*-
"""
Metodo de Regressao linear simples para prever preco das casas
Utiliza somente uma array para prever os resultados

Resultado: Score: 0,49. Um resultado bem ruim
"""

import pandas as pd

'''
Pre-Processamento
'''
base = pd.read_csv('house-prices.csv')
X = base.iloc[:,5].values #sqft_living
y = base.iloc[:,2].values #Price

from sklearn.model_selection import train_test_split
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y,
                                                                  test_size = 0.3,
                                                                  random_state = 0)

X_treinamento = X_treinamento.reshape(-1,1)
X_teste = X_teste.reshape(-1,1)

'''
Processamento
'''

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_treinamento, y_treinamento)
score = regressor.score(X_treinamento, y_treinamento)

'''
Pos-Processamento
'''

#Plota grafico dos dados de treinamento
import matplotlib.pyplot as plt
plt.scatter(X_treinamento, y_treinamento) #Gera Grafico todo
plt.plot(X_treinamento, regressor.predict(X_treinamento), color = 'red')

#Utiliza os dados de teste para gerar as previsoes
previsoes = regressor.predict(X_teste)

#Indica a diferenca entre o valor predito e o real
resultado = abs(y_teste - previsoes) #abs elimina negativos

#Plota R2 e Erro
from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(y_teste, previsoes)
mse = mean_squared_error(y_teste, previsoes)

#Plota grafico com os dados de teste
plt.scatter(X_teste, y_teste) #Gera Grafico todo
plt.plot(X_teste, regressor.predict(X_teste), color = 'red')
