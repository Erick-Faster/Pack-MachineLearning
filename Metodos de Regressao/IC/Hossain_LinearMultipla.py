# -*- coding: utf-8 -*-
"""
[Parte 3] - Metodos de Regressao

Metodo de Regressao linear Multipla para prever preco das casas
Utiliza varios atributos para prever os resultados

Resultado: Score: 0,70. Um resultado melhor do que a Simples
"""

'''
Pre-Processamento
'''
import pandas as pd

base = pd.read_excel('Hossain.xlsx', encoding = 'ISO-8859-1')

X_train = base.iloc[0:68, 1:6].values
y_train = base.iloc[0:68:,6].values

X_test = base.iloc[68:84,1:6].values
y_test = base.iloc[68:84,6].values

'''
Processamento
'''
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

'''
Previsao
'''

previsoes_train = regressor.predict(X_train)
previsoes_test = regressor.predict(X_test)


'''
Tratamento dos Dados
'''
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

#Dados estatisticos do treinamento
mae_train = mean_absolute_error(y_train, previsoes_train)
mse_train = mean_squared_error(y_train, previsoes_train)
rmse_train = sqrt(mean_squared_error(y_train, previsoes_train))

#Dados estatisticos dos testes
mae_test = mean_absolute_error(y_test, previsoes_test)
mse_test = mean_squared_error(y_test, previsoes_test)
rmse_test = sqrt(mean_squared_error(y_test, previsoes_test))

#Calculo do Score
score_train = regressor.score(X_train, y_train)
score_test = regressor.score(X_test, y_test)

#Determinacao da Formula
b = regressor.intercept_
a = regressor.coef_