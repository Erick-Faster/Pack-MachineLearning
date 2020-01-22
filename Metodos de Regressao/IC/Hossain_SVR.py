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

#Necessario Slide em y, pra não dar erro!!

X_train = base.iloc[0:68, 1:6].values
y_train = base.iloc[0:68:,6:7].values #slide

X_test = base.iloc[68:84,1:6].values
y_test = base.iloc[68:84,6:7].values #slide

from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
X_train = scaler_x.fit_transform(X_train)
scaler_x = StandardScaler()
X_test = scaler_x.fit_transform(X_test)

#Necessario Escalonar
scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train)
scaler_y = StandardScaler()
y_test = scaler_y.fit_transform(y_test)

'''
Processamento
'''
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, y_train)

'''
Previsao
'''

previsoes_train = regressor.predict(X_train)
previsoes_test = regressor.predict(X_test)

#Calculo do Score - Deve ocorrer antes do desescalonamento
score_train = regressor.score(X_train, y_train)
score_test = regressor.score(X_test, y_test)

#Necessario Desescalonar
y_train = scaler_y.inverse_transform(y_train)
y_test = scaler_y.inverse_transform(y_test)

previsoes_train = scaler_y.inverse_transform(previsoes_train)
previsoes_test = scaler_y.inverse_transform(previsoes_test)

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

