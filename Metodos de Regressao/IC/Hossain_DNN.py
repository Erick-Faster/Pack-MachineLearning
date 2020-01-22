# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 00:16:29 2019

@author: Faster-PC
"""

'''
Pre-Processamento
'''

import pandas as pd

base = pd.read_excel('Hossain.xlsx', encoding = 'ISO-8859-1')

X_train = base.iloc[0:68, 1:6].values
y_train = base.iloc[0:68:,6:7].values

X_test = base.iloc[68:84,1:6].values
y_test = base.iloc[68:84,6:7].values


from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
X_train = scaler_x.fit_transform(X_train)
scaler_x = StandardScaler()
X_test = scaler_x.fit_transform(X_test)

scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train)
scaler_y = StandardScaler()
y_test = scaler_y.fit_transform(y_test)

'''
Processamento
'''

from keras.models import Sequential
from keras.layers import Dense

regressor = Sequential()
regressor.add(Dense(units = 5, activation = 'relu', input_dim = 5))
regressor.add(Dense(units = 3, activation = 'relu'))
regressor.add(Dense(units = 1, activation = 'linear'))

regressor.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['mean_squared_error'])
regressor.fit(X_train, y_train, batch_size = 2, epochs = 500)

'''
Previsao
'''

previsoes_train = regressor.predict(X_train)
previsoes_test = regressor.predict(X_test)

'''
Tratamento dos Dados
'''

#Necessario Desescalonar
y_train = scaler_y.inverse_transform(y_train)
y_test = scaler_y.inverse_transform(y_test)

previsoes_train = scaler_y.inverse_transform(previsoes_train)
previsoes_test = scaler_y.inverse_transform(previsoes_test)

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