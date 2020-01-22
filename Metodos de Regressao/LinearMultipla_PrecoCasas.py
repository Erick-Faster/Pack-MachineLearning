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
base = pd.read_csv('house-prices.csv')
X = base.iloc[:,3:19].values
y = base.iloc[:,2].values

from sklearn.model_selection import train_test_split
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y,
                                                                  test_size = 0.3,
                                                                  random_state = 0)

'''
Processamento
'''
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_treinamento, y_treinamento)

'''
Pos_Processamento
'''

score = regressor.score(X_treinamento, y_treinamento)
previsoes = regressor.predict(X_teste)

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_teste, previsoes)

regressor.score(X_teste, y_teste)

regressor.intercept_
regressor.coef_