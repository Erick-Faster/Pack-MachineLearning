# -*- coding: utf-8 -*-
"""
Regressao Linear Simples - Pega somente uma array de dados para a previsao
Gera um grafico do tipo y = b0 + b1x

Resultado = Score: 0.86
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

#Plotar o grafico
import matplotlib.pyplot as plt
plt.scatter(X, y)
plt.plot(X, regressor.predict(X), color = 'red')
plt.title("Regressao Linear Simples")
plt.xlabel('Idade')
plt.ylabel('Custo')

#Realizando o teste de previsao. Ambos devem sair iguais.
previsao1 = regressor.predict([[40]])
previsao2 = regressor.intercept_ + regressor.coef_ * 40

#Resultados
score = regressor.score(X, y)

#YellowBrick -> para visualizar dados em ML.
#Gera grafico para encontrar os dados residuais e o R quadrado
from yellowbrick.regressor import ResidualsPlot
visualizador = ResidualsPlot(regressor)
visualizador.fit(X,y)
visualizador.poof()
