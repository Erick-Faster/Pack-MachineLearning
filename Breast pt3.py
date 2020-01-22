# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:46:37 2019

@author: Faster-PC
"""

#####################################
##(6)Tunning - O programa procura a melhor combinacao para obter os dados
#mais precisos. Processo pode demorar horas ou dias
#####################################


import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
previsores = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv')

from sklearn.model_selection import train_test_split
previsores_train, previsores_test, classe_train, classe_test = train_test_split(previsores, classe, test_size=0.25)

'''Estruturando a Rede Neural'''

#Parametros implementados na funcao
def criarRede(optimizer, loss, kernel_initializer, activation, neurons):
    classificador = Sequential()
    
    #Parametros substituidos
    classificador.add(Dense(units = neurons, activation = activation, kernel_initializer = kernel_initializer, input_dim = 30))
    classificador.add(Dropout(0.2)) #Evita Overfitting zerando 20% dos dados de entrada
    
    classificador.add(Dense(units = neurons, activation = activation, kernel_initializer = kernel_initializer))
    classificador.add(Dropout(0.2))
    
    classificador.add(Dense(units=1, activation='sigmoid')) 
        
    classificador.compile(optimizer = optimizer, loss = loss, metrics = ['binary_accuracy'])
    return classificador

classificador = KerasClassifier(build_fn = criarRede)

#Parametros que serao utilizados, em forma de DICIONARIO
parametros = {'batch_size': [10,30],
              'epochs': [50, 100],
              'optimizer': ['adam', 'sgs'],
              'loss': ['binary_crossentropy', 'hinge'],
              'kernel_initializer': ['random_uniform', 'normal'],
              'activation': ['relu','tanh'],
              'neurons': [16, 8]}

#Funcao que configura a busca pelas melhores configuracoes
grid_search = GridSearchCV(estimator = classificador,
                           param_grid = parametros,
                           scoring = 'accuracy',
                           cv = 5)

#Funcao que roda a busca
grid_search = grid_search.fit(previsores, classe)

#Geracao dos dados dos parametros buscados e melhor precisao
melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_