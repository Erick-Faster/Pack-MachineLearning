# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:46:37 2019

@author: Faster-PC
"""

#####################################
##(3)K-fold Cross Validation - Divide dataset em k partes. Geralmente k = 10
#No fim, todos os dados sao usados como teste ou como treino
#Metodo MAIS usado na comunidade cientifica
##(4)Dropout - Evita Overfitting zerando parte dos dados de entrada
#Sempre eh bom fazer dropout após criar um Dense
##(5)Overfitting - Qto maior o desvio-padrao, maior o overfitting
#####################################


import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier

previsores = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv')

from sklearn.model_selection import train_test_split
previsores_train, previsores_test, classe_train, classe_test = train_test_split(previsores, classe, test_size=0.25)

'''Estruturando a Rede Neural'''

def criarRede():
    classificador = Sequential()
    
    classificador.add(Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform', input_dim = 30))
    classificador.add(Dropout(0.2)) #Evita Overfitting zerando 20% dos dados de entrada
    
    classificador.add(Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform'))
    classificador.add(Dropout(0.2))
    
    classificador.add(Dense(units=1, activation='sigmoid')) 
    
    otimizador = keras.optimizers.Adam(lr = 0.001, decay = 0.0001, clipvalue = 0.5)
    
    classificador.compile(optimizer = otimizador, loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
    return classificador

classificador = KerasClassifier(build_fn = criarRede,
                                epochs = 100,
                                batch_size = 10)

resultados = cross_val_score(estimator = classificador,
                             X = previsores, y = classe,
                             cv = 10, scoring = 'accuracy')

media = resultados.mean()
desvio = resultados.std()