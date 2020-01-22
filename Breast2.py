# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 21:11:15 2019

@author: Faster-PC
"""

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

#Importar e montar previsores

previsores = pd.read_csv("entradas-breast.csv")
classe = pd.read_csv("saidas-breast.csv")

previsores_train, previsores_test, classe_train, classe_test = train_test_split(previsores, classe, test_size=0.25)

####Criar Rede###

classificador = Sequential()
classificador.add(Dense(units = 8, activation = 'relu', kernel_initializer = 'normal', input_dim = 30))
classificador.add(Dropout(0.2))

classificador.add(Dense(units = 16, activation = 'relu', kernel_initializer = 'normal'))
classificador.add(Dropout(0.2))

classificador.add(Dense(units = 1, activation = 'sigmoid'))

#Compilar

classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['binary_accuracy'])

classificador.fit(previsores_train, classe_train, batch_size = 10, epochs = 100)

previsoes = classificador.predict(previsores_test)
previsoes = (previsoes > 0.5)

precisao = accuracy_score(classe_test, previsoes)
matriz = confusion_matrix(classe_test, previsoes)

