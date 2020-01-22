# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 21:42:39 2019

@author: Faster-PC
"""

#Part 1 - Importacoes

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

#Part 2 - Banco de Dados
previsores = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv')

previsores_train, previsores_test, classe_train, classe_test = train_test_split(previsores, classe, test_size=0.20)

#Part 3 - Moldando rede neural

RedeNeural = Sequential()
RedeNeural.add(Dense(units=8, activation='relu', kernel_initializer='random_uniform', input_dim = 30))
RedeNeural.add(Dropout(0.2))

RedeNeural.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform'))
RedeNeural.add(Dropout(0.2))

RedeNeural.add(Dense(units=1, activation='sigmoid'))

#Part 4 - Compilando

RedeNeural.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
RedeNeural.fit(previsores_train, classe_train, batch_size=10, epochs=100)

#Part 5 - Teste

previsor = RedeNeural.predict(previsores_test)
previsor = (previsor > 0.5)

precisao = accuracy_score(classe_test, previsor)
matriz = confusion_matrix(classe_test, previsor)
