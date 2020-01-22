# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:46:37 2019

@author: Faster-PC
"""

import pandas as pd
previsores = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv')

from sklearn.model_selection import train_test_split
previsores_train, previsores_test, classe_train, classe_test = train_test_split(previsores, classe, test_size=0.25)

'''Estruturando a Rede Neural'''

import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, accuracy_score

#Cria rede neural sequencial
classificador = Sequential() #Camadas sao adicionadas sequencialmente

#Cria ligacoes densas na camada de ENTRADA
classificador.add(Dense(units = 16, #No de entradas (30) + no saidas (1) / 2
                        activation = 'relu',
                        kernel_initializer = 'random_uniform', #Inicializador dos pesos -> Random
                        input_dim = 30)) #No de entradas

classificador.add(Dense(units = 16, #No de entradas (30) + no saidas (1) / 2
                        activation = 'relu',
                        kernel_initializer = 'random_uniform')) #Inicializador dos pesos -> Random
                        


classificador.add(Dense(units = 16, #No de entradas (30) + no saidas (1) / 2
                        activation = 'relu',
                        kernel_initializer = 'random_uniform', #Inicializador dos pesos -> Random
                        input_dim = 30)) #No de entradas


classificador.add(Dense(units = 1, #Uma soh saida (true or false)
                        activation = 'sigmoid'))

'''Compilando e Configurando'''

otimizador = keras.optimizers.Adam(lr = 0.001, #Learning Rate
                                   decay = 0.0001,#Quanto que ele decrementa ao descer o gradiente
                                   clipvalue = 0.5) #Trava pesos, para n passar de 0.5

classificador.compile(optimizer = otimizador, #Otimizador + utilizado. Se n funcionar, testar outros
                      loss = 'binary_crossentropy', #Calculo do erro para 1 saida True/False
                      metrics = ['binary_accuracy']) #Errado/Certo. Pode ter mais de uma metrica.

classificador.fit(previsores_train, classe_train,
                  batch_size = 10, #Faz ajuste de pesos de 10 em 10
                  epochs = 100) #Qtas vezes o ajuste eh feito

'''Testando'''

previsoes = classificador.predict(previsores_test)
previsoes = (previsoes > 0.5)

precisao = accuracy_score(classe_test, previsoes)
matriz = confusion_matrix(classe_test, previsoes)

resultado = classificador.evaluate(previsores_test, classe_test)