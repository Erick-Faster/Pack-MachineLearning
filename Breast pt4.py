# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:46:37 2019

@author: Faster-PC
"""

#####################################
##(6)Tunning - O programa procura a melhor combinacao para obter os dados
#mais precisos. Processo pode demorar horas ou dias
##(7)Salvar Rede Neural - Posso salvar tanto a rede quanto os pesos calculados
#####################################


import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

previsores = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv')

from sklearn.model_selection import train_test_split
previsores_train, previsores_test, classe_train, classe_test = train_test_split(previsores, classe, test_size=0.25)

'''Estruturando a Rede Neural'''

#Parametros implementados na funcao

classificador = Sequential()
    
#Melhores parametros
classificador.add(Dense(units = 8, activation = 'relu', kernel_initializer = 'normal', input_dim = 30))
classificador.add(Dropout(0.2)) #Evita Overfitting zerando 20% dos dados de entrada
    
classificador.add(Dense(units = 8, activation = 'relu', kernel_initializer = 'normal'))
classificador.add(Dropout(0.2))
    
classificador.add(Dense(units=1, activation='sigmoid')) 
        
classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['binary_accuracy'])

classificador.fit(previsores, classe, batch_size = 10, epochs = 100)


#Adicionar um dado novo, realmente aplicar
novo = np.array([[15.80, 8.34, 111, 900, 0.10, 0.26, 0.08, 0.134, 0.178, 0.20, 0.05, 1098, 0.87, 4500, 145.2, 0.005, 0.04, 0.05, 0.015, 0.03, 0.007, 23.15, 16.64, 178.5, 2018, 0.14, 0.185, 0.84, 158, 0.363]])
previsao = classificador.predict(novo)
previsao = (previsao >0.5)

#Salvar Rede
classificador_json = classificador.to_json()
with open('classificador_breast.json', 'w') as json_file:
    json_file.write(classificador_json)

#Salvar pesos
classificador.save_weights('classificador_breast.h5')