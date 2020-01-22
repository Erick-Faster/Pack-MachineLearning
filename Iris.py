# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 22:22:45 2019

@author: Faster-PC
"""

'''[1]Importando pacotes'''
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

'''[2]Importando base de dados'''
base = pd.read_csv('iris.csv')
previsores = base.iloc[:,0:4].values
classe = base.iloc[:,4].values

'''[3]Convertendo classe (preprocessamento)'''
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)

#Metodo requer 3 dimensoes, pois sao 3 neuronios na saida
classe_dummy = np_utils.to_categorical(classe) #Transforma em 3 dimensoes

##Classe Dummy""
# iris setosa = 100
# iris versicolor = 010
# iris virginica = 001

previsores_train, previsores_test, classe_train, classe_test = train_test_split(previsores, classe_dummy, test_size=0.25)

'''[4]Criando Rede Neural'''
classificador = Sequential()
classificador.add(Dense(units=4, activation='relu', input_dim = 4))
classificador.add(Dense(units=4, activation='relu'))
classificador.add(Dense(units=3, activation='softmax')) #Pra classificacao de varias classes. Tenho 3 saidas

classificador.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['categorical_accuracy']) #Troca o 'binary' para 'categorical'

classificador.fit(previsores_train, classe_train, batch_size = 10, epochs = 1000)

'''[5]Testando e Convertendo Testes'''
previsoes = classificador.predict(previsores_test)
previsoes = (previsoes > 0.5)

#Conversao necessaria, pois matriz nao lida com 3 dimensoes
#'''''''
import numpy as np
classe_test2 = [np.argmax(t) for t in classe_test] #Converte 100 -> 0, 010 -> 1, 001 ->2
previsoes2 = [np.argmax(t) for t in previsoes] #Converte 100 -> 0, 010 -> 1, 001 ->2
#'''''''''

from sklearn.metrics import confusion_matrix, accuracy_score
matriz = confusion_matrix(classe_test2, previsoes2)
precisao = accuracy_score(classe_test2, previsoes2)