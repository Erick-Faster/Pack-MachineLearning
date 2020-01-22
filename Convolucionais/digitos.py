# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 20:48:38 2019

@author: Faster-PC
"""

import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils


(x_treinamento, y_treinamento), (x_teste, y_teste) = mnist.load_data()
plt.imshow(x_treinamento[0], cmap = 'gray')
plt.title('Classe ' + str(y_treinamento[0]))

'''
PRE-PROCESSAMENTO
'''
#Reduzir canais
previsores_treinamento = x_treinamento.reshape(x_treinamento.shape[0], 28, 28, 1) #28x28, 1 canal ao inves de 3
previsores_teste = x_teste.reshape(x_teste.shape[0], 28, 28, 1)

#Converter para float
previsores_treinamento = previsores_treinamento.astype('float32')
previsores_teste = previsores_teste.astype('float32')

#Reduzir para escala de 0 a 1
previsores_treinamento /= 255 #255 eh a escala RGB
previsores_teste /= 255

classe_treinamento = np_utils.to_categorical(y_treinamento, 10) #OneHotEncoder. 10 atributos
classe_teste = np_utils.to_categorical(y_teste, 10) #OneHotEncoder

'''
REDE CONVOLUCIONAL
'''

classificador = Sequential()

#Etapa 1 - Operador de Convolucao
classificador.add(Conv2D(32, #32 combinacoes possiveis de kernels 
                         (3,3), #kernel 3x3
                         input_shape = (28,28,1), #formato da entrada (28x28, 1 canal)
                         activation = 'relu'))

#Etapa 2 - Pooling
classificador.add(MaxPooling2D(pool_size = (2,2))) #Tamanho da matriz do pool

#Etapa 3 - Flattening
classificador.add(Flatten())

#Etapa 4 - Rede Neural
classificador.add(Dense(units=128, activation='relu', ))
classificador.add(Dense(units = 10, activation='softmax' )) #10 por ser 10 saidas
classificador.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
classificador.fit(previsores_treinamento, classe_treinamento, batch_size = 128, epochs = 5,
                  validation_data = (previsores_teste, classe_teste)) #diz os resultados =D

resultado = classificador.evaluate(previsores_teste, classe_teste)