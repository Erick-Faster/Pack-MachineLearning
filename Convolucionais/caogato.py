# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 21:46:51 2019

@author: Faster-PC
"""

from keras.models import Sequential
from keras.layers import Dense, Flatten, MaxPooling2D, Conv2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.preprocessing import image
import numpy as np

'''[1]REDE NEURAL'''
classificador = Sequential()

#Etapa 1
classificador.add(Conv2D(32,
                         (3,3),
                         input_shape = (64, 64, 3), #64x64 valores maiores p converter dimensoes das imagens. 3 por ser RGB
                         activation = 'relu'))
classificador.add(BatchNormalization()) #Deixar valores entre 0 e 1. Acelera rede

#Etapa 2
classificador.add(MaxPooling2D(pool_size = (2,2)))

#Etapa 3
classificador.add(Flatten())

#Etapa 4
classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 1, activation = 'sigmoid'))

'''[2]COMPILANDO'''
classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

'''[3]GERANDO E CONVERTENDO DADOS'''

#Cria um gerador de dados de treinamento e de teste
gerador_treinamento = ImageDataGenerator(rescale = 1./255, # Valores entre 0 e 1
                                        rotation_range = 7, #Rotacoes
                                        horizontal_flip = True,
                                        shear_range = 0.2,
                                        height_shift_range = 0.07,
                                        zoom_range = 0.2)
gerador_teste = ImageDataGenerator(rescale = 1./255)

#Extrai imagens de treinamento do diretorio selecionado
base_treinamento = gerador_treinamento.flow_from_directory('dataset/training_set',
                                                           target_size = (64,64), #Mesmas dimensoes que a rede
                                                           batch_size = 32,
                                                           class_mode = 'binary')

#Extrai imagens de teste do diretorio selecionado
base_teste = gerador_teste.flow_from_directory('dataset/test_set',
                                               target_size = (64,64),
                                               batch_size = 32,
                                               class_mode = 'binary')

'''[4]TREINANDO REDE'''
classificador.fit_generator(base_treinamento, steps_per_epoch = 4000, #Passos por epoca. Sao 4000 imagens, mas divido por 32 p n demorar mto
                            epochs = 10, validation_data = base_teste,
                            validation_steps = 1000) #Sao 1000 imagens no teste

#Verifica quais classes foram atribuidas cada valor
base_treinamento.class_indices

'''[5]TESTANDO COM UMA SO IMAGEM'''
imagem_teste = image.load_img('dataset/gato3.jpg', target_size = (64,64))
imagem_teste = image.img_to_array(imagem_teste)
imagem_teste /= 255
imagem_teste = np.expand_dims(imagem_teste, axis = 0)
previsao = classificador.predict(imagem_teste)

previsao = float(previsao)
if previsao >= 0.5:
    prob = (previsao-0.5)*2*100
    #prob = str(prob)
    print("Animal: Gato\nProbabilidade: %g%%" %prob)
else:
    prob = (1-(previsao*2))*100
    #prob = str(prob)
    print("Animal: Cachorro\nProbabilidade: %g%%" %prob)

'''[6]SALVANDO REDE'''

#Salvar Rede
classificador_json = classificador.to_json()
with open('classificador_caogato.json', 'w') as json_file:
    json_file.write(classificador_json)

#Salvar pesos
classificador.save_weights('classificador_caogato.h5')