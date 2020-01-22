# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 00:16:29 2019

@author: Faster-PC
"""

import pandas as pd

base = pd.read_csv('autos.csv', encoding = 'ISO-8859-1')

'''
PRE-PROCESSAMENTO
'''

#[1]Eliminar colunas inuteis
base = base.drop('dateCrawled', axis = 1) #Axis 1 - apaga a coluna inteira
base = base.drop('dateCreated', axis = 1)
base = base.drop('nrOfPictures', axis = 1)
base = base.drop('postalCode', axis = 1)
base = base.drop('lastSeen', axis = 1)

base['name'].value_counts() #Verificar se mantenho ou nao os atributos 
base = base.drop('name', axis = 1)
base = base.drop('seller', axis = 1)
base = base.drop('offerType', axis = 1)

#[2]Eliminar dados defeituosos (precos mto baixo ou mto alto)
i1 = base.loc[base.price <= 10] #Pega todas as linhas cujo preco <= 10
base = base[base.price > 10] #Mantenho somente valores de preco > 10

i2 = base.loc[base.price > 350000]
base = base[base.price < 350000]

#[3]Substituir valores faltantes por valor que mais aparece

base.loc[pd.isnull(base.vehicleType)] #Verifica os nulls em dada coluna
base['vehicleType'].value_counts() #Conta qtos dados tenho (limousine)
base.loc[pd.isnull(base.gearbox)]
base['gearbox'].value_counts() #manuell
base.loc[pd.isnull(base.model)]
base['model'].value_counts() #golf
base.loc[pd.isnull(base.fuelType)]
base['fuelType'].value_counts() #benzin
base.loc[pd.isnull(base.notRepairedDamage)]
base['notRepairedDamage'].value_counts() #nein

valores = {'vehicleType': 'limousine',
           'gearbox': 'manuell',
           'model':'golf',
           'fuelType':'benzin',
           'notRepairedDamage':'nein'} #crio dicionario com os valores que quero substituir

base = base.fillna(value = valores) #Substitui por valores do dicionario

#[4]Tratar dados usando LabelEncoder

previsores = base.iloc[:, 1:13].values #previsores
preco_real = base.iloc[:,0].values #classe

from sklearn.preprocessing import LabelEncoder 
labelencoder_previsores = LabelEncoder() #Eliminar as strings
previsores[:,0] = labelencoder_previsores.fit_transform(previsores[:,0])
previsores[:,1] = labelencoder_previsores.fit_transform(previsores[:,1])
previsores[:,3] = labelencoder_previsores.fit_transform(previsores[:,3])
previsores[:,5] = labelencoder_previsores.fit_transform(previsores[:,5])
previsores[:,8] = labelencoder_previsores.fit_transform(previsores[:,8])
previsores[:,9] = labelencoder_previsores.fit_transform(previsores[:,9])
previsores[:,10] = labelencoder_previsores.fit_transform(previsores[:,10])

#[5]Transformar em variaveis Dummy usando OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0,1,3,5,8,9,10])
previsores = onehotencoder.fit_transform(previsores).toarray()

'''
REDE NEURAL
'''

from keras.models import Sequential
from keras.layers import Dense

regressor = Sequential()
regressor.add(Dense(units = 158, activation = 'relu', input_dim=316)) #316 p causa das dummys
regressor.add(Dense(units = 158, activation = 'relu'))
regressor.add(Dense(units = 1, activation = 'linear')) #linear n faz nenhum calculo adicional

regressor.compile(loss = 'mean_absolute_error', optimizer = 'adam', metrics = ['mean_absolute_error'])
regressor.fit(previsores, preco_real, batch_size = 300, epochs = 100)

previsores = regressor.predict(previsores)