# -*- coding: utf-8 -*-
"""
[Parte 3] - Metodos de Regressao

Metodo de Regressao linear Multipla para prever preco das casas
Utiliza varios atributos para prever os resultados

Resultado: Score: 0,70. Um resultado melhor do que a Simples
"""

listaConfig = []
listaMAE_train = []
listaMAE_test = []
listaMSE_train = []
listaMSE_test = []
listaRMSE_train = []
listaRMSE_test = []
listaSCORE_train = []
listaSCORE_test = []


'''
Pre-Processamento
'''
import pandas as pd

base = pd.read_excel('Hossain.xlsx', encoding = 'ISO-8859-1')

#Necessario Slide em y, pra não dar erro!!

#activation
x1 = ['logistic', 'tanh', 'relu']

#solver
x2 = ['lbfgs', 'sgd', 'adam']

#neuronios
x3 = []
for i in range (1,3):
    tupla = (i,)
    x3.append(tupla)
    for j in range (1,3):
        tupla = (i,j,)
        x3.append(tupla)
        for k in range (1,3):
            tupla = (i,j,k,)
            x3.append(tupla)
            for l in range (1,3):
                tupla = (i,j,k,l,)
                x3.append(tupla)
                for m in range (1,3):
                    tupla = (i,j,k,l,m,)
                    x3.append(tupla)

#alpha
x4 = [0.0001, 0.005, 0.001, 0.005, 0.01]

#batch_size
x5 = [2,3, 5, 8, 10]

for y1 in x1:
    for y2 in x2:
        for y3 in x3:
            for y4 in x4:
                for y5 in x5:
                    
                    
                    
                    X_train = base.iloc[0:68, 1:6].values
                    y_train = base.iloc[0:68:,6:7].values #slide
                    
                    X_test = base.iloc[68:84,1:6].values
                    y_test = base.iloc[68:84,6:7].values #slide
                    
                    from sklearn.preprocessing import StandardScaler
                    scaler_x_train = StandardScaler()
                    X_train = scaler_x_train.fit_transform(X_train)
                    scaler_x_test = StandardScaler()
                    X_test = scaler_x_test.fit_transform(X_test)
                    
                    #Necessario Escalonar
                    scaler_y_train = StandardScaler()
                    y_train = scaler_y_train.fit_transform(y_train)
                    scaler_y_test = StandardScaler()
                    y_test = scaler_y_test.fit_transform(y_test)
                    
                    '''GRID SEARCH'''
                    '''
                    from sklearn.neural_network import MLPRegressor
                    
                    #Dados Fixos
                    mlp = MLPRegressor(max_iter=1000, tol=0.00001)
                    
                    #Dados Variaveis
                    parameter_space = {
                        'hidden_layer_sizes': [(100,), (96,), (92,), (88,), (84,), 
                                               (80,), (76,), (72,), (68,), (64,), 
                                               (60,), (56,), (52,), (48,), (44,), 
                                               (40,), (36,), (32,), (28,), (24,), 
                                               (20,), (18,), (16,), (14,), (12,), 
                                               (10,), (9,), (8,), (7,), (6,), 
                                               (5,), (4,), (3,), (2,), (1,) ],
                        'activation': ['tanh', 'relu'],
                        'solver': ['sgd', 'adam', 'lbfgs' ],
                        'alpha': [0.0001, 0.001, 0.01]
                    }
                    
                    #Grid Search
                    from sklearn.model_selection import GridSearchCV
                    regressor = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
                    regressor.fit(X_train, y_train)
                    
                    
                    #Resultados
                    
                    medias = regressor.cv_results_['mean_test_score']
                    desvios = regressor.cv_results_['std_test_score']
                    ranks = regressor.cv_results_['rank_test_score']
                    for media, desvio, rank, params in zip(medias, desvios, ranks, regressor.cv_results_['params']):
                        print("Parametros: %r \nRank:%d, %0.3f (+-%0.03f)\n" %(params, rank, media, desvio))
                    
                    #Print Melhor Parametro
                    print("\n***********************")
                    print('Melhor Parametro:\n', regressor.best_params_)
                    print('Score: ',regressor.best_score_)
                    print('Index: ',regressor.best_index_)
                    print('Estimator: ',regressor.best_estimator_)
                    print("***********************\n")
                    '''
                    '''
                    Processamento
                    '''
                
                    
                    
                    print('ok')
                
                    
                    from sklearn.neural_network import MLPRegressor
                    regressor = MLPRegressor(hidden_layer_sizes = y3,
                                             tol = 0.00001,
                                             activation = y1,
                                             solver = y2,
                                             batch_size = y5,
                                             max_iter = 1000,
                                             alpha = y4)
                    regressor.fit(X_train, y_train)
                    
                    
                    '''
                    Previsao
                    '''
                    
                    #Assegurar de que a melhor configuracao seja captada
                    #regressor = regressor.best_estimator_
                    
                    #Calculo do Score - Deve ocorrer antes do desescalonamento
                    score_train = regressor.score(X_train, y_train)
                    score_test = regressor.score(X_test, y_test)
                    
                    previsoes_train = regressor.predict(X_train)
                    previsoes_test = regressor.predict(X_test)
                    
                    #Necessario Desescalonar
                    y_train = scaler_y_train.inverse_transform(y_train)
                    y_test = scaler_y_test.inverse_transform(y_test)
                    
                    previsoes_train = scaler_y_train.inverse_transform(previsoes_train)
                    previsoes_test = scaler_y_test.inverse_transform(previsoes_test)
                    
                    '''
                    Tratamento dos Dados
                    '''
                    from sklearn.metrics import mean_absolute_error, mean_squared_error
                    from math import sqrt
                    
                    #Dados estatisticos do treinamento
                    mae_train = mean_absolute_error(y_train, previsoes_train)
                    mse_train = mean_squared_error(y_train, previsoes_train)
                    rmse_train = sqrt(mean_squared_error(y_train, previsoes_train))
                    
                    #Dados estatisticos dos testes
                    mae_test = mean_absolute_error(y_test, previsoes_test)
                    mse_test = mean_squared_error(y_test, previsoes_test)
                    rmse_test = sqrt(mean_squared_error(y_test, previsoes_test))
                    
                    import matplotlib.pyplot as plt
                    
                    #plt.title('Previsão da rede neural com 2 neurônios ocultos ')
                    plt.xlabel('Valores reais')
                    plt.ylabel('Valores previstos')
                    
                    plt.scatter(y_test, previsoes_test, color='black')
                    plt.axis((0,5,0,5))
                    plt.plot([0,5],[0,5])
                    plt.savefig('grafico.jpeg')
                    
                    plt.show()
                    
                    
                    
                    
                    
                    
                    '''
                    Listas
                    '''
                    print("x1 = %s, x2 = %s, x3 = %s, x4 = %s, x5 = %s" %(y1,y2,y3,y4,y5))
                    listaConfig.append("x1 = %s, x2 = %s, x3 = %s, x4 = %s, x5 = %s" %(y1,y2,y3,y4,y5))
                    listaMAE_train.append(mae_train)
                    listaMAE_test.append(mae_test)
                    listaMSE_train.append(mse_train)
                    listaMSE_test.append(mse_test)
                    listaRMSE_train.append(rmse_train)
                    listaRMSE_test.append(rmse_test)
                    listaSCORE_train.append(score_train)
                    listaSCORE_test.append(score_test)
                    
                    #import numpy as np
                    
                    '''
                    print("%.4f" %np.mean(listaMAE_train))
                    print("%.4f" %np.mean(listaMAE_test))
                    print("%.4f" %np.mean(listaMSE_train))
                    print("%.4f" %np.mean(listaMSE_test))
                    print("%.4f" %np.mean(listaRMSE_train))
                    print("%.4f" %np.mean(listaRMSE_test))
                    print("%.4f" %np.mean(listaSCORE_train))
                    print("%.4f" %np.mean(listaSCORE_test))
                    '''
    
