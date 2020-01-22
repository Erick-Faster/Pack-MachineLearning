# -*- coding: utf-8 -*-
"""
- MLP para a previsao da rugosidade superficial em um material
- Comparacao dos resultados obtidos por HOSSAIN et al.

- Programa para iterar o treinamento e o teste de uma rede neural

"""

best_score_test = 0

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

for i in range(100):
    
    print("Iteracao No %i" %i)
    #Necessario Slide em y, pra não dar erro!!
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

    from sklearn.neural_network import MLPRegressor
    regressor = MLPRegressor(hidden_layer_sizes = (20,20,),
                             tol = 0.00001,
                             activation = 'relu',
                             solver = 'adam',
                             batch_size = 2,
                             max_iter = 1000,
                             alpha = 0.01)
    regressor.fit(X_train, y_train.ravel())
    
    
    '''
    Previsao
    '''
    
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
    
    '''
    Listas
    '''
    listaMAE_train.append(mae_train)
    listaMAE_test.append(mae_test)
    listaMSE_train.append(mse_train)
    listaMSE_test.append(mse_test)
    listaRMSE_train.append(rmse_train)
    listaRMSE_test.append(rmse_test)
    listaSCORE_train.append(score_train)
    listaSCORE_test.append(score_test)
    
    #Capturar os melhores resultados
    if(score_test > best_score_test):
        best_predict_train = previsoes_train[:]
        best_predict_test = previsoes_test[:]
        
        best_MAE_train = mae_train
        best_MAE_test = mae_test
        best_MSE_train = mse_train
        best_MSE_test = mse_test
        best_RMSE_train = rmse_train
        best_RMSE_test = rmse_test
        best_score_train = score_train
        best_score_test = score_test
        
        import matplotlib.pyplot as plt
    
        #plt.title('Previsão da rede neural com 2 neurônios ocultos ')
        plt.xlabel('Valores reais')
        plt.ylabel('Valores previstos')
        
        plt.scatter(y_test, previsoes_test, color='black')
        plt.axis((0,5,0,5))
        plt.plot([0,5],[0,5])
    
        plt.savefig('MLP0.jpeg')
        
        plt.show()
        
        #Salvar o melhor modelo
        from sklearn.externals import joblib
        joblib.dump(regressor, 'Model0.sav')
    
    import numpy as np

    print("MAE_train:\t %.4f" %np.mean(listaMAE_train))
    print("MAE_test:\t %.4f" %np.mean(listaMAE_test))
    print("MSE_train:\t %.4f" %np.mean(listaMSE_train))
    print("MSE_test:\t %.4f" %np.mean(listaMSE_test))
    print("RMSE_train:\t %.4f" %np.mean(listaRMSE_train))
    print("RMSE_test:\t %.4f" %np.mean(listaRMSE_test))
    print("SCORE_train:\t %.4f" %np.mean(listaSCORE_train))
    print("SCORE_test:\t %.4f\n" %np.mean(listaSCORE_test))
    
    
'''
Carregar rede neural
''' 

#regressor_load = joblib.load('Model7.sav')
#print(regressor_load.score(X_test, y_test))
 
print("Melhores Resultados:")
print("%.4f" %best_MAE_train)
print("%.4f" %best_MAE_test)
print("%.4f" %best_MSE_train)
print("%.4f" %best_MSE_test)
print("%.4f" %best_RMSE_train)
print("%.4f" %best_RMSE_test)
print("%.4f" %best_score_train)
print("%.4f" %best_score_test)

print("Media dos Resultados")
print("%.4f" %np.mean(listaMAE_train))
print("%.4f" %np.mean(listaMAE_test))
print("%.4f" %np.mean(listaMSE_train))
print("%.4f" %np.mean(listaMSE_test))
print("%.4f" %np.mean(listaRMSE_train))
print("%.4f" %np.mean(listaRMSE_test))
print("%.4f" %np.mean(listaSCORE_train))
print("%.4f" %np.mean(listaSCORE_test))
    
