# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 22:21:30 2019

@author: Cristian
"""

import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd

dataset = pd.read_csv('salario.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

#set de entrenamiento y prueba
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)

#Adaptación de una regresión lineal al set de pruebas

from sklearn.linear_model import LinearRegression
regresor = LinearRegression()
regresor.fit(x_train, y_train)

#Prediciendo resultadosdel set de pruebas
y_pred = regresor.predict(x_test)

#Visualizacion de resultados (set entrenamiento)

plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, regresor.predict(x_train), color='blue')
plt.title('Salario vs Años de experiencia (set de Prueba)')
plt.xlabel('Años de experiencia')
plt.ylabel('salario')
plt.show()

#Visualizacion de resultados (set prueba)
