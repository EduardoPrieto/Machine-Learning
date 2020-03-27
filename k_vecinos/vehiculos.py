import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('vehiculos.csv')

y = dataset['vehicle_class']
x = dataset.drop('vehicle_class', axis=1)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=45)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(x_train, y_train)

predicciones = knn.predict(x_test)

#EVALUACION

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, predicciones))

print(classification_report(y_test, predicciones))

#Variacion de parametros para k, optimizacion de las predicciones

tasa_error = []

for i in range(1, 30):
	knn = KNeighborsClassifier(n_neighbors=i)
	knn.fit((x_train,y_train))
	prediccion_i = knn.predict(x_test)
	tasa_error.append(np.mean(prediccion_i != y_test))


valores = range(1,30)
plt.plot(valores, tasa_error, color='green', marker='o', markerfacecolor='red', markersize=8)