import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

entrenamiento = pd.read_csv('train.csv')

sns.heatmap(entrenamiento.isnull())

sns.boxplot(x='Pclass', y='Age', data=entrenamiento)

def edad_media(columnas):
	edad = columnas[0]
	clase = columnas[1]
	if pd.isnull(edad):
		if clase == 1:
			return 38
		elif clase == 2:
			return 30
		else:
			return 25
	else:
		return edad

entrenamiento['Age'] = entrenamiento[['Age','Pclass']].apply(edad_media, axis=1)

#se borra la columna cabin puesto que tiene demasiados valore nulos
entrenamiento.drop('Cabin', axis=1, inplace=True)

#se borran los valores alfanumericos puesto que estos no se usaran en el analisis
entrenamiento.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)

sexo = pd.get_dummies(entrenamiento['Sex'], drop_first=True)

entrenamiento = pd.concat([entrenamiento, sexo], axis=1)

entrenamiento.drop('Sex', axis=1, inplace=True)

puerto = pd.get_dummies(entrenamiento['Embarked'], drop_first=True)

entrenamiento = pd.concat([entrenamiento, sexo], axis=1)

entrenamiento.drop('Embarked', axis=1, inplace=True)

y = entrenamiento['Survived']
x = entrenamiento.drop('Survived', axis=1)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=45)

from sklearn.linear_model import LogisticRegression

modelo = LogisticRegression()
modelo.fit(x_train, y_train)

prediccion = modelo.predict(x_test)

from sklearn.metrics import classification_report

print(classification_report(y_test, prediccion))

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, prediccion)

