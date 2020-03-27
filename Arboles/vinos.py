import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

vinos = pd.read_csv('vino.csv')

y = vinos['Wine Type']
x = vinos.drop('Wine Type', axis=1)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

from sklearn.tree import DecisionTreeClassifier

arbol = DecisionTreeClassifier()

arbol.fit(x_train, y_train)

predicciones  = arbol.predict(x_test)


from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, predicciones))

print(classification_report(y_test, predicciones))