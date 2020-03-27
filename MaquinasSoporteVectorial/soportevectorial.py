import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

flores = sns.load_dataset('iris')

x = flores.drop('species', axis=1)

y = flores['species']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

from sklearn.svm import SVC

modelo = SVC(gamma='auto')

modelo.fit(x_train, y_train)

predicciones  = modelo.predict(x_test)


from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, predicciones))

print(classification_report(y_test, predicciones))