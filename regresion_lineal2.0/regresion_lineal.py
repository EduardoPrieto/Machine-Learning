import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

casas = pd.read_csv('USA-Housing.csv')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

x = casa[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]

y = casas['Price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

lrm = LinearRegression()
lrm.fit(x_train, y_train)

from sklearn import metrics

predicciones = lrm.predict(x_test)

plt.scatter(y_test, predicciones)

sns.distplot(y_test - predicciones)

metrics.mean_absolute_error(y_test, predicciones)

metrics.mean_squared_error(y_test, predicciones)

np.sqrt(metrics.mean_squared_error(y_test, predicciones))

