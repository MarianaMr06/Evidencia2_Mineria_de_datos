#%%
#Librerías
import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
#%%
#Carga de archivos
inmu = pd.read_csv("Clusters.csv")
inmu.info()
#%%
#Modelo 1 (Todas las variables)
model = LinearRegression()
type(model)

x = inmu[["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10", "Cocina_equip", "Gimnasio", "Amueblado", "Alberca", "Terraza", "Elevador", "Baños", "Recamaras", "Lugares_estac"]]
y = inmu["Precio_m2"]

model.fit(X = x, y = y)
model.__dict__
#Coeficiente de determinación
determinacion = model.score(x, y)
correlacion = np.sqrt(determinacion)
print("Determinacion:", determinacion)
print("Correlación: ", correlacion)

# Agrega una constante al conjunto de datos (intercepto)
x_with_intercept = sm.add_constant(x)

# Ajusta el modelo
model = sm.OLS(y, x_with_intercept).fit()

# Imprime un resumen del modelo que incluye valores p
print(model.summary())
#%%
#Modelo 2 (Con significantes)
model = LinearRegression()
type(model)

x = inmu[["X2", "X3", "X5","X8", "X10", "Alberca", "Baños", "Recamaras", "Lugares_estac"]]
y = inmu["Precio_m2"]

model.fit(X = x, y = y)
model.__dict__
#Coeficiente de determinación
determinacion = model.score(x, y)
correlacion = np.sqrt(determinacion)
print("Determinacion:", determinacion)
print("Correlación: ", correlacion)

# Agrega una constante al conjunto de datos (intercepto)
x_with_intercept = sm.add_constant(x)

# Ajusta el modelo
model = sm.OLS(y, x_with_intercept).fit()

# Imprime un resumen del modelo que incluye valores p
print(model.summary())

#%%
#Modelo 3 (Con correlaciones más altas)
# data_selected = inmu[["Precio_m2", "X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10", "Cocina_equip", "Gimnasio", "Amueblado", "Alberca", "Terraza", "Elevador", "Baños", "Recamaras", "Lugares_estac"]]

# # Calcula la matriz de correlación
# correlation_matrix = data_selected.corr()

# # Crea el heatmap con seaborn
# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
# plt.title("Heatmap de Correlación entre Variables Seleccionadas")
# plt.show()

model = LinearRegression()
type(model)

x = inmu[["Gimnasio", "Alberca", "Terraza", "Elevador", "Baños", "Recamaras", "Lugares_estac"]]
y = inmu["Precio_m2"]

model.fit(X = x, y = y)
model.__dict__
#Coeficiente de determinación
determinacion = model.score(x, y)
correlacion = np.sqrt(determinacion)
print("Determinacion:", determinacion)
print("Correlación: ", correlacion)

# Agrega una constante al conjunto de datos (intercepto)
x_with_intercept = sm.add_constant(x)

# Ajusta el modelo
model = sm.OLS(y, x_with_intercept).fit()

# Imprime un resumen del modelo que incluye valores p
print(model.summary())

#%%
#Modelo 4 (Con significantes de la anterior)
model = LinearRegression()
type(model)

x = inmu[["Gimnasio", "Alberca", "Terraza", "Baños", "Lugares_estac"]]
y = inmu["Precio_m2"]

model.fit(X = x, y = y)
model.__dict__
#Coeficiente de determinación
determinacion = model.score(x, y)
correlacion = np.sqrt(determinacion)
print("Determinacion:", determinacion)
print("Correlación: ", correlacion)

# Agrega una constante al conjunto de datos (intercepto)
x_with_intercept = sm.add_constant(x)

# Ajusta el modelo
model = sm.OLS(y, x_with_intercept).fit()

# Imprime un resumen del modelo que incluye valores p
print(model.summary())