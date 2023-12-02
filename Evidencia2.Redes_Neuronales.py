from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


inmu = pd.read_csv("Clusters.csv")
inmu.info()

df = inmu

print("MODELO 1")

X = df[["m2_construido","Baños","Recamaras","Lugares_estac"]]
y = df["Precio_m2"]

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=413422)

# Escalar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Definir la arquitectura de la red neuronal para regresión
model = Sequential()
model.add(Dense(units=35, activation='relu', input_dim=4))
model.add(Dense(units=35))

model.add(Dense(units=1, activation='linear'))

# Compilar el modelo para regresión
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo para regresión
model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.1)

# Evaluar el modelo en el conjunto de prueba
loss = model.evaluate(X_test_scaled, y_test)
print(f'Error Cuadrático Medio en el conjunto de prueba: {loss:.2f}')

# Realizar predicciones en el conjunto de prueba
predictions = model.predict(X_test_scaled)

# Visualizar algunas predicciones
for i in range(10):
    print(f'Predicción: {predictions[i][0]:.2f}, Valor Real: {y_test.iloc[i]:.2f}')

# Visualizar los resultados de la predicción con la línea de referencia
plt.scatter(y_test, predictions, label='Datos')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red', label='Línea de Referencia')
plt.title('Predicciones vs. Valores Reales')
plt.xlabel('Valor Real')
plt.ylabel('Predicción')
plt.legend()
plt.show()

# Calcular métricas
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Mostrar las métricas
print(f'Error Cuadrático Medio (MSE): {mse:.2f}')
print(f'Error Absoluto Medio (MAE): {mae:.2f}')
print(f'Coeficiente de Determinación (R^2): {r2:.2f}')

# Calcular estadísticas descriptivas de las predicciones
mean_prediction = np.mean(predictions)
median_prediction = np.median(predictions)
std_dev_prediction = np.std(predictions)
min_prediction = np.min(predictions)
max_prediction = np.max(predictions)

mean_real = np.mean(y)
median_real = np.median(y)
std_dev_real = np.std(y)
min_real= np.min(y)
max_real = np.max(y)

# Mostrar las estadísticas
print(f'Media de las Predicciones: {mean_prediction:.2f}')
print(f'Mediana de las Predicciones: {median_prediction:.2f}')
print(f'Desviación Estándar de las Predicciones: {std_dev_prediction:.2f}')
print(f'Mínimo de las Predicciones: {min_prediction:.2f}')
print(f'Máximo de las Predicciones: {max_prediction:.2f}')

print(f'Media de datos reales: {mean_real:.2f}')
print(f'Mediana de datos reales: {median_real:.2f}')
print(f'Desviación Estándar de datos reales: {std_dev_real:.2f}')
print(f'Mínimo de datos reales: {min_real:.2f}')
print(f'Máximo de datos reales: {max_real:.2f}')

print("MODELO 2/n _____________________________________________________________________________________________________________________________________________________________")

X = inmu[["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10", "Cocina_equip", "Gimnasio", "Amueblado", "Alberca", "Terraza", "Elevador", "Baños", "Recamaras", "Lugares_estac"]]
y = inmu["Precio_m2"]

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=413422)

# Escalar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
 
# Definir la arquitectura de la red neuronal para regresión
model = Sequential()
model.add(Dense(units=35, activation='relu', input_dim=19))
model.add(Dense(units=35))

model.add(Dense(units=1, activation='linear'))

# Compilar el modelo para regresión
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo para regresión
model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.1)

# Evaluar el modelo en el conjunto de prueba
loss = model.evaluate(X_test_scaled, y_test)
print(f'Error Cuadrático Medio en el conjunto de prueba: {loss:.2f}')

# Realizar predicciones en el conjunto de prueba
predictions = model.predict(X_test_scaled)

# Visualizar algunas predicciones
for i in range(10):
    print(f'Predicción: {predictions[i][0]:.2f}, Valor Real: {y_test.iloc[i]:.2f}')

# Calcular métricas
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Mostrar las métricas
print(f'Error Cuadrático Medio (MSE): {mse:.2f}')
print(f'Error Absoluto Medio (MAE): {mae:.2f}')
print(f'Coeficiente de Determinación (R^2): {r2:.2f}')

print("MODELO 3/n _____________________________________________________________________________________________________________________________________________________________")

X = inmu[["X2", "X3", "X5","X8", "X10", "Alberca", "Baños", "Recamaras", "Lugares_estac"]]
y = inmu["Precio_m2"]

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=413422)

# Escalar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
 
# Definir la arquitectura de la red neuronal para regresión
model = Sequential()
model.add(Dense(units=35, activation='relu', input_dim=9))
model.add(Dense(units=35))

model.add(Dense(units=1, activation='linear'))

# Compilar el modelo para regresión
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo para regresión
model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.1)

# Evaluar el modelo en el conjunto de prueba
loss = model.evaluate(X_test_scaled, y_test)
print(f'Error Cuadrático Medio en el conjunto de prueba: {loss:.2f}')

# Realizar predicciones en el conjunto de prueba
predictions = model.predict(X_test_scaled)

# Visualizar algunas predicciones
for i in range(10):
    print(f'Predicción: {predictions[i][0]:.2f}, Valor Real: {y_test.iloc[i]:.2f}')

# Calcular métricas
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Mostrar las métricas
print(f'Error Cuadrático Medio (MSE): {mse:.2f}')
print(f'Error Absoluto Medio (MAE): {mae:.2f}')
print(f'Coeficiente de Determinación (R^2): {r2:.2f}')

print("MODELO 4/n _____________________________________________________________________________________________________________________________________________________________")

X = inmu[["Gimnasio", "Alberca", "Terraza", "Elevador", "Baños", "Recamaras", "Lugares_estac"]]
y = inmu["Precio_m2"]

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=413422)

# Escalar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
 
# Definir la arquitectura de la red neuronal para regresión
model = Sequential()
model.add(Dense(units=35, activation='relu', input_dim=7))
model.add(Dense(units=35))

model.add(Dense(units=1, activation='linear'))

# Compilar el modelo para regresión
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo para regresión
model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.1)

# Evaluar el modelo en el conjunto de prueba
loss = model.evaluate(X_test_scaled, y_test)
print(f'Error Cuadrático Medio en el conjunto de prueba: {loss:.2f}')

# Realizar predicciones en el conjunto de prueba
predictions = model.predict(X_test_scaled)

# Visualizar algunas predicciones
for i in range(10):
    print(f'Predicción: {predictions[i][0]:.2f}, Valor Real: {y_test.iloc[i]:.2f}')

# Calcular métricas
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Mostrar las métricas
print(f'Error Cuadrático Medio (MSE): {mse:.2f}')
print(f'Error Absoluto Medio (MAE): {mae:.2f}')
print(f'Coeficiente de Determinación (R^2): {r2:.2f}')

print("MODELO 5/n _____________________________________________________________________________________________________________________________________________________________")

X = inmu[["Gimnasio", "Alberca", "Terraza", "Baños", "Lugares_estac"]]
y = inmu["Precio_m2"]

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=413422)

# Escalar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
 
# Definir la arquitectura de la red neuronal para regresión
model = Sequential()
model.add(Dense(units=35, activation='relu', input_dim=5))
model.add(Dense(units=35))

model.add(Dense(units=1, activation='linear'))

# Compilar el modelo para regresión
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo para regresión
model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.1)

# Evaluar el modelo en el conjunto de prueba
loss = model.evaluate(X_test_scaled, y_test)
print(f'Error Cuadrático Medio en el conjunto de prueba: {loss:.2f}')

# Realizar predicciones en el conjunto de prueba
predictions = model.predict(X_test_scaled)

# Visualizar algunas predicciones
for i in range(10):
    print(f'Predicción: {predictions[i][0]:.2f}, Valor Real: {y_test.iloc[i]:.2f}')

# Calcular métricas
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Mostrar las métricas
print(f'Error Cuadrático Medio (MSE): {mse:.2f}')
print(f'Error Absoluto Medio (MAE): {mae:.2f}')
print(f'Coeficiente de Determinación (R^2): {r2:.2f}')
