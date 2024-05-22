import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error
import joblib
import matplotlib.pyplot as plt

# Cargar los datos
df = pd.read_csv('dataset/product_prices.csv') 

# Codificar variables categóricas
label_encoder = LabelEncoder()
df['outlet'] = label_encoder.fit_transform(df['outlet'])
df['product_identifier'] = label_encoder.fit_transform(df['product_identifier'])

# Definir características (X) y objetivo (y)
X = df[['outlet', 'product_identifier', 'week_id']]
y = df['sell_price']  # Predecimos sell_price

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Función para mostrar gráficos
def plot_results(y_test, y_pred, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
    plt.xlabel('Actual Sell Price')
    plt.ylabel('Predicted Sell Price')
    plt.title(title)
    plt.show()

# ---- Regresión Lineal ----
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
print('Mean Squared Error (Regresión Lineal):', mse_lr)

# Guardar el modelo
joblib.dump(model_lr, 'model_lr.pkl')

# Mostrar gráfico de resultados
plot_results(y_test, y_pred_lr, 'Linear Regression Results')

# ---- K-Vecinos más Cercanos (KNN) ----
model_knn = KNeighborsRegressor(n_neighbors=5)
model_knn.fit(X_train, y_train)
y_pred_knn = model_knn.predict(X_test)
mse_knn = mean_squared_error(y_test, y_pred_knn)
print('Mean Squared Error (KNN):', mse_knn)

# Guardar el modelo
joblib.dump(model_knn, 'model_knn.pkl')

# Mostrar gráfico de resultados
plot_results(y_test, y_pred_knn, 'K-Nearest Neighbors Results')

# ---- Redes Neuronales ----
model_nn = Sequential()
model_nn.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
model_nn.add(Dense(32, activation='relu'))
model_nn.add(Dense(1))  # Para regresión
model_nn.compile(optimizer='adam', loss='mean_squared_error')

model_nn.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)  # Reduce el número de épocas y aumenta el tamaño del lote para optimizar
loss = model_nn.evaluate(X_test, y_test)
print('Mean Squared Error (Redes Neuronales):', loss)

# Guardar el modelo
model_nn.save('model_nn.h5')

# Predecir y mostrar gráfico de resultados
y_pred_nn = model_nn.predict(X_test)
plot_results(y_test, y_pred_nn, 'Neural Network Results')