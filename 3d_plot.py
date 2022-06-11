# import potrzebnych bibliotek
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


# Dane
data_url = "https://raw.githubusercontent.com/jgrynczewski/tensorflow_intro/master/kc_house_data.csv"
data = pd.read_csv(data_url)

size = np.array(data['sqft_lot'], np.float32)
bedrooms = np.array(data['bedrooms'], np.float32)
price = np.array(data['price'], np.float32)

size_log = np.log(size)
price_log = np.log(price)

# Początkowe wartości parametrów modelu
params = tf.Variable(np.array([10, 0.65, 0.2]))

# model regresji liniowe
def linear_regression(params, feature1=size_log, feature2=bedrooms):
    return params[0] + feature1 * params[1] + feature2 * params[2]

# Funkcja kosztów
def loss_function(params, targets=price_log, feature1=size_log, feature2=bedrooms):
    # prognoza
    predictions = linear_regression(params, feature1, feature2)

    # Średni błąd absolutny
    return tf.keras.losses.mae(targets, predictions)

# Optymalizator Adam
opt = tf.keras.optimizers.Adam()

# Minimalizacja funkcji kosztów + wyświetlenie wartości funkcji kosztów
# dla kolejnych parametrów
for j in range(1000):
    opt.minimize(lambda: loss_function(params), var_list=[params])
    loss = loss_function(params)
    print(loss)
    print(params)
    y_predict = params[0] + params[1] * size_log + params[2] * bedrooms

# Wykres 3d
fig = plt.figure()
ax = plt.axes(projection='3d')

# 10-krotne zmniejszenie liczby punktów wyświetlanych na wykresie
# (w przyśpieszenia interakcji z wykresem)
x_dots = size_log[::10]
y_dots = bedrooms[::10]
z_dots = price_log[::10]

# zbiór punktów
ax.scatter3D(x_dots, y_dots, z_dots, color='black')

x_min = np.min(x_dots)
x_max = np.max(x_dots)

y_min = np.min(y_dots)
y_max = np.max(y_dots)

x = np.linspace(x_min, x_max, 100)
y = np.linspace(y_min, y_max, 100)
z = params[0] + params[1] * x + params[2] * y

# model
ax.plot3D(x, y, z, color='red')

# etykiety osi
ax.set_xlabel('size_log')
ax.set_ylabel('bedrooms')
ax.set_zlabel('price_log')

plt.show()
