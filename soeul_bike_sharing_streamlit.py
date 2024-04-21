import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Disable the warning about using Pyplot without arguments
st.set_option('deprecation.showPyplotGlobalUse', False)

# Load the dataset
df = pd.read_csv('/Users/shivamnath/Downloads/SeoulBikeData.csv')

# Data Preprocessing
dp = df.drop(["Date", "Holiday", "Seasons"], axis=1)
dp.columns = ['bike_count', 'hour', 'temp', 'humidity', 'wind', 'visibility', 'dew_pt_temp', 'radiation', 'rain', 'snow', 'functional']
dp['functional'] = (dp['functional'] == 'Yes').astype(int)
dp = dp[dp['hour'] == 12]
dp = dp.drop(['hour'], axis=1)

# Define temperature range
temperatures = np.linspace(-20, 40, 100).reshape(-1, 1)

# Linear Regression Model
def linear_regression_model(x_train, y_train, x_test):
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return y_pred

# Neural Network Model
def neural_network_model(x_train, y_train, x_test):
    model = Sequential([
    Dense(64, activation='relu', input_shape=(1,)),
    Dense(64, activation='relu'),
    Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=100, verbose=0)
    y_pred = model.predict(x_test)
    return y_pred

# Perform linear regression
x_train_temp = dp['temp'].values.reshape(-1, 1)
y_train_temp = dp['bike_count'].values.reshape(-1, 1)
temp_reg_prediction = linear_regression_model(x_train_temp, y_train_temp, temperatures)

# Neural Network Model
nn_prediction = neural_network_model(x_train_temp, y_train_temp, temperatures)

# Display the predictions

#lr model
st.title('Bike Sharing Predictor')
st.subheader('Linear Regression Model')
st.write(f'Predicted bike count at -20°C: {temp_reg_prediction[0][0]:.2f}')
st.write(f'Predicted bike count at 40°C: {temp_reg_prediction[-1][0]:.2f}')

fig_lr, ax_lr = plt.subplots(figsize=(10, 6))
ax_lr.scatter(dp['temp'], dp['bike_count'], color='blue', label='Data')
ax_lr.plot(temperatures, temp_reg_prediction, color='red', label='Linear Regression')
ax_lr.set_title('Bike Count vs Temperature')
ax_lr.set_xlabel('Temperature (°C)')
ax_lr.set_ylabel('Bike Count')
ax_lr.legend()
ax_lr.grid(True)

st.pyplot(fig_lr)

#nn model
st.subheader('Neural Network Model')
st.write(f'Predicted bike count at -20°C: {nn_prediction[0][0]:.2f}')
st.write(f'Predicted bike count at 40°C: {nn_prediction[-1][0]:.2f}')



fig_nn, ax_nn = plt.subplots(figsize=(10, 6))
ax_nn.scatter(dp['temp'], dp['bike_count'], color='blue', label='Data')
ax_nn.plot(temperatures, nn_prediction, color='green', label='Neural Network')
ax_nn.set_title('Bike Count vs Temperature (Neural Network)')
ax_nn.set_xlabel('Temperature (°C)')
ax_nn.set_ylabel('Bike Count')
ax_nn.legend()
ax_nn.grid(True)

# Display the plots using Streamlit

st.pyplot(fig_nn)
