# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import os
import tensorflow as tf
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Suppress warnings and TensorFlow logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

## Data Acquisition and Preparation
# Function to fetch stock data from Yahoo Finance
def fetch_stock_data(ticker, years=5):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*years)
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Function to add technical indicators to the stock data
def add_technical_indicators(data):
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
    data['RSI'] = calculate_rsi(data['Close'], window=14)
    return data

# Function to calculate Relative Strength Index (RSI)
def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

## Model Creation
# Function to create an LSTM model
def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

## Ensemble Model
# Class to create an ensemble of different models
class EnsembleModel:
    def __init__(self):
        self.models = [
            RandomForestRegressor(n_estimators=100, random_state=42),
            GradientBoostingRegressor(n_estimators=100, random_state=42),
            SVR(kernel='rbf'),
            LinearRegression()
        ]
        self.lstm_model = None
        self.arima_model = None

    # Method to fit all models in the ensemble
    def fit(self, X, y, X_lstm=None):
        for model in self.models:
            model.fit(X, y)
        
        if X_lstm is not None:
            self.lstm_model = create_lstm_model((X_lstm.shape[1], X_lstm.shape[2]))
            self.lstm_model.fit(X_lstm, y, epochs=50, batch_size=32, verbose=0)
        
        try:
            self.arima_model = ARIMA(y, order=(1,1,1))
            self.arima_model = self.arima_model.fit()
        except:
            print("ARIMA model fitting failed. Skipping ARIMA predictions.")

    # Method to make predictions using all fitted models
    def predict(self, X, X_lstm=None):
        predictions = [model.predict(X) for model in self.models]
        if X_lstm is not None and self.lstm_model is not None:
            lstm_pred = self.lstm_model.predict(X_lstm)
            predictions.append(lstm_pred.flatten())
        if self.arima_model is not None:
            try:
                arima_pred = self.arima_model.forecast(steps=len(X))
                predictions.append(arima_pred)
            except:
                print("ARIMA prediction failed. Skipping ARIMA predictions.")
        return np.mean(predictions, axis=0)

## Main Execution
# Set the stock ticker and fetch data
ticker = 'NVDA'
data = fetch_stock_data(ticker)
data = add_technical_indicators(data)
data.dropna(inplace=True)

# Prepare features and target variable
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_50', 'SMA_200', 'EMA_50', 'RSI']
X = data[features].values
y = data['Close'].values

# Scale the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Prepare data for LSTM (reshape to 3D)
X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Perform time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)
ensemble = EnsembleModel()

mse_scores = []
mae_scores = []

# Train and evaluate the model using time series cross-validation
for train_index, test_index in tscv.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]
    X_lstm_train, X_lstm_test = X_lstm[train_index], X_lstm[test_index]

    ensemble.fit(X_train, y_train, X_lstm_train)
    predictions = ensemble.predict(X_test, X_lstm_test)

    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    mse_scores.append(mse)
    mae_scores.append(mae)

# Print average error metrics
print(f"Average MSE: {np.mean(mse_scores)}")
print(f"Average MAE: {np.mean(mae_scores)}")

# Predict next week's price
last_data = X_scaled[-1].reshape(1, -1)
last_data_lstm = last_data.reshape(1, 1, -1)
next_week_prediction = ensemble.predict(last_data, last_data_lstm)[0]

# Visualize the prediction
plt.figure(figsize=(12,6))
plt.plot(data.index[-30:], data['Close'].tail(30), label='Actual Price')
plt.plot(data.index[-1] + timedelta(days=7), next_week_prediction, 'ro', label='Predicted Price')
plt.title(f'{ticker} Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.savefig(f'{ticker}_prediction.png')
plt.close()

# Print current and predicted prices
print(f"Current {ticker} price: ${data['Close'].iloc[-1]:.2f}")
print(f"Predicted {ticker} price for next week: ${next_week_prediction:.2f}")

'''
Summary:
This advanced stock price prediction model uses an ensemble approach, combining
traditional machine learning models (Random Forest, Gradient Boosting, SVM, 
Linear Regression) with deep learning (LSTM) and time series analysis (ARIMA).
It incorporates technical indicators as features and uses time series cross-validation
for robust evaluation. The model fetches recent stock data, prepares it with
technical indicators, trains the ensemble on historical data, and predicts the
stock price for the next week. The results are visualized and saved as a PNG file.
While this model is more sophisticated than simple linear regression, it's important
to note that stock market prediction remains challenging and uncertain. This code
serves as an educational example and should not be used for actual trading decisions
without further refinement and risk management strategies.
'''
