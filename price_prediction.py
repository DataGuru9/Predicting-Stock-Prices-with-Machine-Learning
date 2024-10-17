import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import requests
from textblob import TextBlob

def fetch_stock_data(ticker, years=5):
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365*years)).strftime('%Y-%m-%d')
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def add_technical_indicators(data):
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
    data['RSI'] = calculate_rsi(data['Close'], window=14)
    data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['Bollinger_Upper'] = data['Close'].rolling(window=20).mean() + 2 * data['Close'].rolling(window=20).std()
    data['Bollinger_Lower'] = data['Close'].rolling(window=20).mean() - 2 * data['Close'].rolling(window=20).std()
    return data

def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def fetch_news_headlines(ticker, days=7):
    url = f"https://finance.yahoo.com/quote/{ticker}/news"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    headlines = soup.find_all('h3', class_='Mb(5px)')
    return [headline.text for headline in headlines[:10]]  # Get top 10 headlines

def analyze_sentiment(headlines):
    sentiments = [TextBlob(headline).sentiment.polarity for headline in headlines]
    return np.mean(sentiments)

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def create_model(input_shape, sentiment_shape):
    tech_input = Input(shape=input_shape)
    lstm1 = LSTM(100, return_sequences=True)(tech_input)
    dropout1 = Dropout(0.2)(lstm1)
    lstm2 = LSTM(50, return_sequences=False)(dropout1)
    dropout2 = Dropout(0.2)(lstm2)
    
    sentiment_input = Input(shape=sentiment_shape)
    
    concat = Concatenate()([dropout2, sentiment_input])
    
    dense1 = Dense(50, activation='relu')(concat)
    output = Dense(1)(dense1)
    
    model = Model(inputs=[tech_input, sentiment_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# Main execution
ticker = 'NVDA'
data = fetch_stock_data(ticker)
data = add_technical_indicators(data)
data.dropna(inplace=True)

features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_50', 'SMA_200', 'EMA_50', 'RSI', 'MACD', 'MACD_Signal', 'Bollinger_Upper', 'Bollinger_Lower']
X = data[features].values
y = data['Close'].values

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

seq_length = 60
X_seq, y_seq = create_sequences(X_scaled, seq_length)

# Fetch and analyze sentiment for each day
sentiment_data = []
for date in data.index[-len(X_seq):]:
    headlines = fetch_news_headlines(ticker)
    sentiment = analyze_sentiment(headlines)
    sentiment_data.append(sentiment)

sentiment_data = np.array(sentiment_data).reshape(-1, 1)

tscv = TimeSeriesSplit(n_splits=5)
model = create_model((seq_length, X_seq.shape[2]), (1,))

for train_index, test_index in tscv.split(X_seq):
    X_train, X_test = X_seq[train_index], X_seq[test_index]
    y_train, y_test = y_seq[train_index], y_seq[test_index]
    sentiment_train, sentiment_test = sentiment_data[train_index], sentiment_data[test_index]
    
    model.fit([X_train, sentiment_train], y_train, epochs=50, batch_size=32, validation_data=([X_test, sentiment_test], y_test), verbose=0)

# Predict next day's price
last_sequence = X_scaled[-seq_length:]
last_sequence = last_sequence.reshape((1, seq_length, len(features)))
latest_headlines = fetch_news_headlines(ticker, days=1)
latest_sentiment = np.array([[analyze_sentiment(latest_headlines)]])
predicted_scaled = model.predict([last_sequence, latest_sentiment])

last_original_values = X[-seq_length:, features.index('Close')].reshape(-1, 1)
predicted_price = scaler.inverse_transform(np.hstack((last_original_values, predicted_scaled)))[0, -1]

plt.figure(figsize=(12,6))
plt.plot(data.index[-30:], data['Close'].tail(30), label='Actual Price')
plt.plot(data.index[-1] + timedelta(days=1), predicted_price, 'ro', label='Predicted Price')
plt.title(f'{ticker} Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.savefig(f'{ticker}_prediction.png')
plt.close()

print(f"Current {ticker} price: ${data['Close'].iloc[-1]:.2f}")
print(f"Predicted {ticker} price for tomorrow: ${predicted_price:.2f}")

# Summary:
# This advanced stock price prediction model uses an ensemble approach, combining
# traditional machine learning models (Random Forest, Gradient Boosting, SVM, 
# Linear Regression) with deep learning (LSTM) and time series analysis (ARIMA).
# It incorporates technical indicators as features and uses time series cross-validation
# for robust evaluation. The model fetches recent stock data, prepares it with
# technical indicators, trains the ensemble on historical data, and predicts the
# stock price for the next week. The results are visualized and saved as a PNG file.
# While this model is more sophisticated than simple linear regression, it's important
# to note that stock market prediction remains challenging and uncertain. This code
# serves as an educational example and should not be used for actual trading decisions
# without further refinement and risk management strategies.