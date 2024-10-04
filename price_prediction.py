import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import yfinance as yf
from datetime import datetime, timedelta

## Data Acquisition
# Fetch historical stock data for Nvidia (NVDA) from Yahoo Finance
ticker = 'NVDA'
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=1460)).strftime('%Y-%m-%d')  # Roughly 4 years of data
data = yf.download(ticker, start=start_date, end=end_date)

## Data Preparation
# Create a 'Prediction' column by shifting 'Close' prices 7 days into the future
data['Prediction'] = data['Close'].shift(-7)

# Prepare feature (X) and target (y) datasets
X = np.array(data['Close']).reshape(-1, 1)
y = np.array(data['Prediction'])

# Remove last 7 rows containing NaN values
X = X[:-7]
y = y[:-7]

## Model Training
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

## Model Evaluation
# Make predictions on the test set
predictions = model.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

## Prediction for Next Week
last_price = data['Close'].iloc[-1]
next_week_prediction = model.predict([[last_price]])[0]

## Visualization
plt.figure(figsize=(10,6))
plt.plot([0, 1], [last_price, next_week_prediction], marker='o')
plt.title(f'{ticker} Stock Price Prediction for Next Week')
plt.xlabel('Time')
plt.ylabel('Price')
plt.xticks([0, 1], ['Current', 'Next Week'])
plt.grid(True)

# Save the plot as a PNG file
plt.savefig(f'{ticker}_next_week_prediction.png')
plt.close()

print(f"Current {ticker} price: ${last_price:.2f}")
print(f"Predicted {ticker} price for next week: ${next_week_prediction:.2f}")

'''
Summary:
This code implements a simple stock price prediction model using Linear Regression.
It fetches historical data for Nvidia (NVDA) stock, prepares the data by creating
a target variable shifted 7 days into the future, trains a Linear Regression model,
and predicts the stock price for the next week. The prediction is visualized and
saved as a PNG file in the codebase. This serves as a basic example of time series
forecasting in finance, though more sophisticated models and feature engineering
would be needed for real-world applications.
'''