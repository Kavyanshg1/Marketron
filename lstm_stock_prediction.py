import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import datetime

# Fetch historical stock data for Amazon
ticker = 'AMZN'
start_date = (datetime.datetime.now() - datetime.timedelta(days=180)).strftime('%Y-%m-%d')
end_date = datetime.datetime.now().strftime('%Y-%m-%d')
data = yf.download(ticker, start=start_date, end=end_date)

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, short_window=12, long_window=26, signal_window=9):
    short_ema = series.ewm(span=short_window, adjust=False).mean()
    long_ema = series.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd - signal


# Feature Engineering
data['5_MA'] = data['Adj Close'].rolling(window=5).mean()
data['10_MA'] = data['Adj Close'].rolling(window=10).mean()
data['RSI'] = compute_rsi(data['Adj Close'])
data['MACD'] = compute_macd(data['Adj Close'])
data['Volume'] = data['Volume']

# Create lagged features
for lag in range(1, 6):
    data[f'lag_{lag}'] = data['Adj Close'].shift(lag)

data.dropna(inplace=True)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Prepare training and validation sets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
val_data = scaled_data[train_size:]

# Create datasets for LSTM
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), :-1])  # Features
        y.append(data[i + time_step, -1])       # Target (next value)
    return np.array(X), np.array(y)

X_train, y_train = create_dataset(train_data, time_step=1)
X_val, y_val = create_dataset(val_data, time_step=1)

# Reshape input to be [samples, time steps, features]
# Assuming you want to keep the last column as target and have features in the rest
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])  # Ensure you have the correct features
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2])
# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Make predictions
predictions = model.predict(X_val)
predictions = scaler.inverse_transform(np.concatenate((predictions, np.zeros((predictions.shape[0], data.shape[1] - 1))), axis=1))[:, 0]

# Calculate performance metrics
rmse = np.sqrt(mean_squared_error(y_val, predictions))
mae = mean_absolute_error(y_val, predictions)

# Get the actual dates for plotting
valid_dates = data.index[train_size:len(data)-1]  # Adjust index range to match predictions

# Plotting the results
plt.figure(figsize=(14, 7))
plt.plot(valid_dates, predictions, color='blue', label='Predicted Prices')
plt.plot(valid_dates, data['Adj Close'][train_size:len(data)-1], color='red', label='Actual Prices')
plt.title('Amazon Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

print(f'RMSE: {rmse}, MAE: {mae}')
