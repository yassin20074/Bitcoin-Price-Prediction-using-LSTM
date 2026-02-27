# calling the libraries
import yfinance as yf # to download  historical stock or crypto data directly 
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# fuction to fetch historical price data
def fetch_data(ticker="BTC-USD", period="5y"):
    # Download historical price data
    print(f"Fetching data for {ticker}")
    data = yf.download(ticker, period=period, interval="1d")
    return data[['Close']]

# normalize the prices 
def prepare_data(data, look_back=60):
    #Scale data and create sequences for LSTM
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values)
    
    #here we implement a sliding window
    x, y = [], []
    for i in range(look_back, len(scaled_data)):
        x.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    
# x = input data for the lstm
# y = target values
# scaler= returned so we can inverse transform prediction back to real prices later     
    return np.array(x), np.array(y), scaler