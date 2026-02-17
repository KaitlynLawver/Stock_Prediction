import numpy as np
import pandas as pd
import datetime
import yfinance as yf
import pandas_datareader.data as web
import requests
#from datetime import datetime, timedelta
import os
import sys

import os
import sys


# ... continue with your script ...

def extract_features():

    return_period = 5
    
    START_DATE = (datetime.date.today() - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
    END_DATE = datetime.date.today().strftime("%Y-%m-%d")
    stk_tickers = ['NFLX', 'DIS', 'META']
    ccy_tickers = ['DEXJPUS', 'DEXUSUK']
    idx_tickers = ['^GSPC', '^IXIC', '^VIX']
    
    stk_data = yf.download(stk_tickers, start=START_DATE, end=END_DATE, auto_adjust=False)
    #stk_data = web.DataReader(stk_tickers, 'yahoo')
    ccy_data = web.DataReader(ccy_tickers, 'fred', start=START_DATE, end=END_DATE)
    idx_data = yf.download(idx_tickers, start=START_DATE, end=END_DATE, auto_adjust=False)

    Y = np.log(stk_data.loc[:, ('Adj Close', 'NFLX')]).diff(return_period).shift(-return_period)
    Y.name = 'NFLX_future_return'
    
    X1 = np.log(stk_data.loc[:, ('Adj Close', ('DIS', 'META'))]).diff(return_period)
    X1.columns = X1.columns.droplevel()
    X2 = np.log(ccy_data).diff(return_period)
    # Use only Adjusted Close prices from index data
    idx_adj_close = idx_data['Adj Close']
    
    # Compute log returns
    X3 = np.log(idx_adj_close).diff(return_period)
    
    X = pd.concat([X1, X2, X3], axis=1)
    
    dataset = pd.concat([Y, X], axis=1)
    
    # Align everything to stock trading dates
    dataset = dataset.loc[stk_data.index]
    
    # Forward fill macro / FX data
    dataset = dataset.ffill()
    
    # Drop remaining missing values
    dataset = dataset.dropna()
    
    dataset.index.name = 'Date'
    
    #dataset.to_csv(r"./test_data.csv")
    features = dataset.sort_index()
    features = features.reset_index(drop=True)
    features = features.iloc[:,1:]
    return features


def get_bitcoin_historical_prices(days = 60):
    
    BASE_URL = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    
    params = {
        'vs_currency': 'usd',
        'days': days,
        'interval': 'daily' # Ensure we get daily granularity
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    prices = data['prices']
    df = pd.DataFrame(prices, columns=['Timestamp', 'Close Price (USD)'])
    df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms').dt.normalize()
    df = df[['Date', 'Close Price (USD)']].set_index('Date')
    return df




