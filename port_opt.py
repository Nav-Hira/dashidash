# -*- coding: utf-8 -*-

# Portfolio Optimisation
"""

st.title('Portfolio Optimisation')

pip install yfinance
pip install fredapi
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import scipy

from scipy.optimize import minimize
from fredapi import Fred


#Define tickers

tickers = ['SPY', # S&P500 Index
           'BND', # Bond Index - Vanguard Total Bond Market Index Fund ETF
           'GLD', # SPDR Gold Trust - Largest commodity
           'QQQ', # Invesco largest stock on NASDAQ
           'VTI'] # Vanguard Total Stock Market Index Fund ETF - Largest World Stock Market

tickers2 = ['VTI', 'BLK', 'UBS', 'FNF', 'STT']

import streamlit as st

options = st.multiselect(
    'Select Ticker Symbols',
     tickers)
st.write('You selected:', options)

#Create list of close prices
adj_close_df = pd.DataFrame()

#Download close prices

start_date = '2008-01-01'
end_date = '2023-12-31'

adj_close_df = pd.DataFrame()
for ticker in tickers2:
    data = yf.download(ticker, start = start_date,end = end_date)
    adj_close_df[ticker] = data['Adj Close']

adj_close_df.head(5)

#lognormal returns
import numpy as np

log_returns = np.log(adj_close_df / adj_close_df.shift(1)).dropna() #drop nans and calculate daily returns %)

#calculate covariance

cov_matrix = log_returns.cov()*252 #(52*5 days)
print(cov_matrix)

"""# Portfolio Standard deviation"""

def standard_deviation(weights, cov_matrix):
    variance = weights.T @ cov_matrix @ weights
    return np.sqrt(variance)

# Calculate returns (using historic data, finding the average of daily returns)

def expected_return(weights, log_returns):
    return np.sum(log_returns.mean()*weights)*252

# Calculate Sharpe Ratio [(risk premium = portfolio return - risk free rate)/ Standard deviation]

def sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    return (expected_return(weights, log_returns) - risk_free_rate) / standard_deviation(weights, cov_matrix)



#risk_free_rate = .02 #default risk free rate

from fredapi import Fred
fred = Fred(api_key="6293ea460489ac4a0fd17baca6b39321")
ten_year_treasury_rate = fred.get_series_latest_release('GS10')/100

risk_free_rate = ten_year_treasury_rate.iloc[-1]
print (risk_free_rate)

#Set Initial weights

#initial_weights = np.array([1/len(tickers)]*len(tickers))
initial_weights = np.array([1/len(tickers2)]*len(tickers2))



def neg_sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    return -sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate)

constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
bounds = [(0, 0.4) for _ in range(len(tickers2))]
initial_weights = np.array([1/len(tickers2)]*len(tickers2))

optimized_results = minimize(neg_sharpe_ratio, initial_weights, args=(log_returns, cov_matrix, risk_free_rate), method='SLSQP', constraints=constraints, bounds=bounds)

optimal_weights = optimized_results.x

optimal_weights = optimized_results.x

print("Optimal Weights:")
for ticker, weight in zip(tickers2, optimal_weights):
    print(f"{ticker}: {weight:.4f}")

optimal_portfolio_return = expected_return(optimal_weights, log_returns)
optimal_portfolio_volatility = standard_deviation(optimal_weights, cov_matrix)
optimal_sharpe_ratio = sharpe_ratio(optimal_weights, log_returns, cov_matrix, risk_free_rate)

print(f"Expected Annual Return: {optimal_portfolio_return:.4f}")
print(f"Expected Volatility: {optimal_portfolio_volatility:.4f}")
print(f"Sharpe Ratio: {optimal_sharpe_ratio:.4f}")



plt.figure(figsize=(10, 6))
plt.bar(tickers2, optimal_weights)

plt.xlabel('Assets')
plt.ylabel('Optimal Weights')
plt.title('Optimal Portfolio Weights')

plt.show()
