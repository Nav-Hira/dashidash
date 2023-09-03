# -*- coding: utf-8 -*-

# Portfolio Optimisation

#pip install -r requirements.txt



#pip install yfinance
#pip install fredapi
from fredapi import Fred
from streamlit_searchbox import st_searchbox

import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go



import scipy
import numpy as np
from scipy.optimize import minimize

st.title('Portfolio Optimisation')
st.write('By Nav S. Hira, Aug 2023')

import streamlit as st
import yfinance as yf

# Define a function to search for symbols and company names
def search_symbols_and_names(query):
    # Create a Ticker object to search for symbols
    ticker = yf.Ticker(query)
    info = ticker.info

    # Return the symbol and name if available
    symbol = info.get("symbol", "")
    name = info.get("longName", "")

    return symbol, name




# pass search function to searchbox
selected_value = st_searchbox(
    symbol,
    key=name,


# Streamlit app
st.title("Stock Symbol and Company Name Search")

# Create a search input box with autofill suggestions
search_query = st.text_input("Enter a symbol or company name:", value="", key="search_input")

# Perform the search when the user submits the query
if st.button("Search"):
    symbol, name = search_symbols_and_names(search_query)

    if symbol:
        st.success(f"Symbol: {symbol}")
        st.success(f"Company Name: {name}")
    else:
        st.error("Symbol not found. Please enter a valid symbol or company name.")


#Define tickers

tickers = ['SPY', # S&P500 Index
           'BND', # Bond Index - Vanguard Total Bond Market Index Fund ETF
           'GLD', # SPDR Gold Trust - Largest commodity
           'QQQ', # Invesco largest stock on NASDAQ
           'VTI', # Vanguard Total Stock Market Index Fund ETF - Largest World Stock Market
           'BLK', 'UBS', 'FNF', 'STT'] #Asset Management/PE/Funds

#tickers = ['VTI', 'BLK', 'UBS', 'FNF', 'STT']


ticker_options = st.multiselect(
    'Select Ticker Symbols',
     tickers)


#Create list of close prices
adj_close_df = pd.DataFrame()

#Download close prices

start_date = '2010-01-01'
end_date = '2023-12-31'

adj_close_df = pd.DataFrame()
for ticker in ticker_options:
    data = yf.download(ticker, start = start_date,end = end_date)
    adj_close_df[ticker] = data['Adj Close']

print (adj_close_df.head(5))

if st.checkbox('Show raw data (since 2010)'):
    st.subheader('Raw data')
    st.write(adj_close_df)


import plotly.express as px
df = px.data.stocks()
fig = px.line(adj_close_df, x=adj_close_df.index, y=adj_close_df.columns,
              #hover_data={"adj_close_df.index": "|%B %d, %Y"},
              title='custom tick labels')
fig.update_xaxes(
    dtick="M1",
    tickmode='auto',
    tickformat="%b\n%Y")
fig.show()

st.plotly_chart(fig, use_container_width=True)





#lognormal returns
log_returns = np.log(adj_close_df / adj_close_df.shift(1)).dropna() #drop nans and calculate daily returns %)

#calculate covariance

cov_matrix = log_returns.cov()*252 #(52*5 days)
print(cov_matrix)





def standard_deviation(weights, cov_matrix):
    variance = weights.T @ cov_matrix @ weights
    return np.sqrt(variance)

# Calculate returns (using historic data, finding the average of daily returns)

def expected_return(weights, log_returns):
    return np.sum(log_returns.mean()*weights)*252

# Calculate Sharpe Ratio [(risk premium = portfolio return - risk free rate)/ Standard deviation]

def sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    return (expected_return(weights, log_returns) - risk_free_rate) / standard_deviation(weights, cov_matrix)

st.title('Portfolio Optimisation')

#risk_free_rate = .02 #default risk free rate
st.write('For risk-free rate, either select 10yr Treasure Rate or enter risk-free rate manually (default is 2%)')
if st.checkbox('10 yr Treasury Rate % '):
    fred = Fred(api_key="6293ea460489ac4a0fd17baca6b39321")
    ten_year_treasury_rate = fred.get_series_latest_release('GS10')/100
    risk_free_rate = ten_year_treasury_rate.iloc[-1]
    st.write('Risk-Free Rate = ', risk_free_rate*100, '%')
else: 
    risk_free_rate = st.number_input('Enter Risk-Free Rate value', value =0.02)
    st.write('Risk-Free Rate = ', risk_free_rate*100, '%')



#Set Initial weights

#initial_weights = np.array([1/len(ticker_options)]*len(ticker_options))
initial_weights = np.array([1/len(ticker_options)]*len(ticker_options))



def neg_sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    return -sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate)

constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
bounds = [(0, 0.4) for _ in range(len(ticker_options))]
initial_weights = np.array([1/len(ticker_options)]*len(ticker_options))

optimized_results = minimize(neg_sharpe_ratio, initial_weights, args=(log_returns, cov_matrix, risk_free_rate), method='SLSQP', constraints=constraints, bounds=bounds)

optimal_weights = optimized_results.x
print("Optimal Weights:")

for ticker, weight in zip(ticker_options, optimal_weights):
    print(f"{ticker}: {weight:.4f}")

optimal_portfolio_return = expected_return(optimal_weights, log_returns)
optimal_portfolio_volatility = standard_deviation(optimal_weights, cov_matrix)
optimal_sharpe_ratio = sharpe_ratio(optimal_weights, log_returns, cov_matrix, risk_free_rate)

print(f"Expected Annual Return: {optimal_portfolio_return:.4f}")
print(f"Expected Volatility: {optimal_portfolio_volatility:.4f}")
print(f"Sharpe Ratio: {optimal_sharpe_ratio:.4f}")





colors = ['lightslategray',] * 5
colors[1] = 'crimson'

fig = go.Figure(data=[go.Bar(
    x=ticker_options,
    y=optimal_weights,
    #marker_color=colors # marker color can be a single color value or an iterable
)])
fig.update_layout(title_text='Optimal Portfolio Weights')

st.plotly_chart(fig, use_container_width=True)


