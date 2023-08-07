from notanorm import SqliteDb 
from utils.tickers import read_tickers

import yfinance as yf
import pandas as pd
from datetime import datetime
# from notanorm import MysqlDb 

# accepts all the same parameters as sqlite3.connect
db = SqliteDb("data/options_data.db")

def _setup_db():
    db.query("create table options (underlying_ticker text, ticker text, contract_type text, expiration_date datetime, strike_price integer, spot_price float)")

def yf_current_prices(tickers) -> dict:
    price_dict = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker).history()
        current_price = stock['Close'].iloc[-1]
        price_dict[ticker] = current_price
    
    return price_dict

def add_row(underlying_ticker, ticker, contract_type, expiration_date, strike_price, price_dict=None):
    if price_dict is None:
        raise Exception("No Price Dict")
    date = datetime.strptime(expiration_date, '%Y-%m-%d')
    db.insert("options", underlying_ticker=underlying_ticker, ticker=ticker, contract_type=contract_type, expiration_date=date, strike_price=strike_price, spot_price=price_dict[underlying_ticker])
    
    
def add_rows(dataframe : pd.DataFrame, price_dict=None):
    for _, row in dataframe.iterrows():
        add_row("options", row["underlying_ticker"], row["ticker"], row["contract_type"], row["expiration_date"], row["strike_price"], price_dict)

def read_rows_of_ticker(con, ticker): 
    return pd.read_sql_query(f"SELECT * FROM options WHERE ticker = {ticker}", con)

def read_rows(con): 
    return pd.read_sql_query(f"SELECT * FROM options", con)

def clear_table():
    db.delete_all("options")
    
