from notanorm import SqliteDb 
from tickers import read_tickers
import yfinance as yf
# from notanorm import MysqlDb 

# accepts all the same parameters as sqlite3.connect
db = SqliteDb("data/options_data.db")

def _setup_db():
    db.query("create table options (underlying_ticker text, ticker text, contract_type text, expiration_date datetime, strike_price integer, spot_price float)")

def current_prices() -> dict:
    price_dict = {}
    for ticker in read_tickers():
        stock = yf.Ticker(ticker).history()
        current_price = stock['Close'].iloc[-1]
        price_dict[ticker] = current_price
    
    return price_dict

def add_row(underlying_ticker, ticker, contract_type, expiration_date, strike_price, price_dict=None):
    if price_dict is None:
        raise Exception("No Price Dict")

    db.insert("options", underlying_ticker=underlying_ticker, ticker=ticker, contract_type=contract_type, expiration_date=expiration_date, srtike_price=strike_price, spot_price=price_dict[underlying_ticker])