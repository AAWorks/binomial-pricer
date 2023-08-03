from notanorm import SqliteDb 
import yfinance as yf
# from notanorm import MysqlDb 

# accepts all the same parameters as sqlite3.connect
db = SqliteDb("options_data")

db.query("create table options (underlying_ticker text, ticker text, contract_type text, expiration_date datetime, strike_price integer, spot_price float)")


current_ticker = ""
current_price = 0.00

def add_row (self, underlying_ticker, ticker, contract_type, expiration_date, strike_price):
    if underlying_ticker != current_ticker:
        current_ticker = underlying_ticker
        stock = yf.Ticker(ticker)
        data = stock.history()
        current_price = data['Close'].iloc[-1]
    db.insert("options", underlying_ticker=underlying_ticker, ticker=ticker, contract_type=contract_type, expiration_date=expiration_date, srtike_price=strike_price, spot_price=current_price)
    


