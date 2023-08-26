import requests, time
import random
import numpy as np
import pandas as pd
import yfinance as yf
from yahoo_fin import options
from datetime import datetime

from utils.tickers import read_tickers
from utils.db_wrapper import clear_table, add_rows, yf_current_prices

class Polygon:
    _headers: dict
    _base_url: str

    def __init__(self, key=None, yf_backup=False):
        if key is None:
            with open('data/polygon.txt', 'r') as keyfile:
                key = keyfile.readline().strip()

        self._headers = {
            'Authorization': f'Bearer {key}'
        }

        self._base_url = 'https://api.polygon.io/'
        self._yf_backup = yf_backup
    
    @property
    def risk_free_rate(self):
        def deannualize(annual_rate, periods=(365//4)):
            return (1 + annual_rate) ** (1/periods) - 1

        def get_risk_free_rate():
            annualized = yf.download("^IRX")["Adj Close"]
            daily = annualized.apply(deannualize)

            return pd.DataFrame({"annualized": annualized, "trimonthly": daily})    

        rates = get_risk_free_rate()
        return float(rates["trimonthly"].iloc[-1])
    
    @property
    def nasdaq_tickers(self):
        return read_tickers()
    
    @property
    def last_ticker_prices(self):
        return yf_current_prices(self.nasdaq_tickers)

    @property
    def base_url(self):
        return self._base_url

    def _get_req_url(self, extension: str = ""):
        return self._base_url + extension

    def _query(self, query: str):
        full_query = self._get_req_url(query)
        return requests.request("GET", url=full_query, headers=self._headers)
    
    def _options_query(self, query: str):
        time.sleep(12)
        return self._query(f"v3/reference/options/contracts?{query}")

    def _polygon_options(self, ticker, position="", expired=""):
        if position:
            position = f"&contract_type={position}"
        
        if expired != "":
            expired = f"&expired={str(expired).lower()}"
        
        query = f"underlying_ticker={ticker}{position}{expired}&limit=1000"
        return self._options_query(query)
    
    def _poly_ticker_contracts(self, ticker, expiration):
        json_data = self._polygon_options(ticker).json()
        ticker_data = pd.DataFrame(json_data["results"])
        prices = self._get_eod_stock_prices(self.nasdaq_tickers)
        ticker_data["mark"] = prices[ticker]
        ticker_data["price"] = prices[ticker]
    
    def _yf_ticker_contracts(self, ticker, expiration):
        random.seed(31337)
        np.random.seed(31337)

        expiration = expiration.strftime("%m/%d/%Y")
        try:
            chain = options.get_options_chain(ticker, expiration)
        except ValueError:
            return None

        calls = pd.DataFrame(chain["calls"])
        puts = pd.DataFrame(chain["puts"])
        calls["Type"] = "C"
        puts["Type"] = "P"

        ticker_data = pd.concat([calls, puts]).sort_index(kind='merge')
        ticker_data.reset_index(inplace=True, drop=True)
        
        ticker_data["Contract ID"] = np.random.randint(low=100, high=999, size=len(ticker_data))
        ticker_data["Bid"] = ticker_data["Bid"].apply(pd.to_numeric, errors='coerce')
        ticker_data["Mark"] = ticker_data[["Bid", "Ask"]].mean(axis=1)
        
        #ticker_data["Dividend Yield"] = ticker_data["x"] - ticker_data["Mark"]

        return ticker_data

    def _get_eod_options_data(self, tickers):
        all_ticker_options_data = []
        for ticker in tickers:
            json_data = self._polygon_options(ticker).json()
            ticker_data = pd.DataFrame(json_data["results"])
            all_ticker_options_data.append(ticker_data)
        
        return pd.concat(all_ticker_options_data).sort_index(kind='merge')
    
    def _get_eod_stock_prices(self, tickers):
        ticker_prices = {}
        for ticker in tickers:
            query = f"v2/aggs/ticker/{ticker}/prev?adjusted=true"
            previous_day_details = self._query(query).json()
            price_results = previous_day_details["results"][0]
            
            if "vw" in price_results:
                ticker_prices[ticker] = price_results["vw"] #volume weighted avg
            else:
                ticker_prices[ticker] = price_results["c"] # close
        
        return ticker_prices

    def exchange_status(self, exchange): # i.e. nasdaq -> open, closed, after-hours
        query = "v1/marketstatus/now?"
        market_details = self._query(query).json()
        return market_details["exchanges"][exchange] 

    def expiration_dates(self, ticker):
        return options.get_expiration_dates(ticker.lower())[1:]

    def store_all_eod_data(self):
        clear_table()

        data_methods = {True: self._yf_get_eod_options_data, False: self._get_eod_options_data}
        tickers = read_tickers()

        eod_data = data_methods[self._yf_backup](tickers)

        if self._yf_backup:
            current_stock_prices = yf_current_prices(tickers)
        else:
            current_stock_prices = self._get_eod_stock_prices(tickers)

        add_rows(eod_data, current_stock_prices)
    
    def get_ticker_contracts_given_exp(self, ticker, expiration: str):
        scrape_methods = {True: self._yf_ticker_contracts, False: self._poly_ticker_contracts}

        eod_data = scrape_methods[self._yf_backup](ticker, expiration)

        return eod_data
