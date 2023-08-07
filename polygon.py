import requests
import pandas as pd

from utils.tickers import read_tickers
from utils.db_wrapper import clear_table, add_rows, yf_current_prices

class Polygon:
    _headers: dict
    _base_url: str

    def __init__(self):
        with open('data/polygon.txt', 'r') as keyfile:
            key = keyfile.readline().strip()

        self._headers = {
            'Authorization': f'Bearer {key}'
        }

        self._base_url = 'https://api.polygon.io/'
    
    @property
    def base_url(self):
        return self._base_url

    def _get_req_url(self, extension: str = ""):
        return self._base_url + extension

    def _query(self, query: str):
        full_query = self._get_req_url(query)
        return requests.request("GET", url=full_query, headers=self._headers)
    
    def _options_query(self, query: str):
        return self._query(f"v3/reference/options/contracts?{query}")

    def _options(self, ticker, position="", expired=""):
        if position:
            position = f"&contract_type={position}"
        
        if expired != "":
            expired = f"&expired={str(expired).lower()}"
        
        query = f"underlying_ticker={ticker}{position}{expired}&limit=1000"
        return self._options_query(query)

    def _get_eod_options_data(self, tickers):
        all_ticker_options_data = []
        for ticker in tickers:
            json_data = self._options(ticker).json()
            ticker_data = pd.read_json(json_data)
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

    def store_eod_data(self, use_polygon_for_stock_prices = False):
        clear_table()
        eod_data = pd.read_json(self._get_eod_options_data(read_tickers()))
        tickers = read_tickers()

        if not use_polygon_for_stock_prices:
            current_stock_prices = yf_current_prices(tickers)
        else:
            current_stock_prices = self._get_eod_stock_prices(tickers)

        add_rows(eod_data, current_stock_prices)
