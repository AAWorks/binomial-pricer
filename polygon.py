import requests
import pandas as pd

from utils.tickers import read_tickers
from utils.db_wrapper import clear_table, add_rows, current_prices

class Polygon:
    _headers: dict
    _base_url: str

    def __init__(self):
        with open('data/polygon.txt', 'r') as keyfile:
            key = keyfile.readline().strip()

        self._headers = {
            'Authorization': f'Bearer {key}'
        }

        self._base_url = 'https://api.polygon.io/v3/reference/options/contracts?'
    
    @property
    def base_url(self):
        return self._base_url

    def _get_req_url(self, extension: str = ""):
        return self._base_url + extension

    def _query(self, query: str):
        full_query = self._get_req_url(query)
        return requests.request(full_query, headers=self._headers)

    def _options(self, ticker, position="", expired=""):
        if position:
            position = f"&contract_type={position}"
        
        if expired != "":
            expired = f"&expired={str(expired).lower()}"
        
        query = f"underlying_ticker={ticker}{position}{expired}&limit=1000"
        return self._query(query)

    def _get_eod_data(self, tickers):
        all_ticker_options_data = []
        for ticker in tickers:
            json_data = self._options(ticker)
            ticker_data = pd.read_json(json_data)
            all_ticker_options_data.append(ticker_data)
        
        return pd.concat(all_ticker_options_data).sort_index(kind='merge')

    def store_eod_data(self):
        clear_table()
        eod_data = pd.read_json(self._get_eod_data(read_tickers()))
        add_rows(eod_data, current_prices())
