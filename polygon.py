import requests
import pandas as pd

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
    def headers(self):
        return self._headers

    def _get_req_url(self, extension: str = ""):
        return self._base_url + extension

    def _query(self, query: str):
        pass
    
    def _option(self, ticker, date, position):
        pass

    def _options(self, ticker, start_date, end_date=None, position=None):
        pass

    def _get_eod_data(self):
        pass

    def store_eod_data(self, db):
        db.delete_all("options")
        
        for contract in self._get_eod_data():
            
