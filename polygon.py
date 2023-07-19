import requests

class Polygon:
    _headers: dict
    _base_url: str

    def __init__(self):
        with open('keys/polygon.txt', 'r') as keyfile:
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

    def query(self, query: str):
        pass
    
    def option(self, ticker, date, position):
        pass

    def options(self, ticker, start_date, end_date=None, position=None):
        pass
