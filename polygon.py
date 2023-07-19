import requests

class Polygon:
    _headers: dict

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

    def get_req_url(self, extension = ""):
        return self._base_url + extension