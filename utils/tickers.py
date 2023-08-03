import pandas as pd

def write_tickers():
    payload = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')
    nasdaq_100 = payload[4]

    with open("tickers.txt", "w") as tickers:
        tickers.writelines(", ".join(nasdaq_100["Ticker"]))

def read_tickers():
    with open("tickers.txt", "r") as tickers:
        return tickers.readline().split(", ")
