import time
import sqlite3
import streamlit as st
st.set_page_config(layout="wide", page_title="Options Pricing", page_icon=":gear:")

from streamlit_extras.mandatory_date_range import date_range_picker
from datetime import date

from models.binomial_tree import BinomialTreeOption
from models.black_scholes import BlackScholesOption
from models.monte_carlo import MonteCarloOption

from openai_env import OptionEnv
from models.baseline_tf_dqn import TFAModel

from polygon import Polygon
from utils.tickers import read_tickers
from utils.db_wrapper import read_rows_of_ticker

ALL_MODELS = {
    "Binomial Tree": BinomialTreeOption,
    "Black Scholes": BlackScholesOption,
    "Monte Carlo": MonteCarloOption,
    "Deep Q-Network": TFAModel
}
POLYGON = Polygon(yf_backup=True)

@st.cache_data
def get_tickers():
    return read_tickers()

test_env = OptionEnv()
@st.cache_data
def env_ex(_test):
    return _test.simulate_price_data()

@st.cache_data(ttl=60)
def check_nasdaq_status():
    return POLYGON.exchange_status("nasdaq")

test_sim_data = env_ex(test_env)

ALL_TICKERS = get_tickers()
NASDAQ_STATUS = check_nasdaq_status()

st.title('Quantitative Options Pricing') 
st.caption('Via Black Scholes, Binomial Trees, Monte Carlo Sampling, and a Deep Q-Network Model | By Alejandro Alonso and Roman Chenoweth')

nasdaq, american, eu, dqn = st.tabs(["Options Pricing: NASDAQ-100", "Options Pricing: Custom American Option", "Options Pricing: Custom European Option", "About the Deep Q-Network Model"])

with nasdaq:
    st.info("Pricing Options from the NASDAQ-100 | Work In Progress")
    title_status = NASDAQ_STATUS.title()
    message_type = {"open": st.success, "extended-hours": st.error, "closed": st.error} 
    message = {
        "open": f"Market Status: {title_status}", 
        "extended-hours": f"Feature Not Available During NASDAQ After-Hours", 
        "closed": f"Feature Not Supported While NASDAQ is Closed"
    }
    
    message_type[NASDAQ_STATUS](message[NASDAQ_STATUS])
    with st.expander("Note on Data Source"):
        st.caption("This project was originally designed to use real-time data from Polygon.io. All the tooling is present, however, due to financial constraints, we opted to terminate our subscription to Polygon.io after a month. So while this pricer can be easily reconfigured to use Polygon.io's data, we currently use EOD data from Yahoo Finance.")

    if False: #NASDAQ_STATUS == "open":
        with st.form("nasdaq-price"):
            ticker = st.selectbox("Underlying Ticker", ALL_TICKERS)
            model = st.selectbox("Model", ["All Models", "Binomial Tree", "Monte Carlo", "Deep Q-Network"])
            submit = st.form_submit_button("Calculate Price", use_container_width=True)

        if submit:
            st.info(f"{ticker} Options Data - Pulled {date.today()}")
            st.dataframe(read_rows_of_ticker(sqlite3.Connection("data/options_data.db"), ticker))
            model_name = "all models" if model == "All Models" else f"a {model} model"
            with st.spinner(f"Pricing {ticker} option spread using {model_name}..."):
                time.sleep(3)

with american:
    st.info("Price a Custom American Option")
    with st.form("american-price"):
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        opttype = col1.selectbox("Option Type", ["C", "P"])
        s0 = float(col2.number_input("Spot Price"))
        k = float(col3.number_input("Strike Price"))
        volatility = float(col4.number_input("Volatility"))
        d = float(col6.number_input("Dividend Rate"))
        risk_free_r = float(col5.number_input("Risk Free Rate"))
        cola, colb = st.columns(2)
        with cola:
            date1, date2 = date_range_picker("Purchase-Maturity Datespan")
        model = colb.selectbox("Model", ["All Models", "Binomial Tree", "Deep Q-Network"])
        submit = st.form_submit_button("Calculate Price", use_container_width=True)
    
    if submit:
        model_name = "all models" if model == "All Models" else f"a {model} model"
        with st.spinner(f"Pricing custom option spread using {model_name}..."):
            time.sleep(3)
with eu:
    st.info("Price a Custom European Option")
    with st.form("euro-price"):
        col1, col2, col3, col4, col5 = st.columns(5)
        opttype = col1.selectbox("Option Type", ["C", "P"])
        s0 = float(col2.number_input("Spot Price"))
        k = float(col3.number_input("Strike Price"))
        volatility = float(col4.number_input("Volatility"))
        risk_free_r = float(col5.number_input("Risk Free Rate"))
        cola, colb = st.columns(2)
        with cola:
            date1, date2 = date_range_picker("Purchase-Maturity Datespan")
        model = colb.selectbox("Model", ["All Models", "Black Scholes", "Monte Carlo"])
        submit = st.form_submit_button("Calculate Price", use_container_width=True)
    
    if submit:
        model_name = "all models" if model == "All Models" else f"a {model} model"
        with st.spinner(f"Pricing custom option spread using {model_name}..."):
            time.sleep(3)
with dqn:
    st.info("About the DQN")
    st.line_chart(test_sim_data)

# with pull:
#     with st.form("tmp_pull"):
#         tmpkey = st.text_input("Polygon.io Key")
#         pwd = st.text_input("Admin Passphrase")
#         submitted = st.form_submit_button("Pull", use_container_width=True)
    
#     if submitted:
#         key = tmpkey if tmpkey.strip() != "" else None

#         if pwd == "123457":
#             with st.spinner("Pulling..."):
#                 Polygon(key=key).store_eod_data()
