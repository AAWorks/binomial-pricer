import time
import sqlite3
import streamlit as st
st.set_page_config(layout="wide", page_title="Options Pricing", page_icon=":gear:")

from datetime import date, timedelta

from models.abstract import Option, inputs
from models.binomial_tree import EUBinomialTreeOption, USBinomialTreeOption
from models.black_scholes import BlackScholesOption
from models.monte_carlo import MonteCarloOption

from openai_env import OptionEnv
from models.baseline_tfa_dqn import TFAModel

from polygon import Polygon
from utils.tickers import read_tickers
from utils.db_wrapper import read_rows_of_ticker

POLYGON = Polygon(yf_backup=True)
STANDARD_MODELS = {
    "Black Scholes_eu": [BlackScholesOption],
    "Monte Carlo_eu": [MonteCarloOption],
    "Monte Carlo_us": [MonteCarloOption],
    "Binomial Tree_eu": [EUBinomialTreeOption],
    "Binomial Tree_us": [USBinomialTreeOption],
    "All Models_us": [USBinomialTreeOption, MonteCarloOption],
    "All Models_eu": [BlackScholesOption, EUBinomialTreeOption, MonteCarloOption]
}

@st.cache_data
def get_eu_models():
    return [model[:-3] for model in STANDARD_MODELS if "eu" in model]

@st.cache_data
def get_us_models(_dqn = False):
    models = [model[:-3] for model in STANDARD_MODELS if "us" in model]
    return models if not _dqn else models[:-1] + ["Deep Q-Network"] + [models[-1]]

@st.cache_data
def get_tickers():
    return read_tickers()

@st.cache_data
def env_ex(_test):
    return _test.simulate_price_data()

@st.cache_data(ttl=60)
def check_nasdaq_status():
    return POLYGON.exchange_status("nasdaq")

@st.cache_data
def get_custom_defaults():
    defaults = {}
    
    defaults["maturity"] = date.today() + timedelta(days=365)
    defaults["risk_free_rate"] = POLYGON.risk_free_rate
    defaults["spot"] = 100.0
    defaults["strike"] = 100.0
    defaults["volatility"] = 0.20
    defaults["dividend_rate"] = 0.01

    return defaults

DEFAULTS = get_custom_defaults()
ALL_TICKERS = get_tickers()
NASDAQ_STATUS = check_nasdaq_status()

test_env = OptionEnv(spot=DEFAULTS["spot"], strike=DEFAULTS["strike"], r=DEFAULTS["risk_free_rate"], sigma=DEFAULTS["volatility"], maturity=DEFAULTS["maturity"])
test_sim_data = env_ex(test_env)

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

    if True: # NASDAQ_STATUS == "open":
        with st.form("nasdaq-price"):
            cola, colb = st.columns(2)
            ticker = cola.selectbox("Underlying Ticker", ALL_TICKERS)
            model = colb.selectbox("Model", get_us_models())
            submit = st.form_submit_button("Calculate Price", use_container_width=True)

        if submit:
            st.write(POLYGON.get_eod_data_of_ticker(ticker))
            model_name = "all models" if model == "All Models" else f"a {model} model"
            with st.spinner(f"Pricing {ticker} option spread using {model_name}..."):
                time.sleep(3)

with american:
    st.info("Price a Custom American Option")
    with st.form("american-price"):
        opttype, s0, k, volatility, risk_free_r, d, maturity, model, submit = inputs(get_us_models(_dqn=True), DEFAULTS)
    if submit:
        model_name = "all models" if model == "All Models" else f"a {model} model"
        with st.spinner(f"Pricing custom option spread using {model_name}..."):
            if model == "Deep Q-Network":
                st.error("Not Supported Yet")
                opts = []
            else: 
                opts = [opt(
                    option_type=opttype,
                    strike=k, spot=s0, 
                    maturity=maturity, 
                    implied_volatility=volatility,
                    risk_free_rate=risk_free_r,
                    dividend_rate=d)
                    for opt in STANDARD_MODELS[model + "_us"]]

        for opt in opts:
            st.success(str(opt))
with eu:
    st.info("Price a Custom European Option")
    with st.form("euro-price"):
        opttype, s0, k, volatility, risk_free_r, d, maturity, model, submit = inputs(get_eu_models(), DEFAULTS)
    if submit:
        if maturity - date.today() < timedelta(days=1):
            st.error("Maturity Date Already Passed")
        else:
            model_name = "all models" if model == "All Models" else f"a {model} model"
            with st.spinner(f"Pricing custom option spread using {model_name}..."):
                opts = [opt(
                    option_type=opttype,
                    strike=k, spot=s0, 
                    maturity=maturity, 
                    implied_volatility=volatility,
                    risk_free_rate=risk_free_r,
                    dividend_rate=d)
                    for opt in STANDARD_MODELS[model + "_eu"]]

        for opt in opts:
            st.success(str(opt))

with dqn:
    st.info("Deep Q-Network Breakdown | Work in progress")
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
