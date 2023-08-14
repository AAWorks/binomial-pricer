import time
import sqlite3
import streamlit as st
st.set_page_config(layout="wide", page_title="Options Pricing", page_icon=":gear:")

from datetime import date, timedelta

from models.binomial_tree import BinomialTreeOption
from models.black_scholes import BlackScholesOption
from models.monte_carlo import MonteCarloOption

from openai_env import OptionEnv
from models.baseline_tf_dqn import TFAModel

from polygon import Polygon
from utils.tickers import read_tickers
from utils.db_wrapper import read_rows_of_ticker

POLYGON = Polygon(yf_backup=True)

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

test_env = OptionEnv()
test_sim_data = env_ex(test_env)

DEFAULTS = get_custom_defaults()
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
    st.info("Price a Custom American Option | Work in Progress")
    with st.form("american-price"):
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        opttype = col1.selectbox("Option Type", ["C", "P"])
        s0 = float(col2.number_input("Spot Price", DEFAULTS["spot"]))
        k = float(col3.number_input("Strike Price", DEFAULTS["strike"]))
        volatility = float(col4.number_input("Volatility", DEFAULTS["volatility"]))
        d = float(col6.number_input("Dividend Rate", DEFAULTS["dividend_rate"]))
        risk_free_r = float(col5.number_input("Risk Free Rate", DEFAULTS["risk_free_rate"]))
        cola, colb = st.columns(2)
        maturity = cola.date_input("Maturity Date", DEFAULTS["maturity"])
        model = colb.selectbox("Model", ["Binomial Tree", "Monte Carlo", "Deep Q-Network", "All Models"])
        submit = st.form_submit_button("Calculate Price", use_container_width=True)
    
    if submit:
        model_name = "all models" if model == "All Models" else f"a {model} model"
        with st.spinner(f"Pricing custom option spread using {model_name}..."):
            if model == "Binomial Tree":
                opts = [BinomialTreeOption(origin="us", option_type=opttype, s=s0, 
                                    k=k, maturity=maturity, 
                                    sigma=volatility,
                                    r=risk_free_r,
                                    d=d)]
            elif model == "Monte Carlo":
                opts = [MonteCarloOption(option_type=opttype,
                                    strike=k, spot=s0, 
                                    maturity=maturity, 
                                    implied_volatility=volatility,
                                    risk_free_rate=risk_free_r,
                                    dividend_rate=d)]
            elif model == "Deep Q-Network":
                env = OptionEnv(spot=s0, strike)
                opts = [MonteCarloOption(option_type=opttype,
                                    strike=k, spot=s0, 
                                    maturity=maturity, 
                                    implied_volatility=volatility,
                                    risk_free_rate=risk_free_r,
                                    dividend_rate=d)]
            else:
                opts = [BinomialTreeOption(origin="us", option_type=opttype, 
                                           s=s0, k=k, maturity=maturity, 
                                           sigma=volatility, r=risk_free_r, 
                                           d=d),
                        MonteCarloOption(option_type=opttype, strike=k, stock=s0, 
                                    maturity=maturity, 
                                    implied_volatility=volatility,
                                    risk_free_rate=risk_free_r,
                                    dividend_rate=d),
                        MonteCarloOption(option_type=opttype,
                                    strike=k, spot=s0, 
                                    maturity=maturity, 
                                    implied_volatility=volatility,
                                    risk_free_rate=risk_free_r,
                                    dividend_rate=d)]
        for opt in opts:
            st.success(str(opt))
with eu:
    st.info("Price a Custom European Option")
    with st.form("euro-price"):
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        opttype = col1.selectbox("Option Type", ["C", "P"])
        s0 = float(col2.number_input("Spot Price", DEFAULTS["spot"]))
        k = float(col3.number_input("Strike Price", DEFAULTS["strike"]))
        volatility = float(col4.number_input("Volatility", DEFAULTS["volatility"]))
        risk_free_r = float(col5.number_input("Risk Free Rate", DEFAULTS["risk_free_rate"]))
        d = float(col6.number_input("Dividend Rate", DEFAULTS["dividend_rate"]))
        cola, colb = st.columns(2)
        maturity = cola.date_input("Maturity Date", DEFAULTS["maturity"])
        model = colb.selectbox("Model", ["Black Scholes", "Monte Carlo", "All Models"])
        submit = st.form_submit_button("Calculate Price", use_container_width=True)
    
    if submit:
        if maturity - date.today() < timedelta(days=1):
            st.error("Maturity Date Already Passed")
        else:
            model_name = "all models" if model == "All Models" else f"a {model} model"
            with st.spinner(f"Pricing custom option spread using {model_name}..."):
                if model == "Black Scholes":
                    opts = [BlackScholesOption(option_type=opttype, spot=s0, 
                                        strike=k, maturity=maturity, 
                                        implied_volatility=volatility,
                                        risk_free_rate=risk_free_r,
                                        dividend_rate=d)]
                elif model == "Monte Carlo":
                    opts = [MonteCarloOption(option_type=opttype,
                                        strike=k, spot=s0, 
                                        maturity=maturity, 
                                        implied_volatility=volatility,
                                        risk_free_rate=risk_free_r,
                                        dividend_rate=d)]
                else:
                    opts = [BlackScholesOption(option_type=opttype, spot=s0, 
                                        strike=k, maturity=maturity, 
                                        implied_volatility=volatility,
                                        risk_free_rate=risk_free_r),
                            MonteCarloOption(option_type=opttype, strike=k, stock=s0, 
                                        maturity=maturity, 
                                        implied_volatility=volatility,
                                        risk_free_rate=risk_free_r,
                                        dividend_rate=d)]

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
