import time
import streamlit as st
st.set_page_config(layout="wide", page_title="Options Pricing", page_icon=":chart:")

from models.binomial_tree import BinomialTreeOption
from models.black_scholes import BlackScholesOption
from models.monte_carlo import MonteCarloOption

from openai_env import OptionEnv
from models.baseline_tf_dqn import TFAModel
from models.torch_dqn import Model

from polygon import Polygon
from utils.tickers import read_tickers

@st.cache_data
def get_tickers():
    return read_tickers()

ALL_TICKERS = get_tickers()

st.title('Quantitative Options Pricing') 
st.caption('Via Black Scholes, Binomial Trees, Monte Carlo Sampling, and a Deep Q-Network Model | By Alejandro Alonso and Roman Chenoweth')

nasdaq, american, eu, dqn = st.tabs(["Options Pricing: NASDAQ-100", "Options Pricing: Custom American Option", "Options Pricing: Custom European Option", "About the Deep Q-Network Model"])

with nasdaq:
    st.info("Price Options from the NASDAQ-100")
    with st.form("nasdaq-price"):
        ticker = st.selectbox("Underlying Ticker", ALL_TICKERS)
        model = st.selectbox("Model", ["All Models", "Binomial Tree", "Monte Carlo", "Deep Q-Network"])
        model_name = "all models" if model == "All Models" else f"a {model} model"
        submit = st.form_submit_button("Calculate Price", use_container_width=True)

    if submit:
        with st.spinner(f"Pricing {ticker} option spread using {model_name}..."):
            time.sleep(3)
with american:
    st.info("Price a Custom American Option")
with eu:
    st.info("Price a Custom European Option")
with dqn:
    st.info("About the DQN")
    test_env = OptionEnv()
    ex = test_env.simulate_price_data()
    st.line_chart(ex)
