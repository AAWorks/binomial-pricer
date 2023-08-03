import streamlit as st

from models.binomial_tree import BinomialTreeOption
from models.black_scholes import BlackScholesOption
from models.monte_carlo import MonteCarloOption

from openai_env import OptionEnv
from models.baseline_tf_dqn import TFAModel
from models.torch_dqn import Model

from polygon import Polygon


st.set_page_config(layout="wide", page_title="Options Pricing", page_icon=":gear:")
st.title('Quantitative Options Pricing') 
st.caption('Via Black Scholes, Binomial Trees, Monte Carlo Sampling, and a Deep Q-Network Model | By Alejandro Alonso and Roman Chenoweth')

nasdaq, american, eu, dqn = st.tabs(["Options Pricing: NASDAQ-100", "Options Pricing: Custom American Option", "Options Pricing: Custom European Option", "About the Deep Q-Network Model"])

with nasdaq:
    st.info("Price Options from the NASDAQ-100")
with american:
    st.info("Price a Custom American Option")
with eu:
    st.info("Price a Custom European Option")
with dqn:
    st.info("About the DQN")
    test_env = OptionEnv()
    ex = test_env.simulate_price_data()
    st.line_chart(ex)
