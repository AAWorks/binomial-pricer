import streamlit as st

from models.binomial_tree import BinomialTreeOption
from models.black_scholes import BlackScholesOption
from models.monte_carlo import MonteCarloOption, BasicSim

from openai_env import OptionEnv
from models.baseline_tf_dqn import BaselineModel
from models.torch_dqn import Model

from polygon import Polygon


st.set_page_config(layout="wide", page_title="Options Pricing", page_icon=":gear:")
st.title('Quantitative Options Pricing') 
st.caption('Via Black Scholes, Binomial Trees, and a DQN Model boosted w/ Monte Carlo Sampling')
st.caption("Alejandro Alonso and Roman Chenoweth")


option_filter, dqn, binomial, blk = st.tabs(["Options Filter", "Deep Q-Network Model", "PyTorch-based Black Scholes Model", "QuantLib Binomial Model"])

with option_filter:
    st.info("Query Options Data")
with dqn:
    test_env = OptionEnv()
    ex = test_env.simulate_price_data()
with binomial:
    st.info("Binomial Options Pricing")
with blk:
    st.info("Black Scholes Model for European Options Pricing")
