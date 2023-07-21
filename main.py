import streamlit as st

from models.binomial_tree import BinomialTreeOption
from models.black_scholes import BlackScholesOption
#from models.monte_carlo import MonteCarloOption

from polygon import Polygon

st.set_page_config(layout="wide", page_title="SOP Bot", page_icon=":chart_with_upwards_trend:")
st.title('Options Pricing: Binomial and Black Scholes with Monte Carlo Sampling')
st.caption("By Alejandro Alonso and Roman Chenoweth")


binomial, blk = st.tabs(["Binomial Model", "Black Scholes"])

with binomial:
    st.info("Binomial Options Pricing")
with blk:
    st.info("Black Scholes Model for European Options Pricing")

