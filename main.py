import streamlit as st

st.set_page_config(layout="wide", page_title="SOP Bot", page_icon=":chart_with_upwards_trend:")
st.title('Options Pricing: Binomial and Black Sholes with Monte Carlo Sampling')
st.caption("By Alejandro Alonso and Roman Chenoweth")


binomial, blk = st.tabs(["Binomial Model", "Black Sholes"])

with binomial:
    st.info("Binomial Options Pricing")
with blk:
    st.info("Black Sholes Model for European Options Pricing")

