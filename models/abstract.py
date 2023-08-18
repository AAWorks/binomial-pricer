# Abstract Option Class
import streamlit as st

from abc import ABC


class Option(ABC):
    def __init__(self):
        pass

def inputs(region_models, defaults):
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    opttype = col1.selectbox("Option Type", ["C", "P"])
    s0 = float(col2.number_input("Spot Price", defaults["spot"]))
    k = float(col3.number_input("Strike Price", defaults["strike"]))
    iv = float(col4.number_input("Volatility", defaults["volatility"]))
    d = float(col6.number_input("Dividend Rate", defaults["dividend_rate"]))
    r = float(col5.number_input("Risk Free Rate", defaults["risk_free_rate"]))
    cola, colb = st.columns(2)
    maturity = cola.date_input("Maturity Date", defaults["maturity"])
    model = colb.selectbox("Model", region_models)
    submit = st.form_submit_button("Calculate Price", use_container_width=True)

    return opttype, s0, k, iv, d, r, maturity, model, submit
