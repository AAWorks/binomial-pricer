import streamlit as st

st.set_page_config(layout="wide", page_title="SOP Bot", page_icon=":chart_with_upwards_trend:")
st.title('Options Pricing: From Binomial to Black Shouls :chart_with_upwards_trend:')
st.caption("By Alejandro Alonso and Roman Chenoweth")


region = st.selectbox("Select Market Region", ("United States of America", "European Union"))

if region == "United States of America":
    with st.form("us"):
        st.write("hi")

elif region == "European Union":
    with st.form("eu"):
        st.write("hi")

else:
    st.warning("No Market Region Selected")

