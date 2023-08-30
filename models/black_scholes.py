import torch
from torch.distributions import Normal

import streamlit as st
import pandas as pd

from models.abstract import Model

class BlackScholes(Model):
    def __init__(self, params):
        super().__init__(params, with_tensors=True, name="Black Scholes")

        self._cdf = Normal(0,1).cdf
        self._pdf = lambda x: Normal(0,1).log_prob(x).exp()

    @property
    def npv(self):
        d_1 = (1 / (self._iv * torch.sqrt(self._time))) * (torch.log(self._spot / self._strike) + (self._r + (torch.square(self._iv) / 2)) * self._time)
        d_2 = d_1 - self._iv * torch.sqrt(self._time)
        
        if self._option_type == "P":
            P = self._cdf(d_1) * self._spot - self._cdf(d_2) * self._strike * torch.exp(-self._r * self._time)
            return P
            
        elif self._option_type == "C":
            C = self._cdf(-d_2) * self._strike * torch.exp(-self._r * self._time) - self._cdf(-d_1) * self._spot
            return C

    @property
    def greeks(self):
        npv = self.npv
        npv.backward()
        return {
            "delta": self._spot.grad,
            "rho" : self._r.grad,
            "vega" : self._iv.grad,
            "theta" : self._time.grad,
        }

    def st_visualize(self):
        st.success(str(self))
        st.divider()
        st.subheader("Calculated Greeks")
        greeks = self.greeks
        parsed_greeks = [(k.title(), float(v)) for k, v in greeks.items()]
        data = pd.DataFrame(parsed_greeks, columns=["Greek", "Value"])
        st.dataframe(data, hide_index=True, use_container_width=True)
        st.divider()
