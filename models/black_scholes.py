import torch
from torch.distributions import Normal

import streamlit as st

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
            "delta": self._spot.clone().grad,
            "rho" : self._r.clone().grad,
            "vega" : self._iv.clone().grad,
            "theta" : self._time.clone().grad,
            "epsilon" : self._d.clone().grad,
            "strike_greek" : self._strike.clone().grad
        }
    
    @property
    def gamma(self):
        spot = self._spot.clone()
        delta = torch.autograd.grad(self.price, spot, create_graph=True)[0]
        delta.backward()

        return spot.grad

    def st_visualize(self):
        st.success(str(self))
        st.divider()
        st.subheader("Calculated Gamma")
        st.metric("Gamma", str(self.gamma))
        st.divider()
        st.subheader("Calculated Greeks")
        greeks = self.greeks
        parsed_greeks = [(k, str(v)) for k, v in greeks.items()]
        st.table(parsed_greeks)
        st.divider()
