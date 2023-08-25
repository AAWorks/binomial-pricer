import torch
from torch.distributions import Normal

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
        self.npv.backward()
        spot = self._spot

        return {
            "delta": spot.grad,
            "rho" : self._r.grad,
            "vega" : self._iv.grad,
            "theta" : self._time.grad,
            "epsilon" : self._d.grad,
            "strike_greek" : self._strike.grad
        }
    
    @property
    def gamma(self):
        spot = self._spot
        delta = torch.autograd.grad(self.price, spot, create_graph=True)[0]
        delta.backward()

        return spot.grad
