import torch
from torch.distributions import Normal

from datetime import date, timedelta

class BlackScholesOption:
    def __init__(self, 
                 option_type: str, 
                 spot: float, 
                 strike: float, 
                 maturity: date, 
                 implied_volatility: float,
                 risk_free_rate: float):

        self._option_type = option_type

        self._spot = torch.Tensor(spot)
        self._spot.requires_grad(True)

        self._strike = torch.Tensor(strike)
        self._strike.requires_grad(True)

        self._time = torch.Tensor((maturity - date.today()) / timedelta(days=365))
        self._time.requires_grad(True)

        self._iv = torch.Tensor(implied_volatility)
        self._iv.requires_grad(True)

        self._r = torch.Tensor(risk_free_rate)
        self._r.requires_grad(True)

        self._cdf = Normal(0,1).cdf
        self._pdf = lambda x: Normal(0,1).log_prob(x).exp()

    @property
    def option_type(self): 
        return self._option_type

    @property
    def strike_price(self): 
        return self._strike

    @property
    def spot_price(self): 
        return self._spot

    @property
    def time(self): 
        return self._time

    @property
    def implied_volatility(self): 
        return self._iv
    
    @property
    def risk_free_rate(self):
        return self._r

    @property
    def price(self):
        d_1 = (1 / (self._iv * torch.sqrt(self._time))) * (torch.log(self._spot / self._strike) + (self._r + (torch.square(self._spot) / 2)) * self._time)
        d_2 = d_1 - self._iv * torch.sqrt(self._time)
        
        if self._option_type == "C":
            C = self._cdf(d_1) * self._spot - self._cdf(d_2) * self._strike * torch.exp(-self._r * self._time)
            return C
            
        elif self._option_type == "P":
            P = self._cdf(-d_2) * self._strike * torch.exp(-self._r * self._time) - self._cdf(-d_1) * self._spot
            return P

    @property
    def standard_greeks(self):
        self.price.backward()
        spot = self._spot

        return {
            "delta": spot.grad,
            "rho" : self._r.grad,
            "vega" : self._iv.grad,
            "theta" : self._time.grad,
            "strike_greek" : self._strike.grad
        }
    
    @property
    def gamma(self):
        spot = self._spot
        delta = torch.autograd.grad(self.price, spot, create_graph=True)[0]
        delta.backward()

        return spot.grad
