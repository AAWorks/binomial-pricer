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
                 risk_free_rate: float,
                 dividend_rate: float):

        self._option_type = option_type

        self._spot = torch.tensor(spot, requires_grad=True)
        self._strike = torch.tensor(strike, requires_grad=True)
        self._time = torch.tensor((maturity - date.today()) / timedelta(days=365), requires_grad=True)
        self._iv = torch.tensor(implied_volatility, requires_grad=True)
        self._r = torch.tensor(risk_free_rate, requires_grad=True)
        self._d = torch.tensor(dividend_rate, requires_grad=True)

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
    def dividend_rate(self):
        return self._d

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
    
    def __str__(self):
        return f"Option Price (Black Scholes Model): ${self.npv}"
