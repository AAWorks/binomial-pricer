import torch
from torch.distributions import Normal

class BlackScholesOption:
    def __init__(self, 
                 right="C",
                 K=torch.tensor(100.0, requires_grad=True), 
                 S=torch.tensor(100.0, requires_grad=True), 
                 T = torch.tensor(1.0, requires_grad=True), 
                 sigma = torch.tensor(0.20, requires_grad=True), 
                 r = torch.tensor(0.00, requires_grad=True)):
        self._std_norm_cdf = Normal(0, 1).cdf
        self._std_norm_pdf = lambda x: torch.exp(Normal(0, 1).log_prob(x))
        self._option_type = right
        # Present Value of Exercise Price
        self._k = K
        # Current Stock Price
        self._s = S
        # Time
        self._t = T
        # Volatility
        self._sigma = sigma
        # Risk-Free rate
        self._r = r

    @property
    def option_type(self): return self._option_type

    @property
    def strike_price(self): return self._k

    @property
    def spot_price(self): return self._s

    @property
    def time(self): return self._t

    @property
    def volatility(self): return self._sigma

    @property
    def risk_free_rate(self): return self._r
    
    def price(self):
        d_1 = (1 / (self._sigma * torch.sqrt(self._t))) * (torch.log(self._s / self._k) + (self._r + (torch.square(self._s) / 2)) * self._t)
        d_2 = d_1 - self._sigma * torch.sqrt(self._t)
        
        if self._option_type == "C":
            C = self._std_norm_cdf(d_1) * S - self._std_norm_cdf(d_2) * self._k * torch.exp(-self._r * self._t)
            return C
            
        elif self._option_type == "P":
            P = self._std_norm_cdf(-d_2) * self._k * torch.exp(-self._r * self._t) - self._std_norm_cdf(-d_1) * self._s
            return P
    
    def compute_greeks(self, price: torch.tensor):
        price.backward()
        return {
            "Delta": self._s.grad,
            "Vega": self._sigma.grad,
            "Theta": self._t.grad,
            "Rho": self._r.grad
        }
    
    def exercise_probabilities(self):
        # probability that option will be exercised
        d_1 = (1 / (self._sigma * torch.sqrt(self._t))) * (torch.log(self._s / self._k) + (self._r + (torch.square(self._sigma) / 2)) * self._t)
        # probability that option will be exercised
        d_2 = d_1 - self._sigma * torch.sqrt(self._t)

    def compute_gamma(self):
        delta = self._std_norm_cdf(self._d_1)
        vega = self._s * self._std_norm_pdf(self._d_1) * torch.sqrt(self._t)
        theta = ((self._s * self._std_norm_pdf(self._d_1) * self._sigma) / (2 * torch.sqrt(self._t))) + self._r * self._k * torch.exp(-self._r * self._t) * self._std_norm_cdf(self._d_2)
        rho = self._k * self._t * torch.exp(-self._r * self._t) * self._std_norm_cdf(self._d_2)

        S = torch.tensor(100.0, requires_grad=True)
        price = self._bs_price(self._right, self._k, self._s, self._t, self._sigma, self._r)

        delta = torch.autograd.grad(price, self._s, create_graph=True)[0]
        delta.backward()

        # And the direct Black-Scholes calculation
        gamma = self._std_norm_pdf(self._d_1) / (self._s * self._sigma * torch.sqrt(self._t))

        return gamma, self._s.grad