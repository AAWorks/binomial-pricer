import torch
from datetime import date, timedelta

class MonteCarloOption:
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
    def npv(self):
        torch.manual_seed(42)
        scenarios = 1000000
        dW = self._iv * self._time ** 0.5 * torch.randn(size=(scenarios,))
        r = torch.exp((self._r - self._d - self._iv * self._iv / 2) * self._time + dW)

        if self._option_type == "P":
            payoff = torch.max(self._strike - self._spot*r, torch.zeros(size=(scenarios,)))
        else:
            payoff = torch.max(self._spot*r - self._strike, torch.zeros(size=(scenarios,)))

        return torch.mean(payoff) * torch.exp(-self._r*self._time)
    
    @property
    def greeks(self):
        ov = self.npv
        ov.backward()
        return {
            "delta": self._spot.grad,
            "rho": self._r.grad,
            "vega": self._iv.grad,
            "theta": self._time.grad,
            "epsilon" : self._d.grad,
            "strike_greek": self._strike.grad
        }

    def __str__(self):
        return f"Option Price (Monte Carlo Pricing): ${self.npv}"


def _simulate_ep(policy, env, time_step, base_return=0.0):
    while not time_step.is_last():
        action_step = policy.action(time_step)
        time_step = env.step(action_step.action)
        base_return += time_step.reward
    
    return base_return

def _simulate_eps(policy, env, eps):
    step = env.reset()
    return sum([_simulate_ep(policy, env, step) for _ in range(eps)])

def dqn_sim(policy, env, eps=10, base_return=0.0):
    avg_return = (base_return + _simulate_eps(policy, env, eps)) / eps
    return avg_return.numpy()[0] #tf dqn agent method
