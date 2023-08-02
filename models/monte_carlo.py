import torch

class MonteCarloOption:
    def __init__(self, strike, stock, vol, time, risk_rate, dividend_rate):
        self._vol = vol
        self._time = time
        self._r_rate = risk_rate
        self._d_rate = dividend_rate
        self._stock = stock
        self._strike = strike

    @property
    def price(self):
        torch.manual_seed(42)
        scenarios = 1000000
        dW = self._vol * self._time ** 0.5 * torch.randn(size=(scenarios,))
        r = torch.exp((self._r_rate - self._d_rate - self._vol * self._vol / 2) * self._time + dW)

        payoff = torch.max(self._strike - self._stock*r, torch.zeros(size=(scenarios,)))
        return torch.mean(payoff) * torch.exp(-self._rate*self._time)
    
    @property
    def greeks(self):
        ov = self.price
        ov.backward()
        return {
            "delta": self._stock.grad,
            "rho": self._rate.grad,
            "vega": self._vol.grad,
            "theta": self._time.grad,
            "epsilon": self._dividend.grad,
            "strike_greek": self._strike.grad
        }

    def viz(self):
        """visualize distro"""


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
