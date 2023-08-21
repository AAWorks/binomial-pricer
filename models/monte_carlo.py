import torch

from models.abstract import Option

class MonteCarloOption(Option):
    def __init__(self, **kwargs):
        super().__init__(kwargs, with_tensors=True, name="Monte Carlo")

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
