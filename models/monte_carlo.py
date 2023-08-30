import torch
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from models.abstract import Model

class MonteCarlo(Model):
    def __init__(self, params, region):
        super().__init__(params, with_tensors=True, name="Monte Carlo")
        self._prices=None
        self._region=region

    @property
    def npv(self):
        torch.manual_seed(42)
        scenarios = 1000000

        if self._region == "eu":
            w_t = torch.sqrt(self._time) * torch.randn(size=(scenarios,)) #Brownian Motion
        else:
            dt = torch.tensor(1 / 252)
            w_t = torch.cumsum(torch.sqrt(dt) * torch.randn([scenarios, int(self._time * 252)]), 1)

        dW = self._iv * w_t
        self._prices = self._spot * torch.exp((self._r - self._d - self._iv * self._iv / 2) * self._time + dW)

        if self._region == "eu":
            if self._option_type == "P":
                payoff = torch.max(self._strike - self._prices.clone(), torch.zeros(size=(scenarios,)))
            else:
                payoff = torch.max(self._prices.clone() - self._strike, torch.zeros(size=(scenarios,)))
        else:
            if self._option_type == "P":
                payoff = torch.max(self._strike - torch.mean(self._prices.clone(), axis=1), torch.zeros(scenarios))
            else:
                payoff = torch.max(torch.mean(self._prices.clone(), axis=1) - self._strike, torch.zeros(scenarios))

        return torch.mean(payoff) * torch.exp(-self._r*self._time)

    def _euo_plot(self):
        price_data = self._prices #.clone()
        data = price_data.detach().numpy()
        fig, ax = plt.subplots()
        #ax.rcParams["figure.figsize"] = (15, 10)
        ax.hist(data, bins=25)
        plt.xlabel("Prices")
        plt.ylabel("Occurences")
        plt.title("Distribution of Underlying Price after 1 Year")
        return fig

    def _aso_plot(self):
        price_data = self._prices #.clone()
        data = price_data[0, :].detach().numpy()
        fig, ax = plt.subplots()
        ax.plot(data)
        plt.xlabel("Number of Days in Future")
        plt.ylabel("Underlying Price")
        plt.title("One Possible Price path")
        plt.axhline(y=torch.mean(price_data[0, :]).detach().numpy(), color="r", linestyle="--")
        plt.axhline(y=100, color='g', linestyle="--")
        return fig
    
    @property
    def plot(self):
        plt_types = {"eu": self._euo_plot, "as": self._aso_plot}
        return plt_types.get(self._region)()
    
    @property
    def greeks(self):
        ov = self.npv
        ov.backward()
        return {
            "delta": self._spot.grad,
            "rho": self._r.grad,
            "vega": self._iv.grad,
            "theta": self._time.grad,
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
        st.subheader("Single Path Visualization")
        st.pyplot(self.plot)
        st.divider()

class EUMonteCarlo(MonteCarlo):
    def __init__(self, params):
        super().__init__(params, "eu")

class ASMonteCarlo(MonteCarlo):
    def __init__(self, params):
        super().__init__(params, "as")


def _simulate_ep(policy, env, time_step, base_return=0.0):
    while not time_step.is_last():
        action_step = policy.action(time_step)
        time_step = env.step(action_step.action)
        base_return += time_step.reward
    
    return base_return

def _simulate_eps(policy, env, eps, st_display=False):
    step = env.reset()
    total = 0

    if st_display:
        bar = st.progress(0.0, text=f"Simulating Episodes... (0/{eps})")
    for ep in range(eps):
        total += _simulate_ep(policy, env, step)
        if st_display:
            bar.progress(float(ep / eps), text=f"Simulating Episodes... ({ep}/{eps})")
    
    if st_display:
        bar.progress(1.0, text="Simulation Complete")

    return total

def dqn_sim(policy, env, eps=10, base_return=0.0, st_display=False):
    avg_return = (base_return + _simulate_eps(policy, env, eps, st_display=st_display)) / eps
    return avg_return.numpy()[0] #tf dqn agent method
