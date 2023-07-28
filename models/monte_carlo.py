import numpy as np


class MonteCarloOption:
    def __init__(self):
        pass

    @property
    def price(self):
        pass

    def viz(self):
        pass


class GymSim:
    def __init__(self, environment, policy, episodes=10):
        self._environment = environment
        self._policy = policy
        self._episodes = episodes

    def _simulate_ep(self, time_step, base_return=0.0):
        while not time_step.is_last():
            action_step = self._policy.action(time_step)
            time_step = self._environment.step(action_step.action)
            base_return += time_step.reward
        
        return base_return
    
    def _simulate_eps(self):
        step = self._environment.reset()
        return sum([self._simulate_ep(step) for _ in range(self._episodes)])

    def run_sim(self, base_return=0.0):
        avg_return = (base_return + self._simulate_eps()) / self._episodes
        return avg_return.numpy()[0]
