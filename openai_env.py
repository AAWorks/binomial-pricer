import gym
import numpy as np

class Env:
    def __init__(self, spot=100.0, strike=100.0, r=0.02, sigma=0.20, time=1.0, days=365): 
        self._spot = spot
        self._strike = strike
        self._r = r
        self._sigma = sigma
        self._time = time
        self._days = days

        self._sinit, self._reward, self._day_step = 0, 0, 0
        # from day 0 taking N steps to day N

        self.action_space = gym.spaces.Discrete(2) # Two potential classes --> 0: hold, 1:exercise
        self.observation_space = gym.spaces.Box(low=np.array([0, 0]), high=np.array([np.inf, 1.0]), dtype=np.float32)