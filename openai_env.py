import gym
import numpy as np
import datetime

class Env:
    def __init__(self, spot=100.0, strike=100.0, r=0.02, sigma=0.20, today=datetime.date.today(), maturity=None): 
        self._spot = spot
        self._strike = strike
        self._r = r
        self._sigma = sigma

        if not maturity:
            maturity = datetime.date(today.year + 1, today.month, today.day)

        delta = maturity - today
        self._days = delta / datetime.timedelta(days=1)
        self._float_time = self._days / 365

        self._s_new, self._reward, self._day_step = 0, 0, 0
        # from day 0 taking N steps to day N

        self.action_space = gym.spaces.Discrete(2) # Two potential classes --> 0: hold, 1:exercise
        self.observation_space = gym.spaces.Box(low=np.array([0, 0]), high=np.array([np.inf, 1.0]), dtype=np.float32)
    
    def single_step(self, action: int):
        if action == 1: # exercise opt
            reward = max(self._strike - self._s_new, 0.0) * np.exp(-self._r * self._float_time * (self._day_step / self._days))
            done = True
        else: # hold opt
            if self._day_step == self._days: # if at maturity
                reward = max(self._strike - self._s_new, 0.0) * np.exp(-self._r * self._time)
                done = True
            else: # check tmr
                reward = 0
                self._s_new = self._s_new * np.exp((self._rate - 0.5 * self._sigma ** 2) * (self._time / self._days) + self._sigma * np.sqrt(self._time / self._days) * np.random.normal())
                self._day_step += 1
                done = False

        tao = 1.0 - self._day_step / self.days # time remaining in yrs

        return np.array([self.S1, tao]), reward, done, {}

    def reset(self):
        self._day_step = 0
        self._s_new = self._spot
        tao = 1.0 - self._day_step / self._days        # time to maturity, in unit of years
        return [self.S1, tao]
