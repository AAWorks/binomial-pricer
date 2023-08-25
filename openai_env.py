import gym
import numpy as np
import datetime

class OptionEnv(gym.Env):
    def __init__(self, **kwargs): 
        today=datetime.date.today()
        self._spot = kwargs["spot"]
        self._strike = strike
        self._r = r
        self._sigma = sigma
        self._today = today

        if not maturity:
            maturity = datetime.date(today.year + 1, today.month, today.day)

        self._t = maturity - self._today

        self._s_new, self._reward, self._day_step = 0, 0, 0
        # from day 0 taking N steps to day N

        self.action_space = gym.spaces.Discrete(2) # Two potential classes --> 0: hold, 1:exercise
        self.observation_space = gym.spaces.Box(low=np.array([0, 0]), high=np.array([np.inf, 1.0]), dtype=np.float32)
    
    @property
    def spot(self):
        return self._spot

    @property
    def strike(self):
        return self._strike
    
    @property
    def risk_free_rate(self):
        return self._r

    @property
    def implied_volatility(self):
        return self._sigma
    
    @property
    def n_days(self):
        return self._t / datetime.timedelta(days=1)

    @property
    def float_time(self):
        return self.n_days / 365
    
    def single_step(self, action: int):
        if action == 1: # exercise opt
            reward = max(self._strike - self._s_new, 0.0) * np.exp(-self._r * self.float_time * (self._day_step / self.n_days))
            done = True
        else: # hold opt
            if self._day_step == self.n_days: # if at maturity
                reward = max(self._strike - self._s_new, 0.0) * np.exp(-self._r * self.float_time)
                done = True
            else: # check tmr
                reward = 0
                self._s_new = self._s_new * np.exp((self._r - 0.5 * self._sigma ** 2) * (self.float_time / self.n_days) + self._sigma * np.sqrt(self.float_time / self.n_days) * np.random.normal())
                self._day_step += 1
                done = False

        tao = 1.0 - self._day_step / self.n_days # time remaining in yrs

        return np.array([self._s_new, tao]), reward, done, {}

    def reset(self):
        self._day_step = 0
        self._s_new = self._spot
        tao = 1.0 - self._day_step / self.n_days        # time to maturity, in unit of years
        return [self._s_new, tao]

    def simulate_price_data(self):
        tmp = self.reset()

        sim_prices = []
        sim_prices.append(tmp[0])
        for _ in range(365):
            action = 0
            s_next, _, _, _ = self.single_step(action)
            sim_prices.append(s_next[0])
        
        return sim_prices
