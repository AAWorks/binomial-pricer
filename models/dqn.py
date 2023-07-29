import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import tempfile

from tf_agents.environments import  gym_wrapper           # wrap OpenAI gym
from tf_agents.environments import tf_py_environment      # gym to tf gym
from tf_agents.networks import q_network                  # Q net
from tf_agents.agents.dqn import dqn_agent                # DQN Agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer      # replay buffer
from tf_agents.trajectories import trajectory              # s->s' trajectory
from tf_agents.utils import common                       # loss function

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


class Model:
    def __init__(self, 
                 iterations: int = 20000,
                 steps: int = 10,
                 repbuffer_len: int = 100000,
                 batch_size: int = 256,
                 learning_r: int = 1e-3,
                 n_eps: int = 10,
                 eval_interval: int = 1000,
                 log_interval: int = 200
                 ): # hyperparameters

        self._num_iterations = iterations
        self._collect_steps_per_iteration = steps
        self._replay_buffer_max_length = repbuffer_len
        self._batch_size = batch_size

        self._learning_rate = learning_r
        self._num_eval_episodes = n_eps 

        self._eval_interval = eval_interval 
        self._log_interval = log_interval

    
