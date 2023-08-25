from models.abstract import BaseOption

from openai_env import OptionEnv
from models.baseline_tfa_dqn import TFAModel

from models.binomial_tree import EUBinomialTree, USBinomialTree
from models.black_scholes import BlackScholes
from models.monte_carlo import MonteCarlo

MODELS = {
    "eu": ["Black Scholes", "Binomial Tree", "Monte Carlo"],
    "us": ["Deep Q-Network", "Binomial Tree", "Monte Carlo"]
}

class USOption(BaseOption):
    _model_map = {
        "Binomial Tree": USBinomialTree,
        "Monte Carlo": MonteCarlo
    }

    def __init__(self, **kwargs):
        self._kwargs = kwargs
        super().__init__(kwargs)
    
    def priced(self, model: str):
        return self._model_map[model](self._kwargs)
    
    def dqn(self):
        return None

    def all(self):
        return [self.priced(model) for model in self._model_map.keys()]


class EUOption(BaseOption):
    _model_map = {
        "Black Scholes": BlackScholes,
        "Binomial Tree": EUBinomialTree,
        "Monte Carlo": MonteCarlo
    }

    def __init__(self, **kwargs):
        self._kwargs = kwargs
        super().__init__(kwargs)
    
    def priced(self, model: str):
        return self._model_map[model](self._kwargs)
    
    def all(self):
        return [self.priced(model) for model in self._model_map.keys()]
