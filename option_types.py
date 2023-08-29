from models.abstract import BaseOption

from models.openai_env import OptionEnv
from models.baseline_tfa_dqn import TFAModel

from models.binomial_tree import EUBinomialTree, USBinomialTree
from models.black_scholes import BlackScholes
from models.monte_carlo import EUMonteCarlo, ASMonteCarlo

MODELS = {
    "eu": ["Black Scholes", "Binomial Tree", "Monte Carlo"],
    "us": ["Binomial Tree", "Deep Q-Network"],
    "as": ["Monte Carlo"]
}

class USOption(BaseOption):
    _model_map = {
        "Binomial Tree": USBinomialTree,
        "Deep Q-Network": None
    }

    def __init__(self, **kwargs):
        self._kwargs = kwargs
        super().__init__(kwargs)
    
    def _dqn(self):
        option = TFAModel(OptionEnv, self._kwargs)
        option.init_agent()
        option.build_replay_buffer()
        option.train()
        option.calculate_npv()
        priced_option = option
        return priced_option
    
    def priced(self, model: str):
        if model == "Deep Q-Network":
            return self._dqn()
        return self._model_map[model](self._kwargs)

    def all(self):
        return [self.priced(model) for model in self._model_map.keys()]


class EUOption(BaseOption):
    _model_map = {
        "Black Scholes": BlackScholes,
        "Binomial Tree": EUBinomialTree,
        "Monte Carlo": EUMonteCarlo
    }

    def __init__(self, **kwargs):
        self._kwargs = kwargs
        super().__init__(kwargs)
    
    def priced(self, model: str):
        return self._model_map[model](self._kwargs)
    
    def all(self):
        return [self.priced(model) for model in self._model_map.keys()]


class ASOption(BaseOption):
    _model_map = {
        "Monte Carlo": ASMonteCarlo
    }
