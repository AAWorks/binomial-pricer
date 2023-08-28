from models.baseline_tfa_dqn import TFAModel
from models.openai_env import OptionEnv

from datetime import date, timedelta

def get_custom_defaults():
    defaults = {
        "maturity": date.today() + timedelta(days=1),
        "custom_maturity": date.today() + timedelta(days=365),
        "risk_free_rate": 0.2,
        "spot": 100.0,
        "strike": 100.0,
        "implied_volatility": 0.20,
        "dividend_rate": 0.01
    }

    return defaults

DEFAULTS = get_custom_defaults()

test_defs = {k: v for k, v in DEFAULTS.items() if k != "maturity"}
test_defs["maturity"] = test_defs["custom_maturity"]
del test_defs["custom_maturity"]

option = TFAModel(OptionEnv, test_defs)
with open("data/dqn_log.txt", "a") as f:
    f.write("Initializing Agent...\n")

option.init_agent()
with open("data/dqn_log.txt", "a") as f:
    f.write("Building Replay Buffer...\n")

option.build_replay_buffer()

with open("data/dqn_log.txt", "a") as f:
    f.write("Training Model...\n")

option.train()

with open("data/dqn_log.txt", "a") as f:    
    f.write("Pricing Option...\n")

option.calculate_npv()