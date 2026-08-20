"""
UPP Transport Optimization Package (src)
========================================
Modules:
- config: Paths, parameters, and UPP domain constants.
- data_generator: Stochastic Poisson demand generation engine.
- regression_model: Ordinary Least Squares (OLS) regression & diagnostic tools.
- ucb_bandit: UCB1 Multi-Armed Bandit decision engine.
"""

from src.config import DATA_DIR, PLOTS_DIR
from src.data_generator import generate_demand_dataset
from src.regression_model import fit_ols, calculate_diagnostics, load_and_filter_data
from src.ucb_bandit import UCB1Agent, evaluate_policy_dispatch, calculate_reward

__all__ = [
    'DATA_DIR',
    'PLOTS_DIR',
    'generate_demand_dataset',
    'fit_ols',
    'calculate_diagnostics',
    'load_and_filter_data',
    'UCB1Agent',
    'evaluate_policy_dispatch',
    'calculate_reward'
]
