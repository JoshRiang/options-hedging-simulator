"""hedgesim: delta-hedging simulator for an options book."""
from hedgesim.bs import bs_price, delta, gamma, vega, theta, implied_vol
from hedgesim.simulator import simulate_gbm, simulate_merton_jump, delta_hedge_path
from hedgesim.experiments import run_vol_mismatch, run_rebalance_sweep, run_jump_experiment
from hedgesim.attribution import attribute_pnl

__all__ = [
    "bs_price", "delta", "gamma", "vega", "theta", "implied_vol",
    "simulate_gbm", "simulate_merton_jump", "delta_hedge_path",
    "run_vol_mismatch", "run_rebalance_sweep", "run_jump_experiment",
    "attribute_pnl",
]

__version__ = "0.1.0"
