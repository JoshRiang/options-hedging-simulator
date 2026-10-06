"""Vol-mismatch, rebalance-frequency and jump experiments."""
from __future__ import annotations

import numpy as np
import pandas as pd

from hedgesim.simulator import hedge_paths, simulate_gbm, simulate_merton_jump


def run_vol_mismatch(S0=100.0, K=100.0, T=0.25, r=0.02, sigma_true=0.25,
                     hedge_vols=(0.15, 0.20, 0.25, 0.30, 0.35),
                     n_paths=2000, n_steps=63, seed=7,
                     option="call") -> pd.DataFrame:
    paths = simulate_gbm(S0, mu=r, sigma=sigma_true, T=T,
                         n_steps=n_steps, n_paths=n_paths, seed=seed)
    rows = []
    for sv in hedge_vols:
        pnl = hedge_paths(paths, K, T, r, sv, rebalance_every=1, option=option)
        rows.append({"hedge_vol": sv, "mean_pnl": float(np.mean(pnl)),
                     "std_pnl": float(np.std(pnl, ddof=1)),
                     "p5": float(np.quantile(pnl, 0.05)),
                     "p95": float(np.quantile(pnl, 0.95))})
    return pd.DataFrame(rows)


def run_rebalance_sweep(S0=100.0, K=100.0, T=0.25, r=0.02, sigma=0.25,
                        freqs=(1, 5, 21), n_paths=2000, n_steps=63,
                        seed=7, option="call") -> pd.DataFrame:
    paths = simulate_gbm(S0, mu=r, sigma=sigma, T=T,
                         n_steps=n_steps, n_paths=n_paths, seed=seed)
    rows = []
    for f in freqs:
        pnl = hedge_paths(paths, K, T, r, sigma, rebalance_every=f, option=option)
        rows.append({"rebalance_every": f, "mean_pnl": float(np.mean(pnl)),
                     "std_pnl": float(np.std(pnl, ddof=1)),
                     "rmse": float(np.sqrt(np.mean(pnl**2)))})
    return pd.DataFrame(rows)


def run_jump_experiment(S0=100.0, K=100.0, T=0.25, r=0.02, sigma=0.20,
                        sigma_hedge=0.20, jump_intensity=2.0, jump_mean=-0.05,
                        jump_std=0.08, n_paths=2000, n_steps=63,
                        seed=7, option="call") -> dict:
    base = simulate_gbm(S0, mu=r, sigma=sigma, T=T, n_steps=n_steps,
                        n_paths=n_paths, seed=seed)
    jumpy = simulate_merton_jump(S0, mu=r, sigma=sigma, T=T, n_steps=n_steps,
                                 jump_intensity=jump_intensity, jump_mean=jump_mean,
                                 jump_std=jump_std, n_paths=n_paths, seed=seed)
    pnl_base = hedge_paths(base, K, T, r, sigma_hedge, option=option)
    pnl_jump = hedge_paths(jumpy, K, T, r, sigma_hedge, option=option)
    return {
        "no_jump_mean": float(np.mean(pnl_base)),
        "no_jump_std": float(np.std(pnl_base, ddof=1)),
        "jump_mean": float(np.mean(pnl_jump)),
        "jump_std": float(np.std(pnl_jump, ddof=1)),
        "jump_p5": float(np.quantile(pnl_jump, 0.05)),
    }
