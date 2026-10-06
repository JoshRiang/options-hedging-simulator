"""Path simulation (GBM + Merton jumps) and discrete delta-hedge loop."""
from __future__ import annotations

import numpy as np

from hedgesim.bs import bs_price, delta


def simulate_gbm(S0: float, mu: float, sigma: float, T: float, n_steps: int,
                 n_paths: int = 1, seed: int = 0) -> np.ndarray:
    """Return (n_paths, n_steps+1) price paths under GBM."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = S0
    shock = rng.standard_normal((n_paths, n_steps))
    for i in range(1, n_steps + 1):
        paths[:, i] = paths[:, i - 1] * np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * math_sqrt(dt) * shock[:, i - 1])
    return paths


def math_sqrt(x: float) -> float:
    return x ** 0.5


def simulate_merton_jump(S0: float, mu: float, sigma: float, T: float, n_steps: int,
                         jump_intensity: float = 1.0, jump_mean: float = -0.05,
                         jump_std: float = 0.10, n_paths: int = 1,
                         seed: int = 0) -> np.ndarray:
    """Merton jump-diffusion paths: Poisson jumps with lognormal sizes."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = S0
    for i in range(1, n_steps + 1):
        z = rng.standard_normal(n_paths)
        n_jumps = rng.poisson(jump_intensity * dt, size=n_paths)
        j = rng.normal(jump_mean, jump_std, size=n_paths) * n_jumps
        paths[:, i] = paths[:, i - 1] * np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * math_sqrt(dt) * z + j)
    return paths


def _payoff(S: float, K: float, option: str) -> float:
    return max(S - K, 0.0) if option == "call" else max(K - S, 0.0)


def delta_hedge_path(S: np.ndarray, K: float, T: float, r: float, sigma_hedge: float,
                     rebalance_every: int = 1, option: str = "call",
                     tx_cost: float = 0.0) -> dict:
    """Delta-hedge a SHORT 1-unit option along a price path.

    Returns dict with premium, payoff, hedge pnl series, cash account,
    final hedged pnl (= stock + cash - payoff), trades, n_rebalances.
    """
    n_steps = len(S) - 1
    dt = T / n_steps
    disc = float(np.exp(r * dt))
    ttm = lambda i: max(T - i * dt, 0.0)  # noqa: E731

    premium = bs_price(float(S[0]), K, T, r, sigma_hedge, option)
    d = delta(float(S[0]), K, T, r, sigma_hedge, option)
    shares = d  # long delta against short option
    cash = premium - shares * float(S[0])
    n_reb = 0
    costs = 0.0
    stock_trades = [shares]

    for i in range(1, n_steps + 1):
        cash *= disc
        if i < n_steps and (i % rebalance_every == 0):
            d_new = delta(float(S[i]), K, ttm(i), r, sigma_hedge, option)
            trade = d_new - shares
            cash -= trade * float(S[i]) + abs(trade) * float(S[i]) * tx_cost
            costs += abs(trade) * float(S[i]) * tx_cost
            shares = d_new
            n_reb += 1
        stock_trades.append(shares)

    stock_val = shares * float(S[-1])
    payoff = _payoff(float(S[-1]), K, option)
    total = stock_val + cash - payoff
    return {
        "premium": premium,
        "payoff": payoff,
        "stock_val": stock_val,
        "cash": cash,
        "total_pnl": total,
        "tx_costs": costs,
        "n_rebalances": n_reb,
        "final_shares": shares,
    }


def hedge_paths(paths: np.ndarray, K: float, T: float, r: float, sigma_hedge: float,
                rebalance_every: int = 1, option: str = "call",
                tx_cost: float = 0.0) -> np.ndarray:
    """Vectorised-over-paths convenience: returns array of total PnL per path."""
    return np.array([delta_hedge_path(p, K, T, r, sigma_hedge,
                                      rebalance_every, option, tx_cost)["total_pnl"]
                     for p in paths])
