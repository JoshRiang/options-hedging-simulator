"""Synthetic-only tests: BS math, hedge loop, experiments, attribution."""
import numpy as np

from hedgesim.attribution import attribute_pnl
from hedgesim.bs import bs_price, delta, gamma, implied_vol, theta, vega
from hedgesim.experiments import run_jump_experiment, run_rebalance_sweep, run_vol_mismatch
from hedgesim.simulator import delta_hedge_path, hedge_paths, simulate_gbm, simulate_merton_jump

S0, K, T, R, SIG = 100.0, 100.0, 1.0, 0.05, 0.2


def test_call_price_known_value():
    assert abs(bs_price(S0, K, T, R, SIG, "call") - 10.4506) < 1e-3


def test_put_call_parity():
    c = bs_price(S0, K, T, R, SIG, "call")
    p = bs_price(S0, K, T, R, SIG, "put")
    assert abs((c - p) - (S0 - K * np.exp(-R * T))) < 1e-9


def test_greeks_bounds():
    assert 0.0 < delta(S0, K, T, R, SIG, "call") < 1.0
    assert -1.0 < delta(S0, K, T, R, SIG, "put") < 0.0
    assert gamma(S0, K, T, R, SIG) > 0
    assert vega(S0, K, T, R, SIG) > 0
    assert theta(S0, K, T, R, SIG, "call") < 0


def test_implied_vol_roundtrip():
    price = bs_price(S0, K, T, R, SIG, "call")
    assert abs(implied_vol(price, S0, K, T, R, "call") - SIG) < 1e-4


def test_gbm_shape_and_determinism():
    a = simulate_gbm(100, 0.02, 0.25, 0.25, 10, n_paths=5, seed=1)
    b = simulate_gbm(100, 0.02, 0.25, 0.25, 10, n_paths=5, seed=1)
    assert a.shape == (5, 11) and np.array_equal(a, b) and (a > 0).all()


def test_jump_paths_valid():
    p = simulate_merton_jump(100, 0.02, 0.2, 0.25, 20, n_paths=10, seed=3)
    assert p.shape == (10, 21) and np.isfinite(p).all() and (p > 0).all()


def test_hedge_true_vol_near_zero_mean():
    paths = simulate_gbm(100, R, SIG, 0.25, 63, n_paths=1500, seed=11)
    pnl = hedge_paths(paths, 100, 0.25, R, SIG)
    assert abs(np.mean(pnl)) < 0.5  # discrete hedge ~ unbiased under true vol
    assert np.std(pnl) > 0  # but noisy


def test_coarser_rebalance_increases_rmse():
    paths = simulate_gbm(100, R, SIG, 0.25, 63, n_paths=800, seed=5)
    fine = hedge_paths(paths, 100, 0.25, R, SIG, rebalance_every=1)
    coarse = hedge_paths(paths, 100, 0.25, R, SIG, rebalance_every=21)
    assert float(np.sqrt(np.mean(coarse**2))) > float(np.sqrt(np.mean(fine**2)))


def test_vol_mismatch_monotone_mean():
    df = run_vol_mismatch(sigma_true=0.25, hedge_vols=(0.15, 0.25, 0.35),
                          n_paths=1500, n_steps=63, seed=9)
    means = df.set_index("hedge_vol")["mean_pnl"]
    # Short call hedged with higher vol -> over-hedged -> systematic profit
    assert means.loc[0.35] > means.loc[0.25] > means.loc[0.15]


def test_rebalance_sweep_shape():
    df = run_rebalance_sweep(n_paths=300, n_steps=63, seed=2)
    assert list(df.columns) == ["rebalance_every", "mean_pnl", "std_pnl", "rmse"]
    assert (df["rmse"].iloc[-1] >= df["rmse"].iloc[0])


def test_jump_raises_tail_risk():
    out = run_jump_experiment(n_paths=800, n_steps=63, seed=4)
    assert out["jump_std"] > out["no_jump_std"]
    assert out["jump_p5"] < 0


def test_attribution_identity():
    S = simulate_gbm(100, R, SIG, 0.25, 63, n_paths=1, seed=0)[0]
    res = delta_hedge_path(S, 100, 0.25, R, SIG)
    att = attribute_pnl(res)
    assert abs(att["check_sum"] - att["total_hedged_pnl"]) < 1e-9
