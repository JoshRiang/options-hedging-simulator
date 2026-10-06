# Options Hedging Simulator

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Tests: 12 passed](https://img.shields.io/badge/pytest-12%20passed-green)](tests/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)

Delta-hedge a short European option through time and measure how **discrete
rebalancing**, **volatility misspecification**, and **jumps** create P&L.
Paths are **synthetic** by default (GBM / Merton jump-diffusion, seeded RNG);
a **live** Yahoo-price path is available via `--ticker` with automatic
synthetic fallback offline.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m hedgesim                                   # full demo on synthetic paths
python -m hedgesim --sigma 0.3 --sigma-hedge 0.2 --n-paths 5000
python -m hedgesim --ticker AAPL --K 200              # hedge realised Yahoo path (live)
python -m pytest tests/ -v
```

## Architecture / Method

**Pricer** (`hedgesim/bs.py`) — Black-Scholes for European options, no
dividends, with `d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)`, `d2 = d1 − σ√T`:

- Price: `C = S·N(d1) − K·e^(−rT)·N(d2)` (put via parity)
- Delta: `N(d1)` (call), `N(d1) − 1` (put); Gamma: `φ(d1)/(S·σ√T)`;
  Vega: `S·φ(d1)√T`; Theta per year; implied vol by bisection.

**Paths** (`hedgesim/simulator.py`):

- GBM: `S_{t+dt} = S_t·exp((μ − σ²/2)·dt + σ√dt·Z)`
- Merton jump-diffusion: same diffusion plus Poisson(`λ·dt`) jumps with
  lognormal sizes `N(jump_mean, jump_std)`.

**Hedge loop** (`delta_hedge_path`) — short 1 unit of the option: receive the
BS premium at `σ_hedge`, go long `Δ` shares, fund from a cash account accruing
at `e^(r·dt)`; re-hedge every `k` steps; optional proportional transaction
costs. Final hedged P&L `= stock value + cash − payoff`.

**Attribution** (`attribution.py`) — additive split satisfying the exact
identity `total = premium − payoff + residual − costs` (checksum-verified):
`option_premium_received`, `option_payoff_paid`, `delta_trading_plus_interest`
(residual), `transaction_costs`. **Experiments** (`experiments.py`) — vol-mismatch
sweep, rebalance-frequency sweep, jump experiment. **Loader** (`io_loader.py`) —
synthetic GBM (default) or Yahoo closes (live, labelled as such).

## Sample output (real run)

```bash
python -m hedgesim --n-paths 2000 --n-steps 63 --seed 7
```

```
Delta-hedge short ATM call: S0=100.0 K=100.0 T=0.25 r=0.02 sigma_true=0.25 sigma_hedge=0.25
Single-path attribution: {'option_premium_received': 5.224453276436321, 'option_payoff_paid': -0.0, 'delta_trading_plus_interest': -5.039535479556232, 'transaction_costs': -0.0, 'total_hedged_pnl': 0.18491779688008989, 'check_sum': 0.18491779688008947}

[1] Vol-mismatch (true sigma=0.25):
 hedge_vol  mean_pnl  std_pnl        p5       p95
      0.15 -1.991702 1.161378 -4.147093 -0.537579
      0.20 -0.995903 0.741870 -2.475165 -0.104846
      0.25 -0.002668 0.575967 -0.946593  0.924010
      0.30  0.989188 0.651140  0.113874  2.159444
      0.35  1.980266 0.826386  0.762004  3.455073

[2] Rebalance-frequency sweep:
 rebalance_every  mean_pnl  std_pnl     rmse
               1 -0.002668 0.575967 0.575829
               5 -0.041168 1.204554 1.204957
              21 -0.033695 2.294739 2.294412

[3] Jump experiment (Merton, downside jumps):
{'no_jump_mean': -0.0026682682926640177, 'no_jump_std': 0.575966588044737, 'jump_mean': -0.5604060459572229, 'jump_std': 1.5908976859329134, 'jump_p5': -3.5741214407602926}
```

## Results

1. **Vol mismatch** — hedging with `σ_hedge` above true vol systematically
   over-charges (mean +0.99 at 0.30, +1.98 at 0.35); below true vol it bleeds
   (−1.99 at 0.15, −1.00 at 0.20); at the true vol the mean is ≈ 0 (−0.0027).
   Textbook volatility-risk-premium mechanics, reproduced quantitatively.
2. **Rebalance sweep** — under the correct vol the mean stays ≈ 0 but RMSE
   grows 0.58 → 1.20 → 2.29 as rebalancing coarsens from every step to every
   21 steps: discrete gamma bleed.
3. **Jumps** — unhedgeable downside jumps drag the mean to −0.56, nearly triple
   the std (1.59 vs 0.58), and fatten the left tail (p5 −3.57) even though the
   diffusive vol is "right".

## Limitations

- Constant-vol Black-Scholes hedge; no stochastic vol, no vol surface, no
  smile — vega risk is unmodelled.
- European exercise only, no dividends, no borrow constraints, no market impact.
- Yahoo path is a single realised trajectory, not a statistical sample; live
  fetch requires network and `yfinance`.

## Project structure

```
options-hedging-simulator/
├── hedgesim/
│   ├── bs.py           # Black-Scholes price, delta/gamma/vega/theta, implied vol
│   ├── simulator.py    # GBM + Merton jump paths, discrete delta-hedge loop
│   ├── experiments.py  # vol-mismatch, rebalance-frequency, jump experiments
│   ├── attribution.py  # additive P&L split with exact checksum identity
│   ├── io_loader.py    # synthetic GBM loader + optional Yahoo closes (live)
│   └── __main__.py     # CLI demo
├── tests/
│   └── test_hedgesim.py  # BS reference values, parity, greeks, hedge, attribution
├── requirements.txt
├── LICENSE
└── README.md
```

## Testing

```bash
python -m pytest tests/ -v   # 12 passed, synthetic only, no network
```

Covers the BS call reference value (10.4506 at S=K=100, T=1, r=0.05, σ=0.2),
put-call parity to 1e-9, greek bounds/signs, implied-vol round-trip, hedge-loop
sanity, experiment shapes, and the exact attribution checksum.

## License

MIT — see [LICENSE](LICENSE).
