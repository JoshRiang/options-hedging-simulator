"""CLI demo: delta-hedge a short ATM call on synthetic (or Yahoo) data."""
from __future__ import annotations

import argparse
import numpy as np

from hedgesim.attribution import attribute_pnl
from hedgesim.experiments import run_jump_experiment, run_rebalance_sweep, run_vol_mismatch
from hedgesim.io_loader import load_synthetic, load_yahoo_close
from hedgesim.simulator import delta_hedge_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Delta-hedging simulator demo")
    p.add_argument("--S0", type=float, default=100.0)
    p.add_argument("--K", type=float, default=100.0)
    p.add_argument("--T", type=float, default=0.25)
    p.add_argument("--r", type=float, default=0.02)
    p.add_argument("--sigma", type=float, default=0.25)
    p.add_argument("--sigma-hedge", type=float, default=0.25)
    p.add_argument("--n-paths", type=int, default=2000)
    p.add_argument("--n-steps", type=int, default=63)
    p.add_argument("--rebalance", type=int, default=1)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--ticker", type=str, default=None,
                   help="Optional Yahoo ticker: hedge realised path instead of synthetic")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    print(f"Delta-hedge short ATM call: S0={args.S0} K={args.K} T={args.T} "
          f"r={args.r} sigma_true={args.sigma} sigma_hedge={args.sigma_hedge}")

    if args.ticker:
        try:
            S = load_yahoo_close(args.ticker)
            T = args.T
            res = delta_hedge_path(S, args.K or float(S[0]), T, args.r,
                                   args.sigma_hedge, args.rebalance)
            print(f"Realised path ({args.ticker}, n={len(S)}): {attribute_pnl(res)}")
        except Exception as e:
            print(f"Yahoo load failed ({e}); falling back to synthetic.")
            args.ticker = None

    if not args.ticker:
        paths = load_synthetic(args.S0, args.r, args.sigma, args.T,
                               args.n_steps, args.n_paths, args.seed)
        res0 = delta_hedge_path(paths[0], args.K, args.T, args.r,
                                args.sigma_hedge, args.rebalance)
        print("Single-path attribution:", attribute_pnl(res0))

        print("\n[1] Vol-mismatch (true sigma=%.2f):" % args.sigma)
        print(run_vol_mismatch(args.S0, args.K, args.T, args.r, args.sigma,
                               n_paths=args.n_paths, n_steps=args.n_steps,
                               seed=args.seed).to_string(index=False))
        print("\n[2] Rebalance-frequency sweep:")
        print(run_rebalance_sweep(args.S0, args.K, args.T, args.r, args.sigma,
                                  n_paths=args.n_paths, n_steps=args.n_steps,
                                  seed=args.seed).to_string(index=False))
        print("\n[3] Jump experiment (Merton, downside jumps):")
        print(run_jump_experiment(args.S0, args.K, args.T, args.r, args.sigma,
                                  args.sigma_hedge, n_paths=args.n_paths,
                                  n_steps=args.n_steps, seed=args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
