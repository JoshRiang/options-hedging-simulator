"""Data loading: synthetic GBM paths (default) or Yahoo prices (optional demo)."""
from __future__ import annotations

import numpy as np

from hedgesim.simulator import simulate_gbm


def load_synthetic(S0=100.0, mu=0.02, sigma=0.25, T=0.25, n_steps=63,
                   n_paths=1, seed=7) -> np.ndarray:
    return simulate_gbm(S0, mu, sigma, T, n_steps, n_paths, seed)


def load_yahoo_close(ticker: str, period: str = "1y") -> np.ndarray:
    """Fetch daily closes via yfinance; raises informative error if unavailable."""
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError("yfinance not installed; pip install yfinance") from e
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False,
                     threads=False)
    if df is None or len(df) == 0:
        raise ValueError(f"No data for ticker {ticker!r}")
    if hasattr(df.columns, "levels"):  # MultiIndex on newer yfinance
        df.columns = [c[0] for c in df.columns]
    col = "Close" if "Close" in df.columns else "close"
    return np.asarray(df[col], dtype=float).ravel()
