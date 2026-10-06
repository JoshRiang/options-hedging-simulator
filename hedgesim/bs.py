"""Black-Scholes pricer + greeks (European, no dividends)."""
from __future__ import annotations

import math

SQRT2PI = math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT2PI


def _d1d2(S: float, K: float, T: float, r: float, sigma: float):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        raise ValueError("S, K, T, sigma must be positive")
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def bs_price(S: float, K: float, T: float, r: float, sigma: float, option: str = "call") -> float:
    """European Black-Scholes price. `option` in {"call", "put"}."""
    if T <= 0:
        payoff = max(S - K, 0.0) if option == "call" else max(K - S, 0.0)
        return payoff
    d1, d2 = _d1d2(S, K, T, r, sigma)
    if option == "call":
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    if option == "put":
        return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)
    raise ValueError("option must be 'call' or 'put'")


def delta(S: float, K: float, T: float, r: float, sigma: float, option: str = "call") -> float:
    if T <= 0:
        if option == "call":
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0
    d1, _ = _d1d2(S, K, T, r, sigma)
    if option == "call":
        return _norm_cdf(d1)
    if option == "put":
        return _norm_cdf(d1) - 1.0
    raise ValueError("option must be 'call' or 'put'")


def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0:
        return 0.0
    d1, _ = _d1d2(S, K, T, r, sigma)
    return _norm_pdf(d1) / (S * sigma * math.sqrt(T))


def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0:
        return 0.0
    d1, _ = _d1d2(S, K, T, r, sigma)
    return S * _norm_pdf(d1) * math.sqrt(T)


def theta(S: float, K: float, T: float, r: float, sigma: float, option: str = "call") -> float:
    """Theta per year (negative for long option)."""
    if T <= 0:
        return 0.0
    d1, d2 = _d1d2(S, K, T, r, sigma)
    term = -(S * _norm_pdf(d1) * sigma) / (2.0 * math.sqrt(T))
    if option == "call":
        return term - r * K * math.exp(-r * T) * _norm_cdf(d2)
    if option == "put":
        return term + r * K * math.exp(-r * T) * _norm_cdf(-d2)
    raise ValueError("option must be 'call' or 'put'")


def implied_vol(price: float, S: float, K: float, T: float, r: float,
                option: str = "call", lo: float = 1e-4, hi: float = 5.0,
                tol: float = 1e-6, max_iter: int = 100) -> float:
    """Bisection implied vol."""
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        p = bs_price(S, K, T, r, mid, option)
        if abs(p - price) < tol:
            return mid
        if p < price:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
