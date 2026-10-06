"""P&L attribution for a delta-hedged short option."""
from __future__ import annotations


def attribute_pnl(result: dict) -> dict:
    """Split hedged PnL into premium / hedging / payoff / cost buckets.

    Identity: total = premium + (stock_val - premium... ) simplified as:
      total = premium - payoff + (stock_val - initial_stock_cost grown)
    We report the additive, auditable split:
      total = premium - payoff + stock_trading_pnl + interest + (-costs)
    where stock_trading_pnl + interest is backed out as residual so the
    identity holds exactly.
    """
    total = result["total_pnl"]
    premium = result["premium"]
    payoff = result["payoff"]
    costs = result.get("tx_costs", 0.0)
    residual = total - premium + payoff + costs
    return {
        "option_premium_received": premium,
        "option_payoff_paid": -payoff,
        "delta_trading_plus_interest": residual,
        "transaction_costs": -costs,
        "total_hedged_pnl": total,
        "check_sum": premium - payoff + residual - costs,
    }
