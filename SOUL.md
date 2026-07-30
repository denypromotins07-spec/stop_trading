# SOUL.md - Self-Learning Ledger for HFT Crypto Bot
# This file acts as the bot's persistent memory to evolve its intelligence after every trading session.

## Trade Mistakes
- [ ] Record liquidation events and their causes
- [ ] Track slippage anomalies during high volatility
- [ ] Log failed arbitrage opportunities due to latency
- [ ] Document order book spoofing detection failures

## Adaptive Weights
- momentum_weight: 0.35
- mean_reversion_weight: 0.25
- volume_profile_weight: 0.20
- order_flow_imbalance_weight: 0.20
- last_updated: "PENDING_FIRST_SESSION"

## Regime Memories
### Bull Market Signatures
- High volume breakouts with low slippage
- Sustained bid wall support
- Positive funding rates

### Bear Market Signatures
- Cascading stop-loss triggers
- Liquidity evaporation on bids
- Negative funding rate persistence

### Choppy/Range-Bound Signatures
- Mean-reverting price action
- Balanced order book depth
- Low realized volatility

## Session History
| Date | PnL | Sharpe | Max Drawdown | Regime Detected | Key Lesson |
|------|-----|--------|--------------|-----------------|------------|
| INIT | 0.0 | 0.0    | 0.0          | UNKNOWN         | System initialized |

## Evolution Notes
- v0.1.0: Initial infrastructure deployment
- Pending: First live trading session data ingestion
