# Sequential Bayesian Trading Strategy Simulator

## Overview
This project investigates how traders can evaluate and manage trading strategies with uncertain edge using Bayesian learning and sequential decision-making.

The simulation models:
- Bayesian belief updating
- Sequential stopping rules
- Edge detection thresholds
- Monte Carlo simulation of trading outcomes

---

## Key Questions
- How quickly can we detect a profitable trading strategy?
- How does edge strength affect detection speed?
- What is the trade-off between early stopping and long-run profitability?

---

## Methodology
- Simulated coin-flip trading environment with unknown edge
- Bayesian updating over discrete probability hypotheses
- Sequential decision rules:
  - Stop if high probability of bad edge
  - Trade with variable size depending on confidence
- Parameter sweeps over stopping and detection thresholds

---

## Results

### 1. Weak edges require large sample sizes
Weak edges are difficult to distinguish from noise and require many trades before confidence emerges.

### 2. Stronger edges improve detection and Sharpe
Higher edge leads to faster detection and better risk-adjusted performance.

### 3. Trade-off between Type I and Type II errors
Stopping rules must balance:
- exiting profitable strategies too early
- staying too long in unprofitable ones

---

## Visualisations
PLOTS of Heatmaps and Day 18 Visualisation

- Sharpe heatmaps (Day 17)
- Detection time vs edge (Day 18)
- Detection rate vs edge

---

## Key Insight
Detection difficulty is driven by signal-to-noise ratio, not just expected return.

---

## Tech Stack
- Python
- NumPy
- Pandas
- Matplotlib / Seaborn

---

## Future Work
- Continuous probability models
- Transaction costs
- Real market data backtesting