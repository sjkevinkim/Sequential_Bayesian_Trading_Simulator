# Sequential Bayesian Trading Strategy Simulator

## Overview

This project investigates how a trader should learn, size positions, and decide when to stop trading under uncertainty.

We simulate a simple trading environment (biased coin) and explore:
	•	Bayesian belief updating of edge
	•	Sequential stopping rules (Type I vs Type II error tradeoff)
	•	Position sizing strategies (Fixed vs Kelly-style sizing)

The goal is to understand how edge strength, confidence thresholds, and sizing rules interact to impact:
	•	Profitability
	•	Risk (drawdowns)
	•	Speed of detecting a true edge


---

## Methodology

### 1. Bayesian Learning Framework

We assume the true edge p is unknown and maintain a discrete prior over possible values:

ps = [0.5, 0.55, 0.60, 0.65]
prior = [0.4, 0.2, 0.2, 0.2]

Beliefs are updated after each trade using likelihood weighting:
	•	Heads → weight by p
	•	Tails → weight by 1 - p

### 2. Trading Decisions

Stopping Rule (Sequential Testing)
We stop trading if:
	•	Probability edge is “bad” exceeds threshold "alpha"

This captures:
	•	Type I error → stopping too early (missing real edge)
	•	Type II error → continuing in a bad strategy

Detection Rule
We detect a “good edge” when:
	•	Probability p > 0.5 exceeds threshold "beta"

This measures speed of learning / signal detection


### 3. Position Sizing Strategies

Fixed Sizing
	•	Discrete actions: No trade / Half size / Full size
	•	Based on posterior mean p_hat

Fractional Kelly-Style Sizing
	•	f = max(0, 2*p_hat - 1)
	•	Scales position size continuously with confidence
	•	More aggressive when edge is strong
	•	More conservative when uncertain


### 4. Simulation Setup

For each configuration:
	•	100 trades per run
	•	300–1000 Monte Carlo simulations
  •	Parameter sweep over:
	  •	alpha in {0.6, 0.7, 0.8, 0.9}
	  •	beta in {0.6, 0.7, 0.8, 0.9}
	  •	p in {0.50, 0.52, 0.55, 0.58, 0.60, 0.65}

Metrics tracked:
	•	Mean wealth
	•	Sharpe-like ratio
	•	Maximum drawdown
	•	Detection time
	•	Detection rate


---

## Results

### Parameter Sensitivity (D1)
Heatmaps show how performance varies across stopping thresholds:

### Sharpe Heatmap (true_p = 0.55)
![Sharpe heatmap](figures/sharpe_heatmap_p_0.55.png)

### Sharpe Heatmap (true_p = 0.60)
![Sharpe heatmap](figures/sharpe_heatmap_p_0.60.png)

### Sharpe Heatmap (true_p = 0.65)
![Sharpe heatmap](figures/sharpe_heatmap_p_0.65.png)

Insight:
	•	Conservative stopping (high \alpha) improves survivability
	•	Weak edges require looser stopping to avoid premature exit

### Detection Speed vs Edge (D2)

### Detection Time vs True Edge
![Detection time plot](figures/detection_time_vs_true_p.png)

### Detection Rate vs True Edge
![Detection rate plot](figures/detection_rate_vs_true_p.png)

Insights:
	•	Stronger edges → faster detection
	•	Weak edges → high noise → slow learning
	•	Detection reliability improves sharply as edge increases
  
### Kelly vs Fixed Sizing (D3)
We directly compare strategies:
Sharpe Difference = Kelly - Fixed
Drawdown Difference = Kelly - Fixed

![Kelly - Fixed Sharpe](figures/kelly_fixed_sharpe_0.50.png)
![Kelly - Fixed Sharpe](figures/kelly_fixed_sharpe_0.52.png)
![Kelly - Fixed Sharpe](figures/kelly_fixed_sharpe_0.55.png)
![Kelly - Fixed Sharpe](figures/kelly_fixed_sharpe_0.58.png)
![Kelly - Fixed Sharpe](figures/kelly_fixed_sharpe_0.60.png)
![Kelly - Fixed Sharpe](figures/kelly_fixed_sharpe_0.65.png)

![Kelly - Fixed Drawdown](figures/kelly_fixed_drawdown_0.50.png)
![Kelly - Fixed Drawdown](figures/kelly_fixed_drawdown_0.52.png)
![Kelly - Fixed Drawdown](figures/kelly_fixed_drawdown_0.55.png)
![Kelly - Fixed Drawdown](figures/kelly_fixed_drawdown_0.58.png)
![Kelly - Fixed Drawdown](figures/kelly_fixed_drawdown_0.60.png)
![Kelly - Fixed Drawdown](figures/kelly_fixed_drawdown_0.65.png)


Insights:
	•	Kelly outperforms when:
	•	Edge is strong
	•	Confidence builds quickly
	•	Fixed sizing is more robust when:
	•	Edge is weak
	•	Early variance dominates


---

## Key Insights

	•	Weak edges are dominated by noise:
    		Require long horizons and conservative stopping rules
	•	Stronger edges improve both Sharpe and detection speed:
			Allow more aggressive sizing and faster scaling
	•	Stopping rules encode risk preferences
	•	Low alpha: risk missing good strategies
	•	High alpha: risk staying in bad ones
	•	Kelly sizing adapts to confidence
	•	Exploits strong edges efficiently
	•	But increases sensitivity to estimation error
	•	Trade-off:
    	Early survival vs long-term optimal growth


---

## Tech Stack
- Python
- NumPy
- Pandas
- Matplotlib / Seaborn


---
