# Adaptive Position Sizing and Regime Detection in Trading Systems

## Project Overview

This project investigates how a trader should learn and size positions to manage risk under uncertainty, particularly in environments where the underlying edge may change over time.

To study this in a controlled setting, I begin with a simulation framework based on repeated coin flips. Each flip represents a trade, where outcomes (“Heads” or “Tails”) correspond to profit and loss signals. The true probability of success is unknown and must be inferred from sequential observations.

Using **Monte Carlo simulation**, I run many independent paths to evaluate strategy performance in terms of:
    • wealth growth
    • drawdowns
    • robustness across different scenarios

We develop the project from a simple static idea into increasingly adaptive strategies.

The project initially utilises a static fixed-fraction strategy, where sizing does not respond to outcomes and further introduce fractional Kelly sizing strategy. I extend the project by utilising a continuous Beta distribution over discrete belief priors, a more realistic model for updating beliefs after each observation. 

To capture more realistic trading dynamics , I introduce a regime change, where the true probability shifts from a strong positive edge to a no-edge environment. This creates a non-stationary setting in which strategies must balance responsiveness and stability.

To address this, I develop:

    • Rolling-window estimator (fast but noisy)
    • Bayesian estimator (stable but slow)
    • Hybrid model that combines both approaches

Finally, I incorporate a regime detection mechanism, where divergence between models is used as a signal to reduce position sizes in order to limit drawdowns.

---

### Beta Distribution (D1)

To move from a discrete belief model to a more realistic continuous framework, I model the unknown edge p using a Beta distribution (p ~ B(alpha, beta)):

Starting from a prior of Beta(2,2), beliefs are updated sequentially after each outcome:

    • Heads increases alpha
    • Tails increases beta

The posterior mean (alpha/(alpha+beta)) provides an estimate of the underlying edge, while the full posterior captures the model's confidence.

For true_p = 0.8, edge is strong, hence the beta distribution quickly concentrates around the posterior mean.

![Beta Evolution](figres/beta_evolution.png)

This is useful because trading decisions should depend not only on the estimated edge, but also on how certain the model is that the edge is favourable. I use the posterior distribution to calculate:

    - the probability that the edge is bad (p <= 0.5)
    - the probability that the edge is good (p > 0.5)

These probabilities are then used for stopping and detection rules.

### Regime Change (D2)

### Adaptation (D3)

### Key Insights


---

## Hybrid Models (D4)

### Key Insights

--

## Regime Detection & Control (D5)

### Key Insights

-- 

## Limitations

--

## Tech Stack
- Python
- NumPy
- Pandas
- Matplotlib / Seaborn / Scipy


---
