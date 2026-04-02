# Adaptive Position Sizing and Regime Detection in Trading Systems

## Project Overview

This project investigates how a trader should learn and size positions to manage risk under uncertainty, particularly in environments where the underlying edge may change over time.

To study this in a controlled setting, I begin with a simulation framework based on repeated coin flips. Each flip represents a trade, where outcomes (“Heads” or “Tails”) correspond to profit and loss signals. The true probability of success is unknown and must be inferred from sequential observations.

Using **Monte Carlo simulation**, I run many independent paths to evaluate strategy performance in terms of:
    - wealth growth
    - drawdowns
    - robustness across different scenarios

We develop the project from a simple static idea into increasingly adaptive strategies.

The project initially utilises a static fixed-fraction strategy, where sizing does not respond to outcomes and further introduce fractional Kelly sizing strategy. I extend the project by utilising a continuous Beta distribution over discrete belief priors, a more realistic model for updating beliefs after each observation. 

To capture more realistic trading dynamics , I introduce a regime change, where the true probability shifts from a strong positive edge to a no-edge environment. This creates a non-stationary setting in which strategies must balance responsiveness and stability.

To address this, I develop:
    - Rolling-window estimator (fast but noisy)
    - Bayesian estimator (stable but slow)
    - Hybrid model that combines both approaches

Finally, I incorporate a regime detection mechanism, where divergence between models is used as a signal to reduce position sizes in order to limit drawdowns.

---

### Beta Distribution (D4)

In this framework, the unknown probability of success p is treated as a random variable:

p \sim \text{Beta}(\alpha, \beta)

where:
	•	\alpha represents evidence in favour of successful outcomes
	•	\beta represents evidence in favour of unsuccessful outcomes

Starting from a prior belief of Beta(2,2), the model updates sequentially after each observation:
	•	Heads \rightarrow \alpha + 1
	•	Tails \rightarrow \beta + 1

This gives a simple and interpretable Bayesian learning rule, where the posterior mean is:

\hat{p} = \frac{\alpha}{\alpha + \beta}

This posterior mean is then used to guide trading decisions and, later, position sizing.

### Regime Change (D5)

### Adaptation (D6)

### Key Insights


---

## Hybrid Models (D7)

### Key Insights

--

## Regime Detection & Control (D8)

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
