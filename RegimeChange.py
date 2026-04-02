import random
import numpy as np
from scipy.stats import beta as beta_dist
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------
# 1. Environment
# ---------------------------

def get_true_p(n: int) -> float:
    """True probability changes at step 50."""
    if n < 50:
        return 0.8
    else:
        return 0.45


def run_flip(true_p: float) -> str:
    """Simulate one coin flip."""
    if random.random() < true_p:
        return "H"
    else:
        return "T"


# ---------------------------
# 2. Bayesian belief update
# ---------------------------

def update_belief(alpha: int, beta: int, outcome: str) -> tuple[int, int]:
    """Update Beta posterior after observing outcome."""
    if outcome == "H":
        alpha += 1
    else:
        beta += 1
    return alpha, beta


def expected_p(alpha: int, beta: int) -> float:
    """Posterior mean of Beta(alpha, beta)."""
    return alpha / (alpha + beta)


# ---------------------------
# 3. Position sizing rules
# ---------------------------

def fixed_fraction_size() -> float:
    """Fixed bet fraction each round."""
    return 0.05


def half_kelly_size(p_hat: float) -> float:
    """
    Half-Kelly for an even-money bet.
    Full Kelly for even odds is: f = 2p - 1
    Half Kelly is: 0.5 * (2p - 1)

    We clip at 0 because if p_hat <= 0.5, there is no positive edge.
    """
    full_kelly = 2 * p_hat - 1
    half_kelly = 0.5 * full_kelly
    return max(0.0, half_kelly)


# ---------------------------
# 4. Wealth update
# ---------------------------

def update_wealth(wealth: float, bet_fraction: float, outcome: str) -> float:
    """
    Even-money payoff:
    - win: wealth increases by bet_fraction * wealth
    - lose: wealth decreases by bet_fraction * wealth
    """
    bet_size = wealth * bet_fraction

    if outcome == "H":
        wealth += bet_size
    else:
        wealth -= bet_size

    return wealth


# ---------------------------
# 5. Drawdown helper
# ---------------------------

def compute_max_drawdown(wealth_history: list[float]) -> float:
    peak = wealth_history[0]
    max_drawdown = 0.0

    for wealth in wealth_history:
        if wealth > peak:
            peak = wealth

        drawdown = (peak - wealth) / peak
        max_drawdown = max(max_drawdown, drawdown)

    return max_drawdown


# ---------------------------
# 6. One simulation run
# ---------------------------

def run_single_simulation(
    n_steps: int = 200,
    initial_alpha: int = 2,
    initial_beta: int = 2,
    initial_wealth: float = 100.0
):
    alpha, beta = initial_alpha, initial_beta

    fixed_wealth = initial_wealth
    kelly_wealth = initial_wealth

    true_ps = []
    posterior_means = []
    outcomes = []

    fixed_wealth_history = [fixed_wealth]
    kelly_wealth_history = [kelly_wealth]

    fixed_bets = []
    kelly_bets = []

    for n in range(n_steps):
        true_p = get_true_p(n)
        true_ps.append(true_p)

        # Use current belief before seeing new outcome
        p_hat = expected_p(alpha, beta)
        posterior_means.append(p_hat)

        # Decide bet sizes
        fixed_bet_fraction = fixed_fraction_size()
        kelly_bet_fraction = half_kelly_size(p_hat)

        fixed_bets.append(fixed_bet_fraction)
        kelly_bets.append(kelly_bet_fraction)

        # Generate outcome from true environment
        outcome = run_flip(true_p)
        outcomes.append(outcome)

        # Update wealth
        fixed_wealth = update_wealth(fixed_wealth, fixed_bet_fraction, outcome)
        kelly_wealth = update_wealth(kelly_wealth, kelly_bet_fraction, outcome)

        fixed_wealth_history.append(fixed_wealth)
        kelly_wealth_history.append(kelly_wealth)

        # Update belief after seeing outcome
        alpha, beta = update_belief(alpha, beta, outcome)

    return {
        "true_ps": true_ps,
        "posterior_means": posterior_means,
        "outcomes": outcomes,
        "fixed_wealth_history": fixed_wealth_history,
        "kelly_wealth_history": kelly_wealth_history,
        "fixed_bets": fixed_bets,
        "kelly_bets": kelly_bets,
    }


# ---------------------------
# 7. Run once and inspect
# ---------------------------

single_results = run_single_simulation()

true_ps = single_results["true_ps"]
posterior_means = single_results["posterior_means"]
fixed_wealth_history = single_results["fixed_wealth_history"]
kelly_wealth_history = single_results["kelly_wealth_history"]
kelly_bets = single_results["kelly_bets"]

# Plot 1: True p vs posterior mean
plt.figure(figsize=(8, 5))
plt.plot(true_ps, label="True p")
plt.plot(posterior_means, label="Posterior mean")
plt.axvline(50, linestyle="--", label="Regime change")
plt.xlabel("Step")
plt.ylabel("Probability")
plt.title("True p vs Posterior Mean")
plt.legend()
plt.savefig("figures/true_p_vs_posterior_mean_d2.png")
plt.show()

# Plot 2: Wealth paths
plt.figure(figsize=(8, 5))
plt.plot(fixed_wealth_history, label="Fixed sizing wealth")
plt.plot(kelly_wealth_history, label="Half-Kelly wealth")
plt.axvline(50, linestyle="--", label="Regime change")
plt.xlabel("Step")
plt.ylabel("Wealth")
plt.title("Wealth Paths Under Regime Change")
plt.legend()
plt.savefig("figures/wealth_paths_under_regime_change_d2.png")
plt.show()

# Plot 3: Kelly bet fraction over time
plt.figure(figsize=(8, 5))
plt.plot(kelly_bets, label="Half-Kelly bet fraction")
plt.axvline(50, linestyle="--", label="Regime change")
plt.xlabel("Step")
plt.ylabel("Bet fraction")
plt.title("Half-Kelly Bet Size Over Time")
plt.legend()
plt.savefig("figures/bet_size_d2.png")
plt.show()

# Basic summary stats
fixed_max_dd = compute_max_drawdown(fixed_wealth_history)
kelly_max_dd = compute_max_drawdown(kelly_wealth_history)

print("\n---Single Simulation Results---")
print("Final fixed wealth:", round(fixed_wealth_history[-1], 2))
print("Final half-Kelly wealth:", round(kelly_wealth_history[-1], 2))
print("Fixed max drawdown:", round(fixed_max_dd, 4))
print("Half-Kelly max drawdown:", round(kelly_max_dd, 4))

print("Fixed wealth at step 50:", round(fixed_wealth_history[50], 2))
print("Fixed wealth at final step:", round(fixed_wealth_history[-1], 2))
print("Half-Kelly wealth at step 50:", round(kelly_wealth_history[50], 2))
print("Half-Kelly wealth at final step:", round(kelly_wealth_history[-1], 2))


# ---------------------------
# 8. Many simulation runs
# ---------------------------

def run_many_simulation(
    n_simulations: int=300,
    n_steps: int=200,
    initial_alpha: int=2,
    initial_beta: int=2,
    initial_wealth: float=100.0
    ):

    fixed_final_wealth_path = []
    kelly_final_wealth_path = []

    fixed_drawdown_path = []
    kelly_drawdown_path = []
    
    for _ in range(n_simulations):
        results = run_single_simulation(
            n_steps, initial_alpha, initial_beta, initial_wealth)


        fixed_wealth_history = results["fixed_wealth_history"]
        kelly_wealth_history = results["kelly_wealth_history"]

        fixed_final_wealth_path.append(fixed_wealth_history[-1])
        kelly_final_wealth_path.append(kelly_wealth_history[-1])
        
        fixed_drawdown_path.append(compute_max_drawdown(fixed_wealth_history))
        kelly_drawdown_path.append(compute_max_drawdown(kelly_wealth_history))

    return{
        "fixed_final_wealth_path": fixed_final_wealth_path,
        "kelly_final_wealth_path": kelly_final_wealth_path,
        "fixed_drawdown": fixed_drawdown_path,
        "kelly_drawdown": kelly_drawdown_path,
    } 

# ---------------------------
# 9. Run many and inspect
# ---------------------------

many_results = run_many_simulation()

# Basic summary stats

print("\n---Many Simulation Results---")
print("Average Fixed wealth at final step: ", np.mean(many_results["fixed_final_wealth_path"]))
print("Average Half-Kelly wealth at final step: ", np.mean(many_results["kelly_final_wealth_path"]))
print("Average Fixed max drawdown: ", np.mean(many_results["fixed_drawdown"]))
print("Average Half-Kelly max drawdown: ", np.mean(many_results["kelly_drawdown"]))


    
