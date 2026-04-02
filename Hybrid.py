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
    initial_wealth: float = 100.0,
    w: float=0.5
):
    alpha, beta = initial_alpha, initial_beta


    # Add wealth tracking

    fixed_wealth = initial_wealth
    kelly_wealth = initial_wealth
    combined_wealth = initial_wealth

    true_ps = []
    outcomes = []

    fixed_wealth_history = [fixed_wealth]
    kelly_wealth_history = [kelly_wealth]
    combined_wealth_history = [combined_wealth]


    fixed_bets = []
    kelly_bets = []
    combined_bets = []


    # We add the rolling wealth to compare Bayesian estimate vs Rolling estimate
    rolling_wealth = initial_wealth
    rolling_wealth_history = [initial_wealth]
    rolling_bets = []



    # Adaptation
    
    recent_window = 20

    bayes_estimates = []
    recent_estimates = []
    hybrid_estimates = []


    for n in range(n_steps):
        true_p = get_true_p(n)
        true_ps.append(true_p)

        # Rolling Estimate
        if len(outcomes) == 0:
            recent_p = 0.5
        else:
            recent_outcomes = outcomes[-recent_window:]
            recent_p = recent_outcomes.count("H") / len(recent_outcomes)
        recent_estimates.append(recent_p)

        # Bayesian Estimate
        p_hat = expected_p(alpha, beta)


        # Hybrid Estimate
        p_combined = w * recent_p + (1 - w) * p_hat


        # Decide bet sizes
        fixed_bet_fraction = fixed_fraction_size()
        kelly_bet_fraction = half_kelly_size(p_hat)
        rolling_bet_fraction = half_kelly_size(recent_p)
        combined_bet_fraction = half_kelly_size(p_combined)

        fixed_bets.append(fixed_bet_fraction)
        kelly_bets.append(kelly_bet_fraction)
        rolling_bets.append(rolling_bet_fraction)
        combined_bets.append(combined_bet_fraction)


        # Generate outcome from true environment
        outcome = run_flip(true_p)


        # Update wealth
        fixed_wealth = update_wealth(fixed_wealth, fixed_bet_fraction, outcome)
        kelly_wealth = update_wealth(kelly_wealth, kelly_bet_fraction, outcome)
        rolling_wealth = update_wealth(rolling_wealth, rolling_bet_fraction, outcome)
        combined_wealth = update_wealth(combined_wealth, combined_bet_fraction, outcome)

        fixed_wealth_history.append(fixed_wealth)
        kelly_wealth_history.append(kelly_wealth)
        rolling_wealth_history.append(rolling_wealth)
        combined_wealth_history.append(combined_wealth)


        # Update belief after seeing outcome
        outcomes.append(outcome)
        alpha, beta = update_belief(alpha, beta, outcome)

        bayes_estimates.append(p_hat)
        hybrid_estimates.append(p_combined)
        

    return {
        "true_ps": true_ps,
        "bayes_estimates": bayes_estimates,
        "recent_estimates": recent_estimates,
        "hybrid_estimates": hybrid_estimates,
        "outcomes": outcomes,
        "fixed_wealth_history": fixed_wealth_history,
        "kelly_wealth_history": kelly_wealth_history,
        "rolling_wealth_history": rolling_wealth_history,
        "combined_wealth_history": combined_wealth_history,
        "fixed_bets": fixed_bets,
        "kelly_bets": kelly_bets,
        "rolling_bets": rolling_bets,
        "combined_bets": combined_bets
    }


 

single_results = run_single_simulation()

true_ps = single_results["true_ps"]
bayes_estimates = single_results["bayes_estimates"]
recent_estimates = single_results["recent_estimates"]
hybrid_estimates = single_results["hybrid_estimates"]
outcomes = single_results["outcomes"]
fixed_wealth_history = single_results["fixed_wealth_history"]
fixed_bets = single_results["fixed_bets"]
kelly_wealth_history = single_results["kelly_wealth_history"]
kelly_bets = single_results["kelly_bets"]
rolling_wealth_history = single_results["rolling_wealth_history"]
rolling_bets = single_results["rolling_bets"]
combined_wealth_history = single_results["combined_wealth_history"]
combined_bets = single_results["combined_bets"]

# Plot 1: Wealth paths
plt.figure(figsize=(8, 5))
plt.plot(fixed_wealth_history, label="Fixed sizing wealth")
plt.plot(kelly_wealth_history, label="Bayesian Half-Kelly wealth")
plt.plot(rolling_wealth_history, label="Rolling Half-Kelly wealth")
plt.plot(combined_wealth_history, label="Hybrid wealth")
plt.axvline(50, color="black", linestyle="--", label="Regime change")
plt.xlabel("Step")
plt.ylabel("Wealth")
plt.title("Wealth Paths Under Regime Change")
plt.legend()
plt.show()

# Plot 2: Bet fraction over time
plt.figure(figsize=(8, 5))
plt.plot(kelly_bets, label="Bayesian Half-Kelly bet fraction")
plt.plot(rolling_bets, label="Rolling Half-Kelly bet fraction")
plt.plot(combined_bets, label="Hybrid bet fraction")
plt.axvline(50, color = "black", linestyle="--", label="Regime change")
plt.xlabel("Step")
plt.ylabel("Bet fraction")
plt.title("Bet Fractions Over Time")
plt.legend()
plt.show()



# Basic summary stats
fixed_max_dd = compute_max_drawdown(fixed_wealth_history)
kelly_max_dd = compute_max_drawdown(kelly_wealth_history)
rolling_max_dd = compute_max_drawdown(rolling_wealth_history)
hybrid_max_dd = compute_max_drawdown(combined_wealth_history)

print("\n---Single Simulation Results Fixed vs Half-Kelly vs Rolling vs Hybrid---")
print("Final fixed wealth:", round(fixed_wealth_history[-1], 2))
print("Final half-Kelly wealth:", round(kelly_wealth_history[-1], 2))
print("Final rolling wealth:", round(rolling_wealth_history[-1], 2))
print("Final Hybrid wealth:", round(combined_wealth_history[-1], 2))

print("\nFixed max drawdown:", round(fixed_max_dd, 4))
print("Half-Kelly max drawdown:", round(kelly_max_dd, 4))
print("Rolling max drawdown:", round(rolling_max_dd, 4))
print("Combined max drawdown:", round(hybrid_max_dd, 4))


print("\nFixed wealth at step 50:", round(fixed_wealth_history[50], 2))
print("Rolling wealth at step 50:", round(rolling_wealth_history[50], 2))
print("Half-Kelly wealth at step 50:", round(kelly_wealth_history[50], 2))
print("Hybrid wealth at step 50:", round(combined_wealth_history[50], 2))




# ---------------------------
# 8. Many simulation runs
# ---------------------------

def run_many_simulation(
    n_simulations: int=300,
    n_steps: int=200,
    initial_alpha: int=2,
    initial_beta: int=2,
    initial_wealth: float=100.0,
    w: float=0.5
    ):

    fixed_final_wealth_path = []
    kelly_final_wealth_path = []
    rolling_final_wealth_path = []
    hybrid_final_wealth_path = []

    

    fixed_drawdown_path = []
    kelly_drawdown_path = []
    rolling_drawdown_path = []
    hybrid_drawdown_path = []

    
    for _ in range(n_simulations):
        results = run_single_simulation(
            n_steps, initial_alpha, initial_beta, initial_wealth,w)


        fixed_wealth_history = results["fixed_wealth_history"]
        kelly_wealth_history = results["kelly_wealth_history"]
        rolling_wealth_history = results["rolling_wealth_history"]
        hybrid_wealth_history = results["combined_wealth_history"]


        fixed_final_wealth_path.append(fixed_wealth_history[-1])
        kelly_final_wealth_path.append(kelly_wealth_history[-1])
        rolling_final_wealth_path.append(rolling_wealth_history[-1])
        hybrid_final_wealth_path.append(hybrid_wealth_history[-1])
        
        fixed_drawdown_path.append(compute_max_drawdown(fixed_wealth_history))
        kelly_drawdown_path.append(compute_max_drawdown(kelly_wealth_history))
        rolling_drawdown_path.append(compute_max_drawdown(rolling_wealth_history))
        hybrid_drawdown_path.append(compute_max_drawdown(hybrid_wealth_history))


    return{
        "fixed_final_wealth_path": fixed_final_wealth_path,
        "kelly_final_wealth_path": kelly_final_wealth_path,
        "rolling_final_wealth_path": rolling_final_wealth_path,
        "hybrid_final_wealth_path": hybrid_final_wealth_path,
        "fixed_drawdown": fixed_drawdown_path,
        "kelly_drawdown": kelly_drawdown_path,
        "rolling_drawdown": rolling_drawdown_path,
        "hybrid_drawdown": hybrid_drawdown_path
    } 

# ---------------------------
# 9. Run many and inspect
# ---------------------------

many_results = run_many_simulation()

# Basic summary stats

print("\n---Many Simulation Results Fixed vs Half-Kelly vs Rolling vs Hybrid---")
print("Average Fixed wealth at final step: ", np.mean(many_results["fixed_final_wealth_path"]))
print("Average Half-Kelly wealth at final step: ", np.mean(many_results["kelly_final_wealth_path"]))
print("Average Rolling wealth at final step: ", np.mean(many_results["rolling_final_wealth_path"]))
print("Average Hybrid wealth at final step: ", np.mean(many_results["hybrid_final_wealth_path"]))


print("\nAverage Fixed max drawdown: ", np.mean(many_results["fixed_drawdown"]))
print("Average Half-Kelly max drawdown: ", np.mean(many_results["kelly_drawdown"]))
print("Average Rolling max drawdown: ", np.mean(many_results["rolling_drawdown"]))
print("Average Hybrid max drawdown: ", np.mean(many_results["hybrid_drawdown"]))

# ---------------------------
# 10. Adaptation (Bayesian vs Rolling Estimate) Plot
# ---------------------------

# Plot 3: True p estimates during regime change
plt.figure(figsize=(8,5))
plt.plot(bayes_estimates, label = "Bayesian")
plt.plot(recent_estimates, label = "Rolling")
plt.plot(hybrid_estimates, label = "Hybrid")
plt.plot(true_ps, label = "True p")
plt.axvline(50, color="black", linestyle = "--", label = "Regime Change")
plt.xlabel("Step")
plt.ylabel("Probability")
plt.title("Bayesian vs Rolling vs Hybrid Estimates of p")
plt.legend()
plt.show()


# ---------------------------
# 11. Run many for varying w
# ---------------------------
for w in [0.2, 0.5, 0.8]:
    many_results = run_many_simulation(w=w)
    print(f"\nResults for w = {w}")
    print("Average Hybrid wealth at final step:", np.mean(many_results["hybrid_final_wealth_path"]))
    print("Average Hybrid max drawdown:", np.mean(many_results["hybrid_drawdown"]))
