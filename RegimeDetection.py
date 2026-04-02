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
    w: float=0.5,
    divergence_threshold: float = 0.1,
    control: float = 0.5
):
    alpha, beta = initial_alpha, initial_beta


    # Add wealth tracking

    fixed_wealth = initial_wealth
    kelly_wealth = initial_wealth
    hybrid_wealth = initial_wealth
    hybrid_control_wealth = initial_wealth

    true_ps = []
    outcomes = []

    fixed_wealth_history = [fixed_wealth]
    kelly_wealth_history = [kelly_wealth]
    hybrid_wealth_history = [hybrid_wealth]
    hybrid_control_wealth_history = [hybrid_control_wealth]


    fixed_bets = []
    kelly_bets = []
    hybrid_bets = []
    hybrid_control_bets = []

    # Regime Change Detection

    regime_change_detections = []
    detection_time = None


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
            rolling_estimate = 0.5
        else:
            recent_outcomes = outcomes[-recent_window:]
            rolling_estimate = recent_outcomes.count("H") / len(recent_outcomes)
        recent_estimates.append(rolling_estimate)

        # Bayesian Estimate
        bayesian_estimate = expected_p(alpha, beta)


        # Hybrid Estimate
        hybrid_estimate = w * rolling_estimate + (1 - w) * bayesian_estimate


        # Regime Change Signal Environment

        divergence = abs(rolling_estimate - bayesian_estimate)

        # Decide bet sizes
        fixed_bet_fraction = fixed_fraction_size()
        kelly_bet_fraction = half_kelly_size(bayesian_estimate)
        rolling_bet_fraction = half_kelly_size(rolling_estimate)
        hybrid_bet_fraction = half_kelly_size(hybrid_estimate)
        hybrid_control_bet_fraction = hybrid_bet_fraction


        if len(outcomes) >= recent_window:
            if divergence > divergence_threshold:
                hybrid_control_bet_fraction *= control
                regime_change_detections.append(True)
                if detection_time is None:
                    detection_time = n
            else: regime_change_detections.append(False)
        else:
            regime_change_detections.append(False)

        fixed_bets.append(fixed_bet_fraction)
        kelly_bets.append(kelly_bet_fraction)
        rolling_bets.append(rolling_bet_fraction)
        hybrid_bets.append(hybrid_bet_fraction)
        hybrid_control_bets.append(hybrid_control_bet_fraction)



        # Generate outcome from true environment
        outcome = run_flip(true_p)


        # Update wealth
        fixed_wealth = update_wealth(fixed_wealth, fixed_bet_fraction, outcome)
        kelly_wealth = update_wealth(kelly_wealth, kelly_bet_fraction, outcome)
        rolling_wealth = update_wealth(rolling_wealth, rolling_bet_fraction, outcome)
        hybrid_wealth = update_wealth(hybrid_wealth, hybrid_bet_fraction, outcome)
        hybrid_control_wealth = update_wealth(hybrid_control_wealth, hybrid_control_bet_fraction, outcome)

        fixed_wealth_history.append(fixed_wealth)
        kelly_wealth_history.append(kelly_wealth)
        rolling_wealth_history.append(rolling_wealth)
        hybrid_wealth_history.append(hybrid_wealth)
        hybrid_control_wealth_history.append(hybrid_control_wealth)


        # Update belief after seeing outcome
        outcomes.append(outcome)
        alpha, beta = update_belief(alpha, beta, outcome)

        bayes_estimates.append(bayesian_estimate)
        hybrid_estimates.append(hybrid_estimate)
        

    return {
        "true_ps": true_ps,
        "bayes_estimates": bayes_estimates,
        "recent_estimates": recent_estimates,
        "hybrid_estimates": hybrid_estimates,
        "outcomes": outcomes,
        "fixed_wealth_history": fixed_wealth_history,
        "kelly_wealth_history": kelly_wealth_history,
        "rolling_wealth_history": rolling_wealth_history,
        "hybrid_wealth_history": hybrid_wealth_history,
        "hybrid_control_wealth_history": hybrid_control_wealth_history,
        "fixed_bets": fixed_bets,
        "kelly_bets": kelly_bets,
        "rolling_bets": rolling_bets,
        "hybrid_bets": hybrid_bets,
        "hybrid_control_bets": hybrid_control_bets,
        "regime_change_detections": regime_change_detections,
        "detection_time": detection_time
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
hybrid_wealth_history = single_results["hybrid_wealth_history"]
hybrid_bets = single_results["hybrid_bets"]
hybrid_control_wealth_history = single_results["hybrid_control_wealth_history"]
hybrid_control_bets = single_results["hybrid_control_bets"]
regime_change_detections = single_results["regime_change_detections"]
detection_time = single_results["detection_time"]

# Plot 1: Wealth paths
plt.figure(figsize=(8, 5))
plt.plot(fixed_wealth_history, label="Fixed sizing wealth")
plt.plot(kelly_wealth_history, label="Bayesian Half-Kelly wealth")
plt.plot(rolling_wealth_history, label="Rolling Half-Kelly wealth")
plt.plot(hybrid_wealth_history, label="Hybrid w/o control wealth")
plt.plot(hybrid_control_wealth_history, label="Hybrid w/ control wealth")
plt.axvline(50, color="black", linestyle="--", label="Regime change")
plt.xlabel("Step")
plt.ylabel("Wealth")
plt.title("Wealth Paths Under Regime Change")
plt.legend()
plt.savefig("figures/wealth_paths_under_regime_change.png")
plt.show()

# Plot 2: Bet fraction over time
plt.figure(figsize=(8, 5))
plt.plot(kelly_bets, label="Bayesian Half-Kelly bet fraction")
plt.plot(rolling_bets, label="Rolling Half-Kelly bet fraction")
plt.plot(hybrid_bets, label="Hybrid w/o control bet fraction")
plt.plot(hybrid_control_bets, label="Hybrid w/ control bet fraction")
plt.axvline(50, color = "black", linestyle="--", label="Regime change")
plt.xlabel("Step")
plt.ylabel("Bet fraction")
plt.title("Bet Fractions Over Time")
plt.savefig("figures/bet_fractions_over_time.png")
plt.legend()
plt.show()



# Basic summary stats
fixed_max_dd = compute_max_drawdown(fixed_wealth_history)
kelly_max_dd = compute_max_drawdown(kelly_wealth_history)
rolling_max_dd = compute_max_drawdown(rolling_wealth_history)
hybrid_max_dd = compute_max_drawdown(hybrid_wealth_history)
hybrid_control_max_dd = compute_max_drawdown(hybrid_control_wealth_history)

print("\n---Single Simulation Results Fixed vs Half-Kelly vs Rolling vs Hybrid w/ and w/o control---")
print("Final fixed wealth:", round(fixed_wealth_history[-1], 2))
print("Final half-Kelly wealth:", round(kelly_wealth_history[-1], 2))
print("Final rolling wealth:", round(rolling_wealth_history[-1], 2))
print("Final Hybrid without control wealth:", round(hybrid_wealth_history[-1], 2))
print("Final Hybrid with control wealth:", round(hybrid_control_wealth_history[-1], 2))


print("\nFixed max drawdown:", round(fixed_max_dd, 4))
print("Half-Kelly max drawdown:", round(kelly_max_dd, 4))
print("Rolling max drawdown:", round(rolling_max_dd, 4))
print("Combined max drawdown:", round(hybrid_max_dd, 4))
print("Combined with control max drawdown:", round(hybrid_control_max_dd, 4))


print("\nFixed wealth at step 50:", round(fixed_wealth_history[50], 2))
print("Rolling wealth at step 50:", round(rolling_wealth_history[50], 2))
print("Half-Kelly wealth at step 50:", round(kelly_wealth_history[50], 2))
print("Hybrid without control wealth at step 50:", round(hybrid_wealth_history[50], 2))
print("Hybrid with control wealth at step 50:", round(hybrid_control_wealth_history[50], 2))




# ---------------------------
# 8. Many simulation runs
# ---------------------------

def run_many_simulation(
    n_simulations: int=300,
    n_steps: int=200,
    initial_alpha: int=2,
    initial_beta: int=2,
    initial_wealth: float=100.0,
    w: float=0.5,
    divergence_threshold: float = 0.1,
    control: float=0.5
    ):

    fixed_final_wealth_path = []
    kelly_final_wealth_path = []
    rolling_final_wealth_path = []
    hybrid_final_wealth_path = []
    hybrid_control_final_wealth_path = []


    fixed_drawdown_path = []
    kelly_drawdown_path = []
    rolling_drawdown_path = []
    hybrid_drawdown_path = []
    hybrid_control_drawdown_path = []

    detection_times = []
    detection_flags = []

    
    for _ in range(n_simulations):
        results = run_single_simulation(
            n_steps = n_steps,
            initial_alpha = initial_alpha,
            initial_beta = initial_beta,
            initial_wealth = initial_wealth,
            w = w,
            divergence_threshold = divergence_threshold,
            control = control
            )


        fixed_wealth_history = results["fixed_wealth_history"]
        kelly_wealth_history = results["kelly_wealth_history"]
        rolling_wealth_history = results["rolling_wealth_history"]
        hybrid_wealth_history = results["hybrid_wealth_history"]
        hybrid_control_wealth_history = results["hybrid_control_wealth_history"]


        fixed_final_wealth_path.append(fixed_wealth_history[-1])
        kelly_final_wealth_path.append(kelly_wealth_history[-1])
        rolling_final_wealth_path.append(rolling_wealth_history[-1])
        hybrid_final_wealth_path.append(hybrid_wealth_history[-1])
        hybrid_control_final_wealth_path.append(hybrid_control_wealth_history[-1])
        
        
        fixed_drawdown_path.append(compute_max_drawdown(fixed_wealth_history))
        kelly_drawdown_path.append(compute_max_drawdown(kelly_wealth_history))
        rolling_drawdown_path.append(compute_max_drawdown(rolling_wealth_history))
        hybrid_drawdown_path.append(compute_max_drawdown(hybrid_wealth_history))
        hybrid_control_drawdown_path.append(compute_max_drawdown(hybrid_control_wealth_history))


        detection_time = results["detection_time"]
        if detection_time is None:
            detection_flags.append(False)
            detection_times.append(np.nan)
        else:
            detection_flags.append(True)
            detection_times.append(detection_time)


    return{
        "fixed_final_wealth_path": fixed_final_wealth_path,
        "kelly_final_wealth_path": kelly_final_wealth_path,
        "rolling_final_wealth_path": rolling_final_wealth_path,
        "hybrid_final_wealth_path": hybrid_final_wealth_path,
        "hybrid_control_final_wealth_path": hybrid_control_final_wealth_path,
        "fixed_drawdown": fixed_drawdown_path,
        "kelly_drawdown": kelly_drawdown_path,
        "rolling_drawdown": rolling_drawdown_path,
        "hybrid_drawdown": hybrid_drawdown_path,
        "hybrid_control_drawdown": hybrid_control_drawdown_path,
        "detection_rate": np.mean(detection_flags),
        "average_detection_time": np.nanmean(detection_times)
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
print("Average Hybrid without control wealth at final step: ", np.mean(many_results["hybrid_final_wealth_path"]))
print("Average Hybrid with control wealth at final step: ", np.mean(many_results["hybrid_control_final_wealth_path"]))



print("\nAverage Fixed max drawdown: ", np.mean(many_results["fixed_drawdown"]))
print("Average Half-Kelly max drawdown: ", np.mean(many_results["kelly_drawdown"]))
print("Average Rolling max drawdown: ", np.mean(many_results["rolling_drawdown"]))
print("Average Hybrid without control max drawdown: ", np.mean(many_results["hybrid_drawdown"]))
print("Average Hybrid with control max drawdown: ", np.mean(many_results["hybrid_control_drawdown"]))

print("\nDetection rate:", many_results["detection_rate"])
print("Average detection time:", many_results["average_detection_time"])


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
plt.savefig("figures/true_p_estimates.png")
plt.show()


# ---------------------------
# 11. Run many for varying control
# ---------------------------

for c in [0.25, 0.5, 0.75]:
    many_results = run_many_simulation(control = c)
    print(f"\nResults for control = {c}")
    print("Average Hybrid with control wealth at final step:",
          np.mean(many_results["hybrid_control_final_wealth_path"]))
    print("Average Hybrid with control max drawdown:",
          np.mean(many_results["hybrid_control_drawdown"]))
    print("Average Hybrid without control wealth at final step:",
          np.mean(many_results["hybrid_final_wealth_path"]))
    print("Average Hybrid without control max drawdown:",
          np.mean(many_results["hybrid_drawdown"]))
            
    
