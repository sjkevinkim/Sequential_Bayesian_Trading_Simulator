import random
import numpy as np
from scipy.stats import beta as beta_dist
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------
# 1. Environment
# ---------------------------

def run_flip(true_p):
    if random.random() < true_p: return "H"
    else: return "T"


# ---------------------------
# 2. Belief update
# ---------------------------

def update_belief(alpha, beta, outcome: str):
    if outcome ==  "H":
        alpha += 1
    else: beta += 1
    return alpha, beta


def expected_p(alpha: int, beta: int) -> float:
    return alpha/(alpha+beta)


# ---------------------------
# 3. Decision Detection rules
# ---------------------------


def decide_action(alpha, beta, bad_threshold):
    prob_bad = beta_dist.cdf(0.5, alpha, beta)
    p_hat = expected_p(alpha, beta)
    
    if prob_bad > bad_threshold:
        return "stop"
    elif 0.5 <= p_hat < 0.6:
        return "half"
    else:  # p_hat >= 0.6
        return "full"
    
def detect(alpha, beta, good_threshold):
    prob_good = 1 - beta_dist.cdf(0.5, alpha, beta)
    return prob_good > good_threshold


# ---------------------------
# 4. One simulation run (Fixed)
# ---------------------------


def run_fixed_simulation(
    num_flips: int,
    true_p: float,
    alpha: int,
    beta: int,
    bad_threshold: float,
    good_threshold: float
    ):

    track_p_hat = [expected_p(alpha, beta)]
    track_outcome = []
    wealth = 0
    wealth_path = [wealth]
    detection_time = None

    for n in range(num_flips):
        outcome = run_flip(true_p)
        track_outcome.append(outcome)
        alpha, beta = update_belief(alpha, beta, outcome)
        
        p_hat = expected_p(alpha, beta)
        track_p_hat.append(p_hat)

        action = decide_action(alpha, beta, bad_threshold)

        detection = detect(alpha, beta, good_threshold)

        if action == "stop":
            return wealth, True, n+1, wealth_path, detection_time
        elif action == "half":
            if outcome == "H": wealth += 0.5
            else: wealth -= 0.5
        else:
            if outcome == "H": wealth += 1
            else: wealth -= 1

        wealth_path.append(wealth)

        if detection and detection_time is None:
            detection_time = n+1
        else: pass

    
    return wealth, False, num_flips, wealth_path, detection_time


# ---------------------------
# 5. One simulation run (Kelly)
# ---------------------------


def run_kelly_simulation(
    num_flips: int,
    true_p: float,
    alpha: int,
    beta: int,
    bad_threshold: float,
    good_threshold: float
    ):

    track_p_hat = [expected_p(alpha, beta)]
    track_outcome = []
    wealth = 0
    wealth_path = [wealth]
    detection_time = None

    for n in range(num_flips):
        outcome = run_flip(true_p)
        track_outcome.append(outcome)
        
        alpha, beta = update_belief(alpha, beta, outcome)
        p_hat = expected_p(alpha, beta)
        track_p_hat.append(p_hat)
        size = max(0.0, 0.5 * (2*p_hat - 1))

        action = decide_action(alpha, beta, bad_threshold)

        detection = detect(alpha, beta, good_threshold)

        if action == "stop":
            return wealth, True, n+1, wealth_path, detection_time
        else:
            if outcome == "H": wealth += size
            else: wealth -= size

        wealth_path.append(wealth)

        if detection and detection_time is None:
            detection_time = n+1

    
    return wealth, False, num_flips, wealth_path, detection_time


# ---------------------------
# 6. Drawdown helper
# ---------------------------


def compute_max_drawdown(wealth_path):
    peak = wealth_path[0]
    max_drawdown = 0
    for wealth in wealth_path:
        if wealth > peak:
            peak = wealth
        drawdown = peak - wealth
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    return max_drawdown


# ---------------------------
# 7. Many simulation runs
# ---------------------------

        
def run_many_games(sim_fn, num_games, num_flips, true_p, alpha, beta, bad_threshold, good_threshold):
    track_wealth = []
    track_stopped = []
    track_when_stopped = []
    track_max_drawdown = []
    track_detection_time = []

    for i in range(num_games):
        wealth, stopped, num_trades, wealth_path, detection_time = sim_fn(num_flips, true_p, alpha, beta, bad_threshold, good_threshold)
        track_wealth.append(wealth)
        track_stopped.append(stopped)
        track_when_stopped.append(num_trades)
        max_drawdown = compute_max_drawdown(wealth_path)
        track_max_drawdown.append(max_drawdown)

        if detection_time is None:
            track_detection_time.append(np.nan)
        else: track_detection_time.append(detection_time)
        
         
    
    avg_wealth = np.mean(track_wealth)
    stop_rate = np.mean(track_stopped)
    avg_num_trades = np.mean(track_when_stopped)
    std_wealth = np.std(track_wealth)
    q10, q50, q90 = np.quantile(track_wealth, [0.1, 0.5, 0.9])
    quantiles = [q10, q50, q90]
    average_drawdown = np.mean(track_max_drawdown)
    average_detection_time = np.nanmean(track_detection_time) if track_detection_time else None
    detection_rate = np.mean([not np.isnan(x) for x in track_detection_time])

    return avg_wealth, stop_rate, avg_num_trades, std_wealth, quantiles, average_drawdown, average_detection_time, detection_rate


# ---------------------------
# 8. Run many and inspect
# ---------------------------


true_ps = [0.5, 0.55, 0.60, 0.65]

kelly_results = []
fixed_results = []

# Fixed Sizing Strategy 
for p in true_ps:
    avg_wealth, stop_rate, avg_num_trades, std_wealth, quantiles, average_drawdown, average_detection_time, detection_rate = run_many_games(
        run_fixed_simulation, 300, 100, p, 2, 2, 0.8, 0.8)
    fixed_results.append({
        "strategy": "fixed",
        "true_p": p,
        "avg wealth": avg_wealth,
        "drawdown": average_drawdown,
        "detection time": average_detection_time,
        "detection rate": detection_rate
        })


# Half-Kelly Sizing Strategy 
for p in true_ps:
    avg_wealth, stop_rate, avg_num_trades, std_wealth, quantiles, average_drawdown, average_detection_time, detection_rate = run_many_games(
        run_kelly_simulation, 300, 100, p, 2, 2, 0.8, 0.8)
    kelly_results.append({
        "strategy": "kelly",
        "true_p": p,
        "avg wealth": avg_wealth,
        "drawdown": average_drawdown,
        "detection time": average_detection_time,
        "detection rate": detection_rate
        })

kelly_df = pd.DataFrame(kelly_results)
fixed_df = pd.DataFrame(fixed_results)


# ---------------------------
# 9. PDF plotting function
# ---------------------------

def plot_beta_evolution_grid(snapshots, filename="figures/beta_evolution.png"):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    axes = axes.flatten()

    for ax, (n, (alpha, beta)) in zip(axes, snapshots):
        x = np.linspace(0, 1, 200)
        y = beta_dist.pdf(x, alpha, beta)

        ax.plot(x, y)
        ax.set_title(f"Step {n}")
        ax.set_xlabel("p")
        ax.set_ylabel("Density")

    for ax in axes[len(snapshots):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


# ---------------------------
# 10. Run one and inspect
# ---------------------------

snapshots = []
alpha, beta = 2, 2
true_p = 0.80

for n in range(100):
    outcome = run_flip(true_p)
    alpha, beta = update_belief(alpha, beta, outcome)

    if n in [0, 5, 10, 20, 50]:
        snapshots.append((n, (alpha, beta)))

# Save one combined figure
plot_beta_evolution_grid(snapshots, filename="figures/beta_evolution.png")

