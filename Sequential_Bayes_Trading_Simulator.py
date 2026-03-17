import random
import numpy as np
import pandas as pd

#We need to model a flip first

def run_flip(true_p):
    if random.random() < true_p: return "H"
    else: return "T"

def normalise_weights(weights: list[float]) -> list[float]:
    total = sum(weights)
    if total == 0:
        n = len(weights)
        return [1.0 / n] * n
    return [w / total for w in weights]

def update_belief(belief: list[float], outcome: str, ps: list[float]) -> list[float]:
    new_weights = []
    for w,p in zip(belief, ps):
        if outcome ==  "H":
            likelihood = p
        else: likelihood = 1-p
        new_weights.append(likelihood * w)
    return normalise_weights(new_weights)


def expected_p(belief: list[float], ps: list[float]) -> float:
    return sum(w * p for w,p in zip(belief, ps))


def decide_action(belief, ps, bad_edge_threshold, alpha):
    # Option 2: confidence-based "bad edge" probability
    prob_bad = sum(w for w, p in zip(belief, ps) if p <= bad_edge_threshold)

    # posterior mean p_hat
    p_hat = sum(w * p for w, p in zip(belief, ps))

    if prob_bad > alpha:
        return "stop"
    elif 0.5 <= p_hat < 0.6:
        return "half"
    else:  # p_hat >= 0.6
        return "full"

#when have sufficient evidence to conclude that edge exists
def detect(belief, ps, good_edge_threshold, beta):
    prob_good = sum(w for w,p in zip(belief,ps) if p > good_edge_threshold)
    return prob_good > beta

def run_simulation(
    num_flips: int,
    true_p: float,
    prior: list[float],
    ps: list[float],
    bad_edge_threshold: float,
    alpha: float,
    good_edge_threshold: float,
    beta: float
):

    belief = normalise_weights(prior[:])
    track_belief = [belief[:]]
    track_p_hat = [expected_p(belief, ps)]
    track_outcome = []
    wealth = 0
    wealth_path = [wealth]
    detection_time = None

    for n in range(num_flips):
        outcome = run_flip(true_p)
        track_outcome.append(outcome)
        
        belief = update_belief(belief, outcome, ps)
        p_hat = expected_p(belief, ps)
        
        track_belief.append(belief[:])
        track_p_hat.append(p_hat)

        action = decide_action(belief, ps, bad_edge_threshold, alpha)

        detection = detect(belief, ps, good_edge_threshold, beta)

        if action == "stop":
            return wealth, True, n+1, wealth_path, detection_time
        elif action == "half":
            if outcome == "H": wealth += 0.5
            else: wealth -= 0.5
        else:
            if outcome == "H": wealth += 1
            else: wealth -= 1

        wealth_path.append(wealth)

        if detection == True and detection_time is None:
            detection_time = n+1
        else: pass

    
    return wealth, False, num_flips, wealth_path, detection_time

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

        
def run_many_games(num_games, num_flips, true_p, prior, ps, bad_edge_threshold, alpha, good_edge_threshold, beta):
    track_wealth = []
    track_stopped = []
    track_when_stopped = []
    track_max_drawdown = []
    track_detection_time = []

    for i in range(num_games):
        wealth, stopped, num_trades, wealth_path, detection_time = run_simulation(num_flips, true_p, prior, ps, bad_edge_threshold, alpha, good_edge_threshold, beta)
        track_wealth.append(wealth)
        track_stopped.append(stopped)
        track_when_stopped.append(num_trades)
        max_drawdown = compute_max_drawdown(wealth_path)
        track_max_drawdown.append(max_drawdown)

        if detection_time is not None:
            track_detection_time.append(detection_time)
         
    
    avg_wealth = np.mean(track_wealth)
    stop_rate = np.mean(track_stopped)
    avg_num_trades = np.mean(track_when_stopped)
    std_wealth = np.std(track_wealth)
    q10, q50, q90 = np.quantile(track_wealth, [0.1, 0.5, 0.9])
    quantiles = [q10, q50, q90]
    average_drawdown = np.mean(track_max_drawdown)
    average_detection_time = np.mean(track_detection_time) if track_detection_time else None
    detection_rate = len(track_detection_time)/num_games

    return avg_wealth, stop_rate, avg_num_trades, std_wealth, quantiles, average_drawdown, average_detection_time, detection_rate

ps = [0.5, 0.55, 0.60, 0.65]
prior = [0.4, 0.2, 0.2, 0.2]

results = []

for p in [0.50, 0.55, 0.60, 0.65]:
    for a in [0.7, 0.8, 0.9]:
        for b in [0.7, 0.8, 0.9]:
            avg_wealth, stop_rate, avg_num_trades, std_wealth, quantiles, average_drawdown, average_detection_time, detection_rate = run_many_games(
                1000, 100, p, prior, ps, bad_edge_threshold = 0.5, alpha = a, good_edge_threshold = 0.5, beta = b
                )
            results.append({
                "true_p": p,
                "alpha": a,
                "beta": b,
                "mean wealth": avg_wealth,
                "standard deviation": std_wealth,
                "sharpe-like": avg_wealth/std_wealth if std_wealth>0 else float("nan"),
                "stop rate": stop_rate,
                "average drawdown": average_drawdown,
                "average detection time": average_detection_time,
                "detection rate": detection_rate
                })


df = pd.DataFrame(results)
df_055 = df[df["true_p"] == 0.55]

heatmap_055 = df_055.pivot(
    index = "alpha",
    columns = "beta",
    values = "sharpe-like"
    )


import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(heatmap_055, annot=True, fmt=".2f", cmap="magma")

plt.title("Sharpe-like vs Alpha/Beta (true_p = 0.55)")
plt.xlabel("beta")
plt.ylabel("alpha")

plt.show()







