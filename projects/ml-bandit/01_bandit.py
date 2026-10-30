"""First reinforcement-learning experiment: explore versus exploit.

Three imaginary decisions have different average rewards. The learner sees
only the reward for the option it picked. All inputs are synthetic.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

TRUE_MEAN_REWARD = np.array([9.0, 13.0, 11.0])
N_DAYS = 250
N_RUNS = 100
EPSILON = 0.10


def choose_action(estimated_rewards, rng, epsilon=EPSILON):
    if rng.random() < epsilon:
        return int(rng.integers(3))  # explore
    best = np.flatnonzero(estimated_rewards == estimated_rewards.max())
    return int(rng.choice(best))  # exploit


def update_average(old_average, count, new_reward):
    return old_average + (new_reward - old_average) / count


def one_run(seed):
    rng = np.random.default_rng(seed)
    estimates = np.zeros(3)
    counts = np.zeros(3, dtype=int)
    learned_rewards, random_rewards, chosen_actions = [], [], []
    for _ in range(N_DAYS):
        today = rng.normal(TRUE_MEAN_REWARD, 3.0)
        learned_action = choose_action(estimates, rng)
        baseline_action = int(rng.integers(3))
        reward = float(today[learned_action])
        counts[learned_action] += 1
        estimates[learned_action] = update_average(
            estimates[learned_action], counts[learned_action], reward
        )
        learned_rewards.append(reward)
        random_rewards.append(float(today[baseline_action]))
        chosen_actions.append(learned_action)
    return np.array(learned_rewards), np.array(random_rewards), np.array(chosen_actions)


def main():
    runs = [one_run(seed) for seed in range(N_RUNS)]
    learned = np.array([x[0] for x in runs])
    random = np.array([x[1] for x in runs])
    actions = np.array([x[2] for x in runs])
    mean_learned = learned.mean(axis=0).cumsum()
    mean_random = random.mean(axis=0).cumsum()
    final_optimal = (actions[:, -100:] == TRUE_MEAN_REWARD.argmax()).mean()
    print(f"Simulated games: {N_RUNS}; days each: {N_DAYS}")
    print(f"Mean total reward, learning: {mean_learned[-1]:.2f}")
    print(f"Mean total reward, random:   {mean_random[-1]:.2f}")
    print(f"Best option selected during final 100 days: {final_optimal:.1%}")
    print("Synthetic example only; these are not real bike rental observations.")
    plt.plot(mean_learned, label="Epsilon-greedy (learns)")
    plt.plot(mean_random, label="Random choice")
    plt.title("Learning which option earns more: 100 simulated runs")
    plt.xlabel("Day")
    plt.ylabel("Average cumulative synthetic reward")
    plt.legend()
    plt.tight_layout()
    output = Path(__file__).resolve().parent / "output"
    output.mkdir(exist_ok=True)
    plt.savefig(output / "learning_vs_random.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()