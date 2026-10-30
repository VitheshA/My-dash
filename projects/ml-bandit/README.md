# ML Starter: Three-Choice Learning Experiment

This is a tiny reinforcement-learning **bandit** experiment designed to explain learning under uncertainty. It is conceptually relevant to [ASU's Data Mining and Reinforcement Learning group](https://labs.engineering.asu.edu/hw/), but is **not affiliated with or evaluated by that lab**.

## Run

```bash
pip install -r requirements.txt
python 01_bandit.py
python -m pytest test_bandit.py -q
```

The program compares two strategies for an imaginary problem where three choices have different hidden average rewards. The learner only sees the reward of its selected choice.

- **Random strategy** picks a choice without learning.
- **Epsilon-greedy learner** picks what has earned the highest average reward so far, except it explores at random 10% of the time.
- We compare mean cumulative reward across **100 simulations of 250 days each**, with independently seeded randomness.

## Five terms

| Term | Meaning |
|---|---|
| Action | One of three available choices |
| Reward | What the environment gives after a choice |
| Explore | Try a random option to learn |
| Exploit | Choose the option estimated to be best |
| Regret | Reward missed relative to a reference policy |

## Boundaries

The reward model is **synthetic**, not a real BikeSmart or city dataset. A multi-armed bandit has no state transitions, so this is simpler than full reinforcement learning. No Kaggle score, professor review or research result is claimed.

The code was AI-assisted. Understand it and reproduce the figures yourself before citing it as an independent project.

**How to explain it:** “I simulated three decisions whose payoffs were uncertain. My epsilon-greedy learner remembered average rewards, selected the best-known option most days, and explored randomly 10% of the time. I compared its cumulative rewards against random choice across multiple simulations.”