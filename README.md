# RL Course — V1

Tabular reinforcement learning implementations applied to **FrozenLake-v1** (Gymnasium).

---

## Repository Structure (cookiecutter data-science layout)

```
rl_course_v1/
├── README.md
├── requirements.txt          # pip dependencies
├── environment.yml           # conda environment
├── setup.py                  # editable install
├── train.py                  # CLI entry point (train / evaluate)
│
├── models/
│   └── best.pkl              # optimal policy kernel + V* (Policy Iteration, γ=0.99)
│
├── data/
│   ├── raw/                  # (ignored by git)
│   └── processed/            # (ignored by git)
│
├── reports/
│   └── figures/              # generated plots (ignored by git)
│
├── notebooks/                # exploratory notebooks
│
└── src/rl_course_v1/
    ├── mdp/
    │   ├── base.py           # TabularMDP  (S, A, P, R, γ)
    │   ├── policy.py         # Policy, DeterministicPolicy, ValueFunction, QValueFunction
    │   └── subtasks.py       # Subtask definitions for FrozenLake
    │
    ├── algorithms/
    │   ├── dp.py             # Policy Evaluation/Iteration, Value Iteration (V & Q)
    │   ├── monte_carlo.py    # MC prediction + MC control (ES, ε-greedy)
    │   ├── td.py             # TD(0), n-step TD, TD(λ)  [state-value based]
    │   ├── sarsa.py          # Sarsa(0), n-step Sarsa, Sarsa(λ)
    │   └── q_learning.py     # Q-learning (off-policy)
    │
    ├── agents/
    │   └── agent.py          # TabularAgent + AgentConfig (train / eval / save / load)
    │
    └── exploration/
        └── strategies.py     # ε-greedy (decaying), UCB1, Boltzmann, Count-bonus
```

---

## Setup

```bash
# Conda (recommended)
conda env create -f environment.yml
conda activate rl_course_v1

# or pip
pip install -e .
```

---

## MDP Representation

The environment is modelled as a finite MDP $(S, A, P, R, \gamma)$:

| Component | Implementation | Description |
|-----------|---------------|-------------|
| **States** $S$ | `TabularMDP.n_states` | Discrete flat indices (16 for 4×4 FrozenLake) |
| **Actions** $A$ | `TabularMDP.n_actions` | Left/Down/Right/Up (4) |
| **Transitions** $P(s'\|s,a)$ | `TabularMDP.P_matrix` — shape `(S,A,S)` | Built from `env.unwrapped.P` |
| **Rewards** $R(s,a)$ | `TabularMDP.R_matrix` — shape `(S,A)` | Expected reward $\sum_{s'} P(s'\|s,a)\,r$ |
| **Discount** $\gamma$ | `TabularMDP.gamma` | Default 0.99 |
| **Policy** $\pi(a\|s)$ | `Policy.probs` — shape `(S,A)` | Stochastic; deterministic as special case |
| **Subtasks** | `Subtask` | Named state-space slices + sub-goals |

The agent must eventually bootstrap its own $(P, R)$ estimates from experience —
model-based RL / world-model learning — using the `TabularMDP` matrices as
the ground truth to compare against.

---

## Algorithms Implemented

### Dynamic Programming (model-based, exact)

| Function | File | Description |
|----------|------|-------------|
| `policy_evaluation` | `dp.py` | Iterative Bellman expectation → $V^\pi$ |
| `q_policy_evaluation` | `dp.py` | Iterative Bellman expectation → $Q^\pi$ |
| `policy_improvement_v` | `dp.py` | One-step greedy improvement on $V$ |
| `policy_improvement_q` | `dp.py` | One-step greedy improvement on $Q$ |
| `policy_iteration` | `dp.py` | PE + PI loop → $\pi^*$, $V^*$ |
| `q_policy_iteration` | `dp.py` | Q-based PI loop → $\pi^*$, $Q^*$ |
| `value_iteration` | `dp.py` | Bellman optimality operator → $V^*$ |

### Monte Carlo (model-free, episodic)

| Function | File | Description |
|----------|------|-------------|
| `mc_prediction_first_visit` | `monte_carlo.py` | First-visit MC → $V^\pi$ |
| `mc_prediction_every_visit` | `monte_carlo.py` | Every-visit MC → $V^\pi$ |
| `mc_control_epsilon_greedy` | `monte_carlo.py` | On-policy MC control → $\pi^*$ |
| `mc_control_exploring_starts` | `monte_carlo.py` | MC-ES control → $\pi^*$ |

### Temporal Difference (model-free, online)

| Function | File | Description |
|----------|------|-------------|
| `td0_prediction` | `td.py` | TD(0) → $V^\pi$ |
| `n_step_td_prediction` | `td.py` | Forward-view $n$-step TD → $V^\pi$ |
| `td_lambda_prediction` | `td.py` | Backward-view TD(λ) via eligibility traces → $V^\pi$ |
| `n_step_td_control` | `td.py` | Forward-view $n$-step TD + greedy → $\pi^*$ |
| `td_lambda_control` | `td.py` | Backward-view TD(λ) + greedy → $\pi^*$ |
| `sarsa_zero` | `sarsa.py` | One-step on-policy Sarsa → $Q^\pi$ |
| `n_step_sarsa` | `sarsa.py` | Forward-view $n$-step Sarsa → $Q^*$ |
| `sarsa_lambda` | `sarsa.py` | Backward-view Sarsa(λ) via eligibility traces → $Q^*$ |
| `q_learning` | `q_learning.py` | Off-policy Q-learning → $Q^*$ |

### Exploration

| Class | Description |
|-------|-------------|
| `EpsilonGreedy` | ε-greedy with linear decay |
| `UCB1` | Upper Confidence Bound (tabular) |
| `BoltzmannExplorer` | Softmax / Boltzmann |
| `CountBonus` | Intrinsic curiosity via $\beta/\sqrt{N(s,a)}$ |

---

## Usage

```bash
# Policy Iteration (exact DP, best for FrozenLake):
python train.py --algorithm policy_iteration --save

# Q-learning:
python train.py --algorithm q_learning --n_episodes 20000 --save

# Sarsa(λ):
python train.py --algorithm sarsa_lambda --lam 0.8 --alpha 0.05 --n_episodes 30000 --save

# Evaluate saved policy:
python train.py --eval_only --checkpoint models/best.pkl
```

### Programmatic API

```python
import gymnasium as gym
from rl_course_v1.agents import TabularAgent, AgentConfig

env    = gym.make("FrozenLake-v1", is_slippery=True)
config = AgentConfig(algorithm="q_learning", n_episodes=20_000, gamma=0.99)
agent  = TabularAgent(config)
agent.train(env)
agent.evaluate(env, n_episodes=500)
agent.save("models/my_agent.pkl")
```

---

## Best Policy (V1 Submission)

Trained via **exact Policy Iteration** on the full MDP model (γ = 0.99, stochastic FrozenLake):

```
Optimal policy (← Down → ↑):
  <  ^  ^  ^
  <  H  <  H
  ^  v  <  H
  H  >  v  H

V* (4×4, rounded):
 0.542  0.499  0.471  0.457
 0.559  0.000  0.358  0.000
 0.592  0.643  0.615  0.000
 0.000  0.742  0.863  0.000
```

Saved to `models/best.pkl` (cloudpickle).  Load with:

```python
from rl_course_v1.agents import TabularAgent
agent = TabularAgent.load("models/best.pkl")
```

---

## Citations & References

1. **Sutton, R. S., & Barto, A. G.** (2018). *Reinforcement Learning: An Introduction* (2nd ed.).
   MIT Press. http://incompleteideas.net/book/the-book.html
   — Primary textbook. All algorithms reference chapter numbers inline.

2. **Watkins, C. J. C. H., & Dayan, P.** (1992). Q-learning.
   *Machine Learning*, 8(3–4), 279–292.
   https://doi.org/10.1007/BF00992698

3. **Gymnasium** (formerly OpenAI Gym). Farama Foundation.
   https://gymnasium.farama.org/

4. **NumPy** (Harris et al., 2020). *Nature*, 585, 357–362.
   https://doi.org/10.1038/s41586-020-2649-2

---

## Collaborations

*(List any collaborators or external resources used here per syllabus requirements.)*

---

## License

MIT
