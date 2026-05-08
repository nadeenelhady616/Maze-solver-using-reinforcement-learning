# Maze RL Project — README & Implementation Roadmap

## What's Ready Right Now ✅

```
maze_rl_project/
├── maze_env/
│   ├── __init__.py       ✅  Gymnasium registration (7 env variants)
│   └── maze_env.py       ✅  Full MazeEnv (Gymnasium API)
├── utils/
│   ├── __init__.py
│   └── oracle.py         ✅  BFS oracle + distance map
├── algorithms/
│   ├── q_learning.py     🔲  Stub with full spec
│   ├── dqn.py            🔲  Stub with full spec
│   └── policy_gradient.py 🔲 Stub with full spec
├── configs/
│   ├── q_learning.yaml   ✅  Ready hyper-params
│   └── dqn.yaml          ✅  Ready hyper-params
├── results/
│   ├── logs/             (auto-filled by training scripts)
│   ├── plots/            ✅  maze_generators.png saved
│   └── models/           (auto-filled by save calls)
├── tests/                (add pytest unit tests here)
├── notebooks/            (add Jupyter exploration notebooks here)
├── requirements.txt      ✅
└── test_env.py           ✅  8/8 tests passing
```

---

## Environment Features

| Feature | Value |
|---|---|
| Gymnasium API | ✅ `reset(seed=)`, `step()`, `render()`, `close()` |
| Action space | Discrete(4) — N/E/S/W |
| Maze generators | `dfs`, `prim`, `rooms` |
| Obs types | `discrete` [row,col,gr,gc], `local` 5×5 window, `image` full map |
| Reward types | `sparse`, `dense` (Δ-Manhattan), `potential` (γΦ−Φ') |
| Multi-goal | ✅ visit N waypoints in sequence |
| Truncation | ✅ `max_steps` parameter |
| Oracle | ✅ BFS optimal baseline + distance map |

---

## What's Still Missing — Implementation Guide

---

### 1. `algorithms/q_learning.py`  ← Start Here

**Difficulty:** ⭐⭐ (beginner-friendly)
**Works with:** `obs_type="discrete"`, `MazeEnv-v0`

#### Steps

```python
# Pseudocode
Q = defaultdict(lambda: np.zeros(4))   # state → Q-values

for episode in range(n_episodes):
    obs, _ = env.reset()
    state = tuple(obs)
    done = False

    while not done:
        # ε-greedy action
        if rng.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        next_obs, reward, term, trunc, _ = env.step(action)
        next_state = tuple(next_obs)
        done = term or trunc

        # Bellman update
        best_next = np.max(Q[next_state]) if not term else 0.0
        Q[state][action] += alpha * (reward + gamma * best_next - Q[state][action])

        state = next_state

    epsilon = max(epsilon_end, epsilon * decay_rate)
```

#### What to Log
- Episode return (sum of rewards)
- Episode length (steps to goal)
- Success rate (% episodes where agent reached goal before truncation)
- ε over time

#### Expected Results (10×10 DFS maze, dense reward)
- Episode 0–500: mostly truncated, return ≈ −5 to 5
- Episode 500–2000: success rate climbs to ~60%
- Episode 2000+: success rate >90%, avg steps ≈ 50–100

---

### 2. `algorithms/dqn.py`

**Difficulty:** ⭐⭐⭐ (intermediate — requires PyTorch)
**Works with:** all `obs_type` values
**Install:** `pip install torch`

#### Architecture

```python
# For discrete obs (4 numbers in)
class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 4)
        )
    def forward(self, x): return self.net(x.float())

# For image obs (21×21×1 image in) — add after Q-learning works
class QNetCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Sequential(nn.Linear(32*4*4, 128), nn.ReLU(), nn.Linear(128, 4))
    def forward(self, x): return self.fc(self.conv(x.float() / 1.0))
```

#### Replay Buffer

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(np.array, zip(*batch))

    def __len__(self): return len(self.buffer)
```

#### Extensions Roadmap
1. **Double DQN** — 1 line change in target computation
2. **Dueling DQN** — split V and A streams in QNet
3. **Prioritised Replay** — weight transitions by |TD error|

---

### 3. `algorithms/policy_gradient.py`

**Difficulty:** ⭐⭐⭐⭐ (advanced)
**Install:** `pip install torch`

#### Recommended Order
1. REINFORCE (no baseline) on sparse 5×5
2. REINFORCE + baseline (mean return)
3. REINFORCE + learned V(s) baseline
4. A2C with n-step rollouts
5. PPO with clipped surrogate

#### Actor-Critic Network Skeleton

```python
class ActorCritic(nn.Module):
    def __init__(self, obs_dim=4, n_actions=4, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU())
        self.actor  = nn.Linear(hidden, n_actions)   # → logits
        self.critic = nn.Linear(hidden, 1)            # → V(s)

    def forward(self, x):
        h = self.shared(x.float())
        return self.actor(h), self.critic(h)

    def act(self, obs):
        logits, value = self(obs)
        dist   = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value
```

---

### 4. Training Entry Point — `train.py` (missing, create next)

```
maze_rl_project/
└── train.py    ← loads config YAML, picks algorithm, runs training loop
```

Suggested interface:
```bash
python train.py --algo q_learning --config configs/q_learning.yaml
python train.py --algo dqn        --config configs/dqn.yaml
```

---

### 5. Evaluation Script — `evaluate.py` (missing)

```python
# What it should do:
# 1. Load a saved model / Q-table
# 2. Run N greedy episodes
# 3. Report: mean return, std, success rate, mean steps
# 4. Compare against BFS oracle (optimality ratio)
# 5. Save a GIF of one solved episode
```

---

### 6. Notebooks (missing, add to `notebooks/`)

| Notebook | Purpose |
|---|---|
| `01_env_exploration.ipynb` | Visualise maze, run oracle, plot distance map heatmap |
| `02_q_learning.ipynb` | Train Q-learning, plot learning curves |
| `03_dqn.ipynb` | Train DQN, compare obs types |
| `04_comparison.ipynb` | Side-by-side algorithm comparison charts |

---

### 7. Tests (missing, add to `tests/`)

```
tests/
├── test_maze_grid.py    ← unit test wall removal, BFS reachability
├── test_generators.py   ← all mazes are perfect (no isolated cells)
└── test_env_api.py      ← gymnasium.utils.check_env(env) conformance
```

Run with: `pip install pytest && pytest tests/`

---

## Recommended Development Order

```
Week 1 — Q-Learning
  ├── Implement q_learning.py
  ├── Create train.py entry point
  ├── Train on MazeEnv-v0 (10×10 dense reward)
  └── Plot: return curve + success rate

Week 2 — DQN
  ├── Implement QNet + ReplayBuffer
  ├── Add Double DQN
  └── Compare DQN vs Q-learning on same env

Week 3 — Policy Gradient
  ├── Implement REINFORCE
  ├── Add A2C
  └── Try PPO via stable-baselines3 as reference

Week 4 — Analysis
  ├── evaluate.py + comparison notebook
  ├── Ablation: obs_type (discrete vs image)
  ├── Ablation: reward_type (sparse vs dense vs potential)
  └── Scale to 20×20 maze
```

---

## Quick Start

```bash
# Install deps
pip install -r requirements.txt

# Verify environment works
python test_env.py        # should print 8/8 passed

# Use the env in your own script
import sys; sys.path.insert(0, '.')
from maze_env import MazeEnv
env = MazeEnv(maze_size=(10,10), seed=42)
obs, info = env.reset()
obs, r, term, trunc, info = env.step(2)   # move South
```
