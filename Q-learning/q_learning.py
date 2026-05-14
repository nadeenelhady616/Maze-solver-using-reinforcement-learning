# to run: py q_learning.py
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from collections import defaultdict, deque
import numpy as np
import pygame

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

for p in (_HERE, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from maze_env import MazeEnv
# ============================================================
# CONFIG
# ============================================================

DEFAULTS = dict(
    maze_size=(8, 8),
    maze_type="prim",
    reward_type="potential",
    alpha=0.05,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.10,
    epsilon_decay=500_000,
    n_episodes=2000,
    max_steps=300,
    maze_seed=42,
    log_every=100,
    save_path=os.path.join(_HERE, "results", "q_table.pkl"),
)

# ============================================================
# Q TABLE
# ============================================================
class QTable:

    def __init__(self, n_actions=4):
        self.n_actions = n_actions
        self.table = {}

    def key(self, obs):
        return tuple(int(x) for x in obs)

    def get(self, obs):
        k = self.key(obs)
        if k not in self.table:
            self.table[k] = np.zeros(
            self.n_actions,
            dtype=np.float32
        )
        return self.table[k]

    def best_action(self, obs):
        qvals = self.get(obs)
        max_q = np.max(qvals)
        best = np.flatnonzero(qvals == max_q)
        return int(np.random.choice(best))

    def update(self, obs, action, value):
        self.get(obs)[action] = value

    def max_q(self, obs):
        return float(np.max(self.get(obs)))

    @property
    def size(self):

        return len(self.table)

# ============================================================
# HELPERS
# ============================================================
def valid_actions(env):

    r, c = env._agent_pos
    valid = []

    for a in range(4):

        d = env._action_to_dir[a]
        if env._grid.can_move(r, c, d):
            valid.append(a)

    return valid

def linear_decay(step, start, end, decay_steps):
    ratio = min(step / decay_steps, 1.0)
    return start + ratio * (end - start)


def epsilon_greedy(q, obs, epsilon, rng, valid):

    if rng.random() < epsilon:
        return int(rng.choice(valid))
    qvals = q.get(obs)
    best_q = np.max(qvals[valid])
    best = [a for a in valid if qvals[a] == best_q]
    return int(rng.choice(best))

# ============================================================
# TRAIN
# ============================================================

def train(cfg):

    if cfg["maze_seed"] is None:
        cfg["maze_seed"] = np.random.randint(0, 100000)

    env = MazeEnv(
        maze_size=cfg["maze_size"],
        maze_type=cfg["maze_type"],
        obs_type="discrete",
        reward_type=cfg["reward_type"],
        max_steps=cfg["max_steps"],
    )

    rows, cols = cfg["maze_size"]

    obs, _ = env.reset()

    print("=" * 70)
    print(
        f" Q-Learning | {rows}x{cols} | "
        f"{cfg['maze_type']} | reward={cfg['reward_type']}"
    )
    print("=" * 70)

    q = QTable()
    rng = np.random.default_rng(cfg["maze_seed"])
    total_steps = 0
    recent_returns = deque(maxlen=200)
    recent_solved = deque(maxlen=200)
    start_time = time.time()

    screen = None
    CELL = 40

    for ep in range(1, cfg["n_episodes"] + 1):
        obs, _ = env.reset(seed=cfg["maze_seed"])
        ep_return = 0
        solved = False

        for step in range(cfg["max_steps"]):
            epsilon = linear_decay(
                total_steps,
                cfg["epsilon_start"],
                cfg["epsilon_end"],
                cfg["epsilon_decay"],
            )

            valid = valid_actions(env)
            action = epsilon_greedy( q,obs,epsilon,rng,valid,)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            old_q = q.get(obs)[action]
            target = reward
            if not done:
                target += cfg["gamma"] * q.max_q(next_obs)
            new_q = old_q + cfg["alpha"] * (target - old_q)
            q.update(obs, action, new_q)
            obs = next_obs
            ep_return += reward
            total_steps += 1
            if terminated:
                solved = True
                break
            if truncated:
                break
        recent_returns.append(ep_return)

        recent_solved.append(float(solved))

        if ep % cfg["log_every"] == 0:

            print(
                f"Ep {ep:5d}/{cfg['n_episodes']} | "
                f"eps={epsilon:.3f} | "
                f"avg_return={np.mean(recent_returns):7.2f} | "
                f"solved={np.mean(recent_solved)*100:5.1f}% | "
                f"states={q.size:4d} | "
                f"time={time.time()-start_time:.0f}s"
            )

    env.close()

    print("\nTraining complete.")
    print(f"Q-table states: {q.size}")

    return q

# ============================================================
# EVALUATE
# ============================================================

def evaluate(q, cfg, n_eval=100):

    env = MazeEnv(
        maze_size=cfg["maze_size"],
        maze_type=cfg["maze_type"],
        obs_type="discrete",
        reward_type=cfg["reward_type"],
        max_steps=cfg["max_steps"],
    )

    solved = 0

    steps_list = []

    for _ in range(n_eval):

        obs, _ = env.reset(seed=cfg["maze_seed"])

        for step in range(cfg["max_steps"]):

            valid = valid_actions(env)
            qvals = q.get(obs)
            best_q = np.max(qvals[valid])
            best = [a for a in valid if qvals[a] == best_q]
            action = int(np.random.choice(best))

            obs, reward, terminated, truncated, _ = env.step(action)

            if terminated:
                solved += 1
                steps_list.append(step + 1)
                break

            if truncated:
                break

    env.close()

    rate = solved / n_eval * 100

    avg_steps = np.mean(steps_list) if steps_list else float("nan")

    print("\nEvaluation")
    print("-" * 40)
    print(f"Solve rate : {rate:.1f}%")
    print(f"Avg steps  : {avg_steps:.1f}")

# ============================================================
# SAVE / LOAD
# ============================================================

def save_model(q, cfg, path):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(
            {
               "table": q.table,
               "config": cfg,
            },
            f,
        )

    print(f"\nSaved model -> {path}")


def load_model(path):

    with open(path, "rb") as f:
        data = pickle.load(f)
    q = QTable()
    q.table = data["table"]
    return q, data["config"]

    return data["q_table"], data["config"]

# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
    "--maze-seed",
    type=int,
    default=None,
    )

    parser.add_argument(
        "--maze-size",
        nargs=2,
        type=int,
        default=list(DEFAULTS["maze_size"]),
    )

    parser.add_argument(
        "--maze-type",
        type=str,
        default=DEFAULTS["maze_type"],
    )

    parser.add_argument(
        "--reward-type",
        type=str,
        default=DEFAULTS["reward_type"],
    )

    parser.add_argument(
        "--n-episodes",
        type=int,
        default=DEFAULTS["n_episodes"],
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULTS["max_steps"],
    )
    args = parser.parse_args()
    cfg = DEFAULTS.copy()
    cfg.update(vars(args))
    cfg["maze_size"] = tuple(cfg["maze_size"])
    return cfg

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    cfg = parse_args()
    q = train(cfg)
    evaluate(q, cfg)
    save_model(q, cfg, cfg["save_path"])