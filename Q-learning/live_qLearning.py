from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from collections import deque
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
    maze_type="dfs",
    reward_type="potential",
    alpha=0.05,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.10,
    epsilon_decay=500_000,
    n_episodes=2000,
    max_steps=600,
    maze_seed=42,
    log_every=200,
    watch_training=False,
    watch_every=50,
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
            self.table[k] = np.zeros(self.n_actions, dtype=np.float32)
        return self.table[k]

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


def epsilon_greedy(q, obs, eps, rng, valid):
    if rng.random() < eps:
        return int(rng.choice(valid))

    qvals = q.get(obs)
    best_q = np.max(qvals[valid])
    best = [a for a in valid if qvals[a] == best_q]
    return int(rng.choice(best))


def linear_decay(step, start, end, decay):
    return start + min(step / decay, 1.0) * (end - start)

# ============================================================
# TRAIN
# ============================================================

def train(cfg):

    env = MazeEnv(
        maze_size=cfg["maze_size"],
        maze_type=cfg["maze_type"],
        obs_type="discrete",
        reward_type=cfg["reward_type"],
        max_steps=cfg["max_steps"],
    )

    rows, cols = cfg["maze_size"]

    q = QTable()
    rng = np.random.default_rng(cfg["maze_seed"])

    total_steps = 0
    recent_returns = deque(maxlen=200)
    recent_solved = deque(maxlen=200)

    screen = None
    CELL = 40

    for ep in range(1, cfg["n_episodes"] + 1):

        obs, _ = env.reset(seed=cfg["maze_seed"])
        ep_return = 0
        solved = False

        epsilon = linear_decay(
            total_steps,
            cfg["epsilon_start"],
            cfg["epsilon_end"],
            cfg["epsilon_decay"],
        )

        for step in range(cfg["max_steps"]):

            valid = valid_actions(env)
            action = epsilon_greedy(q, obs, epsilon, rng, valid)

            next_obs, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated

            old_q = q.get(obs)[action]
            target = reward + (0 if done else cfg["gamma"] * q.max_q(next_obs))

            q.update(obs, action, old_q + cfg["alpha"] * (target - old_q))

            obs = next_obs
            ep_return += reward
            total_steps += 1

            if terminated:
                solved = True
                break
            if truncated:
                break

            # ====================================================
            # SAFE WATCH MODE (NOT SLOWING TRAINING TOO MUCH)
            # ====================================================
            if cfg["watch_training"] and ep % cfg["watch_every"] == 0:
                if screen is None:
                    pygame.init()
                    screen = pygame.display.set_mode((cols * CELL, rows * CELL))

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        cfg["watch_training"] = False

                render(env, screen, CELL)
                pygame.time.delay(5) 

        recent_returns.append(ep_return)
        recent_solved.append(float(solved))

        if ep % cfg["log_every"] == 0:
            print(
                f"Ep {ep:5d} | eps={epsilon:.3f} | "
                f"avg_return={np.mean(recent_returns):7.2f} | "
                f"solved={np.mean(recent_solved)*100:5.1f}% | "
                f"states={q.size:5d}"
            )

    env.close()
    return q

# ============================================================
# RENDER
# ============================================================

def render(env, screen, cell=40):

    rows = env.maze_rows
    cols = env.maze_cols

    screen.fill((20, 20, 20))

    for r in range(rows):
        for c in range(cols):

            x = c * cell
            y = r * cell

            if env._grid.walls[r, c, 0]:
                pygame.draw.line(screen, (255,255,255), (x,y), (x+cell,y))
            if env._grid.walls[r, c, 1]:
                pygame.draw.line(screen, (255,255,255), (x+cell,y), (x+cell,y+cell))
            if env._grid.walls[r, c, 2]:
                pygame.draw.line(screen, (255,255,255), (x,y+cell), (x+cell,y+cell))
            if env._grid.walls[r, c, 3]:
                pygame.draw.line(screen, (255,255,255), (x,y), (x,y+cell))

    ar, ac = env._agent_pos
    pygame.draw.circle(screen, (50,150,255),
                       (ac*cell+cell//2, ar*cell+cell//2), cell//3)

    gr, gc = env._goals[0]
    pygame.draw.circle(screen, (255,80,80),
                       (gc*cell+cell//2, gr*cell+cell//2), cell//3)

    pygame.display.flip()

# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--maze-size", nargs=2, type=int,
                        default=list(DEFAULTS["maze_size"]))

    parser.add_argument("--maze-type", type=str,
                        default=DEFAULTS["maze_type"])

    parser.add_argument("--reward-type", type=str,
                        default=DEFAULTS["reward_type"])

    parser.add_argument("--n-episodes", type=int,
                        default=DEFAULTS["n_episodes"])

    parser.add_argument("--max-steps", type=int,
                        default=DEFAULTS["max_steps"])

    parser.add_argument("--watch-training", action="store_true")

    parser.add_argument("--watch-every", type=int,
                        default=DEFAULTS["watch_every"])

    args = parser.parse_args()

    cfg = DEFAULTS.copy()
    cfg.update(vars(args))
    cfg["maze_size"] = tuple(cfg["maze_size"])

    if cfg["maze_seed"] is None:
        cfg["maze_seed"] = np.random.randint(0, 100000)

    return cfg

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    cfg = parse_args()
    q = train(cfg)