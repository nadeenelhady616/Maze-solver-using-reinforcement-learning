# to run: py live_policygradient.py --watch-training

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque

import numpy as np
import torch
import pygame

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

for p in (_HERE, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from maze_env import MazeEnv
from policy_gradient import PolicyNetwork, REINFORCE, preprocess_state


# ============================================================
# CONFIG
# ============================================================

DEFAULTS = dict(
    maze_size=(5, 5),
    maze_type="dfs",
    reward_type="potential",
    lr=1e-4,
    gamma=0.99,
    entropy_coef=0.001,
    n_episodes=2000,
    max_steps=300,
    maze_seed=42,
    log_every=100,
    watch_training=False,
    watch_every=100,
)


# ============================================================
# DISPLAY CONFIG
# ============================================================

CELL   = 80
MARGIN = 20

WHITE = (230, 230, 230)
BLACK = (25,  25,  25)
BLUE  = (40,  120, 255)
RED   = (220, 60,  60)


# ============================================================
# RENDER
# ============================================================

def render(env, screen):

    rows = env.maze_rows
    cols = env.maze_cols

    screen.fill(BLACK)

    for r in range(rows):
        for c in range(cols):

            x = MARGIN + c * CELL
            y = MARGIN + r * CELL

            pygame.draw.rect(screen, WHITE, (x, y, CELL, CELL))

            walls = env._grid.walls[r, c]

            if walls[0]:
                pygame.draw.line(screen, BLACK, (x, y), (x + CELL, y), 4)
            if walls[1]:
                pygame.draw.line(screen, BLACK, (x + CELL, y), (x + CELL, y + CELL), 4)
            if walls[2]:
                pygame.draw.line(screen, BLACK, (x, y + CELL), (x + CELL, y + CELL), 4)
            if walls[3]:
                pygame.draw.line(screen, BLACK, (x, y), (x, y + CELL), 4)

    gr, gc = env._goals[0]
    pygame.draw.circle(
        screen, RED,
        (MARGIN + gc * CELL + CELL // 2, MARGIN + gr * CELL + CELL // 2),
        CELL // 4,
    )

    ar, ac = env._agent_pos
    pygame.draw.circle(
        screen, BLUE,
        (MARGIN + ac * CELL + CELL // 2, MARGIN + ar * CELL + CELL // 2),
        CELL // 3,
    )

    pygame.display.flip()


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

    state, _ = env.reset(seed=cfg["maze_seed"])
    state = preprocess_state(state)
    state_dim  = state.shape[0]
    action_dim = env.action_space.n

    print("=" * 70)
    print(
        f" Policy Gradient Live | "
        f"{cfg['maze_size'][0]}x{cfg['maze_size'][1]} | "
        f"{cfg['maze_type']} | "
        f"reward={cfg['reward_type']}"
    )
    print("=" * 70)

    policy = PolicyNetwork(state_dim, action_dim)
    agent  = REINFORCE(
        policy,
        lr=cfg["lr"],
        gamma=cfg["gamma"],
        entropy_coef=cfg["entropy_coef"],
    )

    recent_returns = deque(maxlen=100)
    solved_recent  = deque(maxlen=100)
    start_time     = time.time()

    screen = None
    rows, cols = cfg["maze_size"]

    for ep in range(1, cfg["n_episodes"] + 1):

        state, _ = env.reset(seed=cfg["maze_seed"])
        done      = False
        total_reward = 0
        log_probs = []
        rewards   = []
        entropies = []
        solved    = False
        step      = 0

        while not done and step < cfg["max_steps"]:

            state_t = preprocess_state(state)
            probs   = policy(state_t)
            dist    = torch.distributions.Categorical(probs)
            action  = dist.sample()

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            reward -= 0.01
            if terminated:
                reward += 10
                solved = True

            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())
            rewards.append(reward)
            total_reward += reward
            state = next_state
            step += 1

            # ------------------------------------------------
            # LIVE WATCH — جوا الـ step loop زي الـ Q-Learning
            # ------------------------------------------------
            if cfg["watch_training"] and ep % cfg["watch_every"] == 0:

                if screen is None:
                    pygame.init()
                    screen = pygame.display.set_mode(
                        (cols * CELL + 2 * MARGIN, rows * CELL + 2 * MARGIN)
                    )
                    pygame.display.set_caption("Policy Gradient — Training Live")

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        cfg["watch_training"] = False

                render(env, screen)
                pygame.time.delay(10)

        agent.update(log_probs, rewards, entropies)
        recent_returns.append(total_reward)
        solved_recent.append(float(solved))

        if ep % cfg["log_every"] == 0:
            print(
                f"Ep {ep:5d}/{cfg['n_episodes']} | "
                f"avg_return={np.mean(recent_returns):7.2f} | "
                f"solved={np.mean(solved_recent)*100:5.1f}% | "
                f"time={time.time()-start_time:.0f}s"
            )

    env.close()

    if screen is not None:
        pygame.quit()

    print("\nTraining complete.")
    return policy


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

    return cfg


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    cfg = parse_args()
    train(cfg)