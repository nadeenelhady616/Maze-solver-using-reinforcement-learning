# to run: python policy_gradient.py

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    maze_size=(5, 5),
    maze_type="dfs",
    reward_type="potential",
    lr=1e-4,
    gamma=0.99,
    entropy_coef=0.001,
    n_episodes=1000,
    max_steps=300,
    maze_seed=42,
    log_every=50,
    save_path=os.path.join(_HERE, "results", "policy.pkl"),
)


# ============================================================
# HELPERS
# ============================================================

def preprocess_state(state):

    if isinstance(state, tuple):
        state = state[0]

    if isinstance(state, dict):
        for key in ["state", "observation", "obs"]:
            if key in state:
                state = state[key]
                break

    state = np.array(state, dtype=np.float32)
    state = state / 4.0

    return torch.tensor(state, dtype=torch.float32)


# ============================================================
# POLICY NETWORK
# ============================================================

class PolicyNetwork(nn.Module):

    def __init__(self, state_dim, action_dim):

        super().__init__()

        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, action_dim)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return F.softmax(self.out(x), dim=-1)


# ============================================================
# REINFORCE AGENT
# ============================================================

class REINFORCE:

    def __init__(self, policy, lr=1e-4, gamma=0.99, entropy_coef=0.001):

        self.policy = policy

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=lr,
        )

        self.gamma = gamma
        self.entropy_coef = entropy_coef

    def compute_returns(self, rewards):
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return torch.tensor(returns, dtype=torch.float32)

    def update(self, log_probs, rewards, entropies):

        returns = self.compute_returns(rewards)

        loss = 0

        for log_p, G, ent in zip(log_probs, returns, entropies):
            loss += -log_p * G - self.entropy_coef * ent

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()


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
    state_dim = state.shape[0]
    action_dim = env.action_space.n

    print("=" * 70)
    print(
        f" Policy Gradient | "
        f"{cfg['maze_size'][0]}x{cfg['maze_size'][1]} | "
        f"{cfg['maze_type']} | "
        f"reward={cfg['reward_type']}"
    )
    print("=" * 70)

    policy = PolicyNetwork(state_dim, action_dim)

    agent = REINFORCE(
        policy,
        lr=cfg["lr"],
        gamma=cfg["gamma"],
        entropy_coef=cfg["entropy_coef"],
    )

    recent_returns = deque(maxlen=100)
    solved_recent = deque(maxlen=100)
    start_time = time.time()

    for ep in range(1, cfg["n_episodes"] + 1):

        state, _ = env.reset(seed=cfg["maze_seed"])
        done = False
        total_reward = 0
        log_probs = []
        rewards = []
        entropies = []
        solved = False
        step = 0

        while not done and step < cfg["max_steps"]:

            state_t = preprocess_state(state)
            probs = policy(state_t)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

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
    print("\nTraining complete.")

    return policy


# ============================================================
# EVALUATE
# ============================================================

def evaluate(policy, cfg, n_eval=100):

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

        state, _ = env.reset(seed=cfg["maze_seed"])
        done = False

        for step in range(cfg["max_steps"]):

            state_t = preprocess_state(state)

            with torch.no_grad():
                probs = policy(state_t)

            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state

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

def save_model(policy, cfg, path):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(
            {
                "state_dict": policy.state_dict(),
                "state_dim":  next(iter(policy.parameters())).shape[1],
                "action_dim": policy.out.out_features,
                "config":     cfg,
            },
            f,
        )

    print(f"Saved model -> {path}")


def load_model(path):

    with open(path, "rb") as f:
        data = pickle.load(f)

    policy = PolicyNetwork(data["state_dim"], data["action_dim"])
    policy.load_state_dict(data["state_dict"])
    policy.eval()

    return policy, data["config"]


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
    policy = train(cfg)
    evaluate(policy, cfg)
    save_model(policy, cfg, cfg["save_path"])