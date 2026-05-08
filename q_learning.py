"""
algorithms/q_learning.py  [STUB — TO BE IMPLEMENTED]
------------------------------------------------------
Tabular Q-Learning for the MazeEnv with discrete observations.

Implementation checklist:
  □ Q-table: dict or numpy array indexed by (row, col, goal_row, goal_col)
  □ ε-greedy policy with linear/exponential decay schedule
  □ Bellman update:  Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') − Q(s,a)]
  □ Training loop with episode logging
  □ save / load Q-table (numpy .npy)
  □ Evaluation loop (greedy policy, no exploration)

Hyper-parameters to expose (see configs/q_learning.yaml):
  alpha (learning rate):    0.1
  gamma (discount factor):  0.99
  epsilon_start:            1.0
  epsilon_end:              0.05
  epsilon_decay_steps:      50_000
  n_episodes:               5_000
  max_steps_per_ep:         500

How to test convergence:
  Plot episode return vs episode number.
  A working agent should reach the goal reliably by episode ~1000
  on a 10×10 maze.
"""

raise NotImplementedError(
    "Q-Learning is not yet implemented. "
    "See the docstring above for a full implementation guide."
)
