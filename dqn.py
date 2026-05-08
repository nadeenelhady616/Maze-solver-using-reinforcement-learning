"""
algorithms/dqn.py  [STUB — TO BE IMPLEMENTED]
----------------------------------------------
Deep Q-Network (DQN) — works with all obs_types.

Implementation checklist (Mnih et al. 2015 + improvements):
  □ QNetwork: MLP for discrete obs, CNN for image obs
      - discrete branch: Linear → ReLU → Linear → ReLU → Linear(4)
      - image   branch:  Conv2d×2 → Flatten → Linear×2 → Linear(4)
  □ ReplayBuffer: circular buffer, sample random minibatches
  □ Target network: hard-copy every C steps (or soft-update τ)
  □ ε-greedy exploration (same schedule as Q-learning)
  □ Huber loss on TD error
  □ Training loop with:
      - warm-up phase (fill buffer before updates)
      - gradient clipping (max_norm=10)
      - logging: loss, epsilon, episode return
  □ save / load model weights (torch.save / torch.load)

Extensions to add (one at a time):
  □ Double DQN       — use online net to SELECT action, target net to EVALUATE
  □ Dueling DQN      — split stream into V(s) and A(s,a)
  □ Prioritised Replay — weight samples by |TD error|
  □ n-step returns   — Bellman target uses n-step sum + γⁿ·max Q(sₙ,·)

Hyper-parameters (see configs/dqn.yaml):
  lr:                1e-3
  gamma:             0.99
  batch_size:        64
  buffer_size:       50_000
  target_update:     500   (steps)
  epsilon_start:     1.0
  epsilon_end:       0.05
  epsilon_decay:     10_000

Dependencies:
  pip install torch  (CPU build is fine for 10×10 mazes)
"""

raise NotImplementedError(
    "DQN is not yet implemented. See the docstring above."
)
