"""
algorithms/policy_gradient.py  [STUB — TO BE IMPLEMENTED]
----------------------------------------------------------
REINFORCE + Actor-Critic (A2C) policy gradient methods.

REINFORCE (vanilla policy gradient)
  □ PolicyNetwork: maps obs → softmax(logits) over 4 actions
  □ Collect full episode trajectories
  □ Compute discounted returns G_t = Σ_{k≥t} γ^(k-t) r_k
  □ Loss: -Σ_t log π(a_t|s_t) · G_t    (negative because we ascend)
  □ Optional baseline: subtract mean(G) or a learned V(s) to reduce variance
  □ Entropy bonus: add β·H(π) to encourage exploration

A2C (synchronous Advantage Actor-Critic)
  □ ActorCriticNetwork: shared trunk → policy head + value head
  □ Advantage estimate: A_t = r_t + γ·V(s_{t+1}) - V(s_t)  (TD-error)
  □ Policy loss:  -Σ log π(a_t|s_t) · A_t
  □ Value  loss:   Σ (V(s_t) - G_t)²
  □ Combined:     policy_loss + c1·value_loss - c2·entropy
  □ n-step rollouts (instead of full episodes)

Extensions:
  □ PPO (Proximal Policy Optimisation) — clip ratio, multiple epochs
      ratio = π_new / π_old; clip to [1-ε, 1+ε] where ε=0.2
  □ GAE (Generalised Advantage Estimation) — λ-weighted TD returns

Hyper-parameters (see configs/policy_gradient.yaml):
  lr:           3e-4
  gamma:        0.99
  entropy_coef: 0.01
  value_coef:   0.5
  n_steps:      128   (for A2C rollout length)
  gae_lambda:   0.95  (for PPO/GAE)
  clip_eps:     0.2   (for PPO)

Tip: Start with REINFORCE on sparse-reward 5×5, then scale up to A2C on 10×10.
"""

raise NotImplementedError(
    "Policy Gradient is not yet implemented. See the docstring above."
)
