"""
test_env.py
-----------
Verifies the maze environment is working correctly.

Run from the SAME folder as maze_env.py and oracle.py:
    python test_env.py

All 8 tests should pass and a PNG image will be saved showing the 3 maze types.
"""

import sys
import os

# Add the current folder to Python's search path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless — no GUI window needed
import matplotlib.pyplot as plt

# ── Fixed imports: flat structure, no subfolders ──────────────────────────
from maze_env import MazeEnv, GENERATORS
from oracle   import OracleAgent, compute_distance_map
# ─────────────────────────────────────────────────────────────────────────

# Coloured checkmarks for the terminal output
PASS = "\033[92m✔\033[0m"
FAIL = "\033[91m✘\033[0m"


def check(condition, message):
    """
    Assert a condition is True and print a coloured result.
    If the condition is False, raise AssertionError and stop the test.
    """
    status = PASS if condition else FAIL
    print(f"  {status} {message}")
    if not condition:
        raise AssertionError(message)


# ===========================================================================
# TEST 1 — Basic API
# ===========================================================================
# Verifies:
#   - env creates without errors
#   - reset() returns the right observation shape
#   - agent starts at (0,0)
#   - step() returns the right types
def test_basic():
    print("\n[Test 1] Basic API")
    env = MazeEnv(maze_size=(5, 5), seed=42)

    # reset() must return (observation, info_dict)
    obs, info = env.reset()

    check(obs.shape == (4,),          f"obs shape is {obs.shape}, expected (4,)")
    check(info["agent_pos"] == (0, 0), f"agent should start at (0,0), got {info['agent_pos']}")
    check(env.action_space.n == 4,    "action space must have exactly 4 actions")

    # Take one step — action 2 = South
    obs2, reward, terminated, truncated, info2 = env.step(2)

    check(isinstance(reward, float),  "reward must be a float")
    check(not terminated,             "should not be terminated after 1 step")

    env.close()


# ===========================================================================
# TEST 2 — Random rollout
# ===========================================================================
# Runs a full episode with random actions to check the episode loop works.
# max_steps=200 guarantees the episode MUST end via truncation.
def test_random_rollout():
    print("\n[Test 2] Random rollout until done/truncated")
    env = MazeEnv(maze_size=(8, 8), max_steps=200, seed=7)
    obs, _ = env.reset()
    total_reward = 0.0

    for step in range(500):
        action = env.action_space.sample()                      # random action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            check(True, f"Episode ended at step {step+1} | total_reward={total_reward:.2f}")
            break

    env.close()


# ===========================================================================
# TEST 3 — Oracle BFS agent
# ===========================================================================
# The BFS oracle knows the maze layout and computes the shortest path.
# If the oracle can navigate to the goal, the maze is solvable and the
# step/reward system is working correctly.
def test_oracle():
    print("\n[Test 3] Oracle BFS agent")

    env = MazeEnv(maze_size=(10, 10), reward_type="sparse", seed=0)
    obs, _ = env.reset(seed=0)     # seed=0 → reproducible maze
    grid   = env.get_maze_grid()   # get the maze layout for BFS

    agent = OracleAgent()
    agent.reset(grid, start=(0, 0), goal=(9, 9))

    steps = 0
    total_reward = 0.0
    terminated = False

    while not agent.done:
        action = agent.act()                                            # optimal action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if terminated:
            break

    check(terminated, f"Oracle reached the goal in {steps} steps (reward={total_reward})")
    check(steps <= env.maze_rows * env.maze_cols * 2,
          f"Oracle used a reasonable number of steps: {steps}")

    # Check the distance map
    dist_map = compute_distance_map(env.get_maze_grid(), goal=(9, 9))
    check(dist_map[0, 0] < np.inf,  "Start (0,0) is reachable from goal")
    check(dist_map[9, 9] == 0,      "Distance at goal itself is 0")

    env.close()


# ===========================================================================
# TEST 4 — All maze generators
# ===========================================================================
# Each generator should produce a solvable maze.
# We verify this by checking that BFS finds a non-empty path.
def test_generators():
    print("\n[Test 4] Maze generators")

    for gen_name in GENERATORS:   # "dfs", "prim", "rooms"
        env = MazeEnv(maze_size=(7, 7), maze_type=gen_name, seed=1)
        env.reset()
        grid  = env.get_maze_grid()
        agent = OracleAgent()
        agent.reset(grid, start=(0, 0), goal=(6, 6))

        check(agent._plan is not None and len(agent._plan) > 0,
              f"'{gen_name}' generator: solvable (path length = {len(agent._plan)})")
        env.close()


# ===========================================================================
# TEST 5 — Observation types
# ===========================================================================
# Each obs_type must produce exactly the expected array shape,
# both right after reset() and after a step().
def test_obs_types():
    print("\n[Test 5] Observation types")

    # (obs_type, expected_numpy_shape)
    configs = [
        ("discrete", (4,)),          # flat vector
        ("local",    (5, 5, 4)),     # 5×5 window, 4 wall flags per cell
        ("image",    (21, 21, 1)),   # (2*10+1)×(2*10+1)×1 = 21×21×1
    ]

    for obs_type, expected_shape in configs:
        env = MazeEnv(maze_size=(10, 10), obs_type=obs_type, seed=3)

        obs, _ = env.reset()
        check(obs.shape == expected_shape,
              f"obs_type='{obs_type}': reset shape {obs.shape} == {expected_shape}")

        obs2, _, _, _, _ = env.step(1)
        check(obs2.shape == expected_shape,
              f"obs_type='{obs_type}': step  shape {obs2.shape} == {expected_shape}")

        env.close()


# ===========================================================================
# TEST 6 — Reward types and truncation
# ===========================================================================
# Each reward type must not crash, and the episode must end
# (either via goal or truncation within max_steps=10).
def test_rewards_and_truncation():
    print("\n[Test 6] Reward types & truncation")

    for rtype in ["sparse", "dense", "potential"]:
        env = MazeEnv(maze_size=(5, 5), reward_type=rtype, max_steps=10, seed=5)
        env.reset()

        for _ in range(15):   # run past max_steps to ensure truncation fires
            _, r, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                break

        check(terminated or truncated,
              f"reward_type='{rtype}': episode ended correctly")
        env.close()


# ===========================================================================
# TEST 7 — Multi-goal variant
# ===========================================================================
# With n_goals=3, the agent must visit 2 intermediate waypoints then the
# final goal. The oracle handles each leg sequentially.
def test_multi_goal():
    print("\n[Test 7] Multi-goal variant")
    env = MazeEnv(maze_size=(8, 8), n_goals=3, max_steps=2000, seed=11)
    obs, info = env.reset()

    check(len(env._goals) == 3, f"Should have 3 goals, got {len(env._goals)}")

    grid  = env.get_maze_grid()
    start = (0, 0)
    total_steps = 0
    terminated  = False

    # Oracle solves each leg: start→goal1, goal1→goal2, goal2→goal3
    for goal in env._goals:
        agent = OracleAgent()
        agent.reset(grid, start, goal)
        while not agent.done:
            _, _, terminated, truncated, info = env.step(agent.act())
            total_steps += 1
            if terminated or truncated:
                break
        start = goal   # next leg starts from where we just arrived

    check(terminated, f"All 3 goals reached in {total_steps} steps")
    env.close()


# ===========================================================================
# VISUAL TEST — Save PNG comparing all 3 generators
# ===========================================================================
# Not a pass/fail test — just saves an image so you can visually inspect
# what DFS, Prim, and Rooms mazes look like.
def save_visual():
    print("\n[Visual] Saving maze renders ...")
    os.makedirs("results", exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#1a1a2e")

    for ax, gen_name in zip(axes, list(GENERATORS.keys())):
        env = MazeEnv(maze_size=(8, 8), maze_type=gen_name, seed=42)
        env.reset()
        img = env._render_image()
        ax.imshow(img)
        ax.set_title(gen_name.upper(), color="white", fontsize=13, fontweight="bold")
        ax.axis("off")
        env.close()

    plt.suptitle("Maze RL — Generator Comparison", color="white", fontsize=15, y=1.02)
    plt.tight_layout()

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "maze_generators.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  {PASS} Saved: {out_path}")


# ===========================================================================
# MAIN — run all tests
# ===========================================================================
if __name__ == "__main__":
    print("=" * 55)
    print("  Maze RL Environment — Test Suite")
    print("=" * 55)

    tests = [
        test_basic,
        test_random_rollout,
        test_oracle,
        test_generators,
        test_obs_types,
        test_rewards_and_truncation,
        test_multi_goal,
        save_visual,
    ]

    passed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  \033[91mFAILED: {e}\033[0m")
        except Exception as e:
            print(f"  \033[91mERROR in {test_fn.__name__}: {e}\033[0m")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 55)
    print(f"  Results: {passed}/{len(tests)} tests passed")
    print("=" * 55)