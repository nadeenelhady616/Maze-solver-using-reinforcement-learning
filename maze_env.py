"""
maze_env.py
-----------
A Gymnasium-compatible 2D maze environment for reinforcement learning.
Single-file version — no package subfolders needed.

Usage:
    from maze_env import MazeEnv
    env = MazeEnv(maze_size=(10, 10), seed=42)
    obs, info = env.reset()
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# ===========================================================================
# PART 1 — DIRECTIONS
# ===========================================================================
# We have 4 possible moves: North, East, South, West.
# Each direction maps to a (row_delta, col_delta) — how the agent's position
# changes when it moves that way.
# North = row-1 (up in the grid), South = row+1, East = col+1, West = col-1.

class Dir(IntEnum):
    N = 0
    E = 1
    S = 2
    W = 3


# How each direction changes (row, col)
_DELTA = {
    Dir.N: (-1,  0),
    Dir.E: ( 0, +1),
    Dir.S: (+1,  0),
    Dir.W: ( 0, -1),
}

# What direction is the reverse of each direction
_OPPOSITE = {Dir.N: Dir.S, Dir.E: Dir.W, Dir.S: Dir.N, Dir.W: Dir.E}


# ===========================================================================
# PART 2 — MAZE GRID (the data structure that stores walls)
# ===========================================================================
# The maze is a grid of CELLS. Between every two adjacent cells there is
# either a WALL (you can't pass) or an OPENING (you can pass).
#
# We store walls as:  walls[row, col, direction] = True/False
#   True  → wall EXISTS (you're blocked)
#   False → wall REMOVED (you can move through)
#
# Example 3×3 grid:
#   Initially every cell has 4 walls (N, E, S, W).
#   The generator removes walls to carve passages.

class MazeGrid:
    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        # Start with ALL walls present (True = wall exists)
        self.walls = np.ones((rows, cols, 4), dtype=bool)

    def in_bounds(self, r: int, c: int) -> bool:
        """Check if (r, c) is inside the grid."""
        return 0 <= r < self.rows and 0 <= c < self.cols

    def remove_wall(self, r: int, c: int, d: Dir):
        """
        Remove the wall between cell (r,c) and its neighbour in direction d.
        Must remove BOTH sides — if you remove the east wall of cell A,
        you must also remove the west wall of cell B (its eastern neighbour).
        """
        self.walls[r, c, d] = False
        dr, dc = _DELTA[d]
        nr, nc = r + dr, c + dc
        if self.in_bounds(nr, nc):
            self.walls[nr, nc, _OPPOSITE[d]] = False

    def can_move(self, r: int, c: int, d: Dir) -> bool:
        """True if the agent at (r,c) can move in direction d (no wall, in bounds)."""
        if self.walls[r, c, d]:
            return False                       # wall is blocking
        dr, dc = _DELTA[d]
        return self.in_bounds(r + dr, c + dc) # destination must be inside grid

    def neighbours(self, r: int, c: int) -> List[Tuple[int, int, Dir]]:
        """Return all valid (nr, nc, direction) neighbours of cell (r, c)."""
        result = []
        for d in Dir:
            dr, dc = _DELTA[d]
            nr, nc = r + dr, c + dc
            if self.in_bounds(nr, nc):
                result.append((nr, nc, d))
        return result

    def to_binary_map(self) -> np.ndarray:
        """
        Convert the maze to a pixel image for CNN observations.
        Returns a (2*rows+1) × (2*cols+1) array where:
          1 = wall pixel
          0 = open space pixel
        Each cell becomes a 1-pixel square, walls fill the gaps between them.
        """
        H = 2 * self.rows + 1
        W = 2 * self.cols + 1
        img = np.ones((H, W), dtype=np.uint8)   # start: everything is wall
        for r in range(self.rows):
            for c in range(self.cols):
                img[2*r+1, 2*c+1] = 0           # the cell itself is open
                if not self.walls[r, c, Dir.S] and r + 1 < self.rows:
                    img[2*r+2, 2*c+1] = 0        # passage going south
                if not self.walls[r, c, Dir.E] and c + 1 < self.cols:
                    img[2*r+1, 2*c+2] = 0        # passage going east
        return img

    def clone(self) -> "MazeGrid":
        g = MazeGrid(self.rows, self.cols)
        g.walls = self.walls.copy()
        return g


# ===========================================================================
# PART 3 — MAZE GENERATORS
# ===========================================================================
# These functions START with a fully-walled grid and CARVE passages
# until every cell is reachable (a "perfect maze" — exactly one path
# between any two cells, so it's always solvable).

def generate_dfs(rows: int, cols: int, rng: np.random.Generator) -> MazeGrid:
    """
    Recursive Backtracker (DFS).
    Think of it like exploring a building in the dark with a torch:
      1. Start at a random cell, mark it visited.
      2. Pick a random unvisited neighbour, knock down the wall, move there.
      3. If no unvisited neighbours, backtrack until you find one.
      4. Stop when every cell has been visited.
    Result: long, winding corridors with few dead-ends.
    """
    grid = MazeGrid(rows, cols)
    visited = np.zeros((rows, cols), dtype=bool)

    def dfs(r, c):
        visited[r, c] = True
        dirs = list(Dir)
        rng.shuffle(dirs)           # random order = random maze each time
        for d in dirs:
            dr, dc = _DELTA[d]
            nr, nc = r + dr, c + dc
            if grid.in_bounds(nr, nc) and not visited[nr, nc]:
                grid.remove_wall(r, c, d)
                dfs(nr, nc)         # recurse into the new cell

    dfs(0, 0)
    return grid


def generate_prim(rows: int, cols: int, rng: np.random.Generator) -> MazeGrid:
    """
    Randomized Prim's Algorithm.
    Grows the maze like a crystal from one seed cell:
      1. Start with one cell "in the maze". Add its walls to a frontier list.
      2. Pick a random wall from the frontier.
      3. If the cell on the other side is NOT yet in the maze, knock the wall
         down, add that cell, add ITS walls to the frontier.
      4. Repeat until frontier is empty.
    Result: wider, more branchy mazes with many short dead-ends.
    """
    grid = MazeGrid(rows, cols)
    in_maze = np.zeros((rows, cols), dtype=bool)
    in_maze[0, 0] = True

    # frontier: list of (from_r, from_c, direction, to_r, to_c)
    frontier = [(0, 0, d, 0+_DELTA[d][0], 0+_DELTA[d][1])
                for d in Dir if grid.in_bounds(_DELTA[d][0], _DELTA[d][1])]

    while frontier:
        idx = int(rng.integers(len(frontier)))
        r, c, d, nr, nc = frontier[idx]
        frontier.pop(idx)
        if not in_maze[nr, nc]:
            grid.remove_wall(r, c, d)
            in_maze[nr, nc] = True
            for nnr, nnc, nd in grid.neighbours(nr, nc):
                if not in_maze[nnr, nnc]:
                    frontier.append((nr, nc, nd, nnr, nnc))

    return grid


def generate_random_rooms(rows: int, cols: int, rng: np.random.Generator,
                           n_rooms: int = 4) -> MazeGrid:
    """
    Random Rooms: DFS maze + open rectangular rooms carved into it.
    The DFS gives guaranteed connectivity; the rooms create open areas.
    Result: feels like a dungeon — corridors connecting open chambers.
    """
    grid = generate_dfs(rows, cols, rng)   # start with a solvable maze
    for _ in range(n_rooms):
        rh = int(rng.integers(2, max(3, rows // 3)))
        rw = int(rng.integers(2, max(3, cols // 3)))
        r0 = int(rng.integers(0, rows - rh))
        c0 = int(rng.integers(0, cols - rw))
        # Remove all internal walls inside the room rectangle
        for r in range(r0, r0 + rh):
            for c in range(c0, c0 + rw):
                for d in [Dir.E, Dir.S]:
                    if grid.in_bounds(r + _DELTA[d][0], c + _DELTA[d][1]):
                        grid.remove_wall(r, c, d)
    return grid


# Dictionary so we can look up generators by name string
GENERATORS = {
    "dfs":   generate_dfs,
    "prim":  generate_prim,
    "rooms": generate_random_rooms,
}


# ===========================================================================
# PART 4 — THE ENVIRONMENT (MazeEnv)
# ===========================================================================
# This is the core Gymnasium environment class.
# Gymnasium is the standard RL library interface — every environment must
# implement: reset(), step(), render(), close()
#
# The agent interacts with it in a loop:
#   obs, info = env.reset()
#   while not done:
#       action = agent.pick_action(obs)         # agent decides what to do
#       obs, reward, terminated, truncated, info = env.step(action)
#
# The environment tells the agent:
#   obs        — what the agent can SEE (position, or image, etc.)
#   reward     — how good or bad the last action was (signal to learn from)
#   terminated — True if the episode ended naturally (reached goal)
#   truncated  — True if we hit the max_steps limit (time ran out)
#   info       — extra debug data (not used for learning)

class MazeEnv(gym.Env):
    """
    2-D Maze Environment — fully Gymnasium compatible.

    Parameters
    ----------
    maze_size    : (rows, cols) of the maze grid
    maze_type    : which generator to use: "dfs", "prim", "rooms"
    obs_type     : what the agent observes:
                     "discrete" → [agent_row, agent_col, goal_row, goal_col]
                     "local"    → 5×5 window of wall bitmasks around agent
                     "image"    → full binary pixel map (H×W×1)
    reward_type  : how reward is computed:
                     "sparse"    → 0 every step, +1 only at goal
                     "dense"     → +1 if closer to goal, -1 if farther
                     "potential" → mathematically unbiased shaping
    max_steps    : episode ends (truncated) after this many steps
    n_goals      : number of waypoints agent must visit in order
    local_window : size of local observation window (obs_type="local" only)
    render_mode  : "human", "rgb_array", or None
    seed         : fixed seed for reproducible mazes
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        maze_size: Tuple[int, int] = (10, 10),
        maze_type: str = "dfs",
        obs_type: str = "discrete",
        reward_type: str = "dense",
        max_steps: int = 500,
        n_goals: int = 1,
        local_window: int = 5,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.maze_rows, self.maze_cols = maze_size
        self.maze_type  = maze_type
        self.obs_type   = obs_type
        self.reward_type = reward_type
        self.max_steps  = max_steps
        self.n_goals    = n_goals
        self.local_window = local_window
        self.render_mode  = render_mode

        assert maze_type in GENERATORS, f"Unknown maze_type '{maze_type}'"

        # ------------------------------------------------------------------
        # ACTION SPACE
        # The agent can take 4 actions: 0=N, 1=E, 2=S, 3=W
        # Discrete(4) tells Gymnasium "there are 4 possible actions".
        # ------------------------------------------------------------------
        self.action_space = spaces.Discrete(4)
        self._action_to_dir = [Dir.N, Dir.E, Dir.S, Dir.W]

        # ------------------------------------------------------------------
        # OBSERVATION SPACE
        # Tells Gymnasium what shape and range the observations will be.
        # The agent's neural network (or Q-table) will receive observations
        # in this format at every step.
        # ------------------------------------------------------------------
        if obs_type == "discrete":
            # 4 integers: [agent_row, agent_col, goal_row, goal_col]
            low  = np.array([0, 0, 0, 0], dtype=np.int32)
            high = np.array([self.maze_rows-1, self.maze_cols-1,
                             self.maze_rows-1, self.maze_cols-1], dtype=np.int32)
            self.observation_space = spaces.Box(low, high, dtype=np.int32)

        elif obs_type == "local":
            # A 5×5 window of cells centred on the agent.
            # Each cell has 4 wall flags (N/E/S/W) → shape (5, 5, 4)
            w = local_window
            self.observation_space = spaces.Box(
                low=0, high=1, shape=(w, w, 4), dtype=np.uint8)

        elif obs_type == "image":
            # Full binary pixel map of the maze.
            # Size: (2*rows+1) × (2*cols+1) × 1 channel
            H = 2 * self.maze_rows + 1
            W = 2 * self.maze_cols + 1
            self.observation_space = spaces.Box(
                low=0, high=1, shape=(H, W, 1), dtype=np.uint8)
        else:
            raise ValueError(f"Unknown obs_type '{obs_type}'")

        # Internal state — all set properly in reset()
        self._rng:        Optional[np.random.Generator] = None
        self._grid:       Optional[MazeGrid] = None
        self._agent_pos:  Tuple[int, int] = (0, 0)
        self._goals:      List[Tuple[int, int]] = []
        self._goal_idx:   int = 0
        self._step_count: int = 0

        # Rendering handles
        self._fig = None
        self._ax  = None

        if seed is not None:
            self._rng = np.random.default_rng(seed)

    # =======================================================================
    # reset() — start a new episode
    # =======================================================================
    # Called at the beginning of every episode.
    # Must return: (observation, info_dict)
    # Generates a fresh maze, places the agent at (0,0), places goals.
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)   # Gymnasium bookkeeping

        # Set up the random number generator
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        elif self._rng is None:
            self._rng = np.random.default_rng()

        # Generate a fresh maze
        gen_fn = GENERATORS[self.maze_type]
        self._grid = gen_fn(self.maze_rows, self.maze_cols, self._rng)

        # Agent always starts at top-left corner
        self._agent_pos = (0, 0)

        # Place goals
        # For 1 goal: always bottom-right corner (most distant from start)
        # For N goals: N-1 random intermediates + bottom-right as final
        all_cells = [(r, c)
                     for r in range(self.maze_rows)
                     for c in range(self.maze_cols)
                     if (r, c) != (0, 0)]
        rng_list = list(all_cells)
        self._rng.shuffle(rng_list)

        if self.n_goals == 1:
            self._goals = [(self.maze_rows - 1, self.maze_cols - 1)]
        else:
            chosen = rng_list[:self.n_goals - 1]
            chosen.append((self.maze_rows - 1, self.maze_cols - 1))
            self._goals = chosen

        self._goal_idx   = 0
        self._step_count = 0

        return self._get_obs(), self._get_info()

    # =======================================================================
    # step(action) — the agent takes one action
    # =======================================================================
    # Called every timestep. The agent picks an action (0-3), we:
    #   1. Try to move the agent in that direction
    #   2. Compute the reward
    #   3. Check if the goal was reached (terminated)
    #   4. Check if we ran out of steps (truncated)
    # Returns: (obs, reward, terminated, truncated, info)
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        assert self._grid is not None, "Call reset() before step()."

        d = self._action_to_dir[int(action)]
        prev_r, prev_c = self._agent_pos

        # Move the agent (only if no wall is blocking)
        if self._grid.can_move(prev_r, prev_c, d):
            dr, dc = _DELTA[d]
            self._agent_pos = (prev_r + dr, prev_c + dc)
        # If movement is blocked, agent stays in place
        # (this is intentional — hitting walls is a valid action result)

        # Compute reward before checking goal (needs prev_pos)
        reward = self._compute_reward(prev_r, prev_c)

        # Check if agent reached the current goal
        terminated = False
        current_goal = self._goals[self._goal_idx]
        if self._agent_pos == current_goal:
            self._goal_idx += 1
            if self._goal_idx >= len(self._goals):
                terminated = True      # all goals reached → episode ends
                reward += 1.0          # bonus reward for completing the maze

        self._step_count += 1
        truncated = (not terminated) and (self._step_count >= self.max_steps)

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    # =======================================================================
    # render() — visualise the current state
    # =======================================================================
    def render(self):
        if self.render_mode is None:
            return

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        img = self._render_image()

        if self.render_mode == "rgb_array":
            return img

        if self._fig is None:
            self._fig, self._ax = plt.subplots(figsize=(6, 6))
            plt.ion()

        self._ax.clear()
        self._ax.imshow(img)
        self._ax.axis("off")
        self._ax.set_title(
            f"Step {self._step_count} | Goal {self._goal_idx}/{len(self._goals)}")
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    def close(self):
        if self._fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
            self._fig = None

    # =======================================================================
    # INTERNAL HELPERS
    # =======================================================================

    def _get_obs(self) -> np.ndarray:
        """Build the observation array that the agent will receive."""
        r, c = self._agent_pos
        goal_idx = min(self._goal_idx, len(self._goals) - 1)
        gr, gc   = self._goals[goal_idx]

        if self.obs_type == "discrete":
            # Simple 4-number vector: where am I, where is the goal?
            return np.array([r, c, gr, gc], dtype=np.int32)

        elif self.obs_type == "local":
            # 5×5 window of wall data centered on the agent.
            # The agent can "see" what walls exist in nearby cells.
            w    = self.local_window
            half = w // 2
            obs  = np.zeros((w, w, 4), dtype=np.uint8)
            for i in range(w):
                for j in range(w):
                    mr, mc = r - half + i, c - half + j
                    if self._grid.in_bounds(mr, mc):
                        obs[i, j] = self._grid.walls[mr, mc].astype(np.uint8)
                    else:
                        obs[i, j] = 1   # out-of-bounds = solid wall
            return obs

        else:  # image
            # Full pixel map of the maze with a single channel
            binary = self._grid.to_binary_map()
            return binary[:, :, np.newaxis].astype(np.uint8)

    def _get_info(self) -> Dict:
        """Extra information returned with each step (for debugging, not learning)."""
        r, c   = self._agent_pos
        goal_idx = min(self._goal_idx, len(self._goals) - 1)
        gr, gc = self._goals[goal_idx]
        return {
            "agent_pos":    (r, c),
            "current_goal": (gr, gc),
            "goal_idx":     self._goal_idx,
            "steps":        self._step_count,
            "manhattan":    abs(r - gr) + abs(c - gc),
        }

    def _compute_reward(self, prev_r: int, prev_c: int) -> float:
        """
        Three reward strategies:

        sparse   — signal only at the goal. Hard to learn from because
                   the agent must accidentally find the goal first.

        dense    — reward every step based on distance change.
                   +1 if agent moved closer, -1 if farther away.
                   Easier to learn but can mislead in complex mazes.

        potential — mathematically principled shaping using γΦ(s') - Φ(s).
                   Equivalent to dense but with theoretical guarantees that
                   the optimal policy is unchanged.
        """
        goal_idx = min(self._goal_idx, len(self._goals) - 1)
        gr, gc   = self._goals[goal_idx]
        r, c     = self._agent_pos

        if self.reward_type == "sparse":
            return 0.0

        elif self.reward_type == "dense":
            prev_dist = abs(prev_r - gr) + abs(prev_c - gc)
            curr_dist = abs(r - gr)      + abs(c - gc)
            return float(prev_dist - curr_dist)   # positive = got closer

        elif self.reward_type == "potential":
            norm     = self.maze_rows + self.maze_cols
            prev_phi = -(abs(prev_r - gr) + abs(prev_c - gc)) / norm
            curr_phi = -(abs(r - gr)      + abs(c - gc))      / norm
            gamma    = 0.99
            return float(gamma * curr_phi - prev_phi)

        return 0.0

    def _render_image(self) -> np.ndarray:
        """Build an RGB image of the current maze state for display."""
        binary = self._grid.to_binary_map()
        H, W   = binary.shape
        scale  = 20   # each maze pixel becomes a 20×20 block
        img    = np.zeros((H * scale, W * scale, 3), dtype=np.uint8)

        # Paint wall pixels dark, open pixels light
        for r in range(H):
            for c in range(W):
                color = (30, 30, 30) if binary[r, c] else (230, 230, 230)
                img[r*scale:(r+1)*scale, c*scale:(c+1)*scale] = color

        def fill_cell(row, col, rgb):
            """Paint a single maze cell with a given colour."""
            pr = (2*row+1) * scale
            pc = (2*col+1) * scale
            img[pr:pr+scale, pc:pc+scale] = rgb

        # Draw goals: intermediate goals green, final goal red
        for idx, (gr, gc) in enumerate(self._goals):
            color = (220, 50, 50) if idx == len(self._goals)-1 else (50, 180, 50)
            fill_cell(gr, gc, color)

        # Draw agent in blue
        ar, ac = self._agent_pos
        fill_cell(ar, ac, (50, 100, 220))

        return img

    def get_maze_grid(self) -> MazeGrid:
        """Return a copy of the maze grid (useful for BFS oracle)."""
        assert self._grid is not None, "Call reset() first."
        return self._grid.clone()