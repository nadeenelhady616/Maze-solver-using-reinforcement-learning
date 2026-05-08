"""
oracle.py
---------
BFS-based oracle agent — finds the SHORTEST possible path through any maze.

Why do we need this?
  - It gives us a PERFECT BASELINE: the minimum steps any agent could take.
  - We can compare our trained RL agent against it: how close to optimal?
  - It's used in tests to verify the maze is actually solvable.
  - It can generate training demonstrations for imitation learning.

How BFS works:
  Start at the source cell. Explore all neighbours level by level
  (1 step away, then 2 steps, then 3...). The first time you reach
  the goal, you're guaranteed to have found the shortest path.
"""

from __future__ import annotations

from collections import deque
from typing import List, Optional, Tuple

import numpy as np

from maze_env import Dir, MazeGrid, _DELTA


def bfs(
    grid: MazeGrid,
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> Optional[List[int]]:
    """
    Find the shortest path from start to goal using Breadth-First Search.

    Returns a list of actions (integers 0-3 matching the env action space),
    or None if no path exists (should never happen in a perfect maze).

    Example: [2, 1, 2, 2, 1] means go S, E, S, S, E
    """
    if start == goal:
        return []

    # Map action index → direction (matches MazeEnv action space)
    action_map = [Dir.N, Dir.E, Dir.S, Dir.W]   # 0, 1, 2, 3

    # BFS queue stores: (current_position, path_taken_so_far)
    queue: deque = deque()
    queue.append((start, []))
    visited = {start}

    while queue:
        (r, c), path = queue.popleft()
        for action_idx, d in enumerate(action_map):
            if grid.can_move(r, c, d):
                dr, dc = _DELTA[d]
                npos   = (r + dr, c + dc)
                if npos == goal:
                    return path + [action_idx]   # found it!
                if npos not in visited:
                    visited.add(npos)
                    queue.append((npos, path + [action_idx]))

    return None   # unreachable (shouldn't happen in a well-formed maze)


def compute_distance_map(
    grid: MazeGrid,
    goal: Tuple[int, int],
) -> np.ndarray:
    """
    Compute BFS distance from EVERY cell to the goal.
    Returns a (rows × cols) array where cell[r][c] = number of steps to goal.

    This is useful for:
      - Visualising the maze difficulty (heatmap)
      - Computing optimal potential-based rewards
      - Checking reachability of every cell
    """
    dist = np.full((grid.rows, grid.cols), fill_value=np.inf)
    dist[goal] = 0
    queue: deque = deque([goal])
    action_map = [Dir.N, Dir.E, Dir.S, Dir.W]

    while queue:
        r, c = queue.popleft()
        for d in action_map:
            if grid.can_move(r, c, d):
                dr, dc = _DELTA[d]
                nr, nc = r + dr, c + dc
                if np.isinf(dist[nr, nc]):
                    dist[nr, nc] = dist[r, c] + 1
                    queue.append((nr, nc))

    return dist


class OracleAgent:
    """
    A step-by-step agent that follows the BFS optimal path.
    Use it like any other agent:
        agent = OracleAgent()
        agent.reset(grid, start, goal)
        while not agent.done:
            action = agent.act()
            obs, r, term, trunc, info = env.step(action)
    """

    def __init__(self):
        self._plan: List[int] = []
        self._ptr:  int = 0

    def reset(self, grid: MazeGrid, start: Tuple[int, int], goal: Tuple[int, int]):
        """Compute the optimal path from start to goal for this grid."""
        path = bfs(grid, start, goal)
        self._plan = path if path is not None else []
        self._ptr  = 0

    def act(self, obs=None) -> int:
        """Return the next action in the plan."""
        if self._ptr < len(self._plan):
            action = self._plan[self._ptr]
            self._ptr += 1
            return action
        return 0   # plan exhausted — shouldn't happen if used correctly

    @property
    def done(self) -> bool:
        """True when all planned actions have been executed."""
        return self._ptr >= len(self._plan)