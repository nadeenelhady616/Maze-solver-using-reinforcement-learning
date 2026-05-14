from __future__ import annotations

import os
import pickle
import sys
import time

import pygame

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

for p in (_HERE, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from maze_env import MazeEnv
from q_learning import load_model


CELL = 60
MARGIN = 20

WHITE = (230, 230, 230)
BLACK = (25, 25, 25)
BLUE = (40, 120, 255)
RED = (220, 60, 60)
GREEN = (60, 200, 80)


def draw(screen, env, agent, goal):

    rows = env.maze_rows
    cols = env.maze_cols
    screen.fill(BLACK)
    grid = env.get_maze_grid()

    for r in range(rows):
        for c in range(cols):
            x = MARGIN + c * CELL
            y = MARGIN + r * CELL

            pygame.draw.rect(
                screen,
                WHITE,
                (x, y, CELL, CELL),
            )

            walls = grid.walls[r, c]

            if walls[0]:
                pygame.draw.line(screen, BLACK, (x, y), (x + CELL, y), 4)

            if walls[1]:
                pygame.draw.line(
                    screen,
                    BLACK,
                    (x + CELL, y),
                    (x + CELL, y + CELL),
                    4,
                )

            if walls[2]:
                pygame.draw.line(
                    screen,
                    BLACK,
                    (x, y + CELL),
                    (x + CELL, y + CELL),
                    4,
                )

            if walls[3]:
                pygame.draw.line(
                    screen,
                    BLACK,
                    (x, y),
                    (x, y + CELL),
                    4,
                )

    gr, gc = goal

    pygame.draw.circle(
        screen,
        RED,
        (
            MARGIN + gc * CELL + CELL // 2,
            MARGIN + gr * CELL + CELL // 2,
        ),
        CELL // 4,
    )

    ar, ac = agent

    pygame.draw.circle(
        screen,
        BLUE,
        (
            MARGIN + ac * CELL + CELL // 2,
            MARGIN + ar * CELL + CELL // 2,
        ),
        CELL // 3,
    )

    pygame.display.flip()

def main():
    model_path = os.path.join(_HERE, "results", "q_table.pkl")
    q, cfg = load_model(model_path)

    env = MazeEnv(
        maze_size=cfg["maze_size"],
        maze_type=cfg["maze_type"],
        obs_type="discrete",
        reward_type=cfg["reward_type"],
        max_steps=cfg["max_steps"],
    )

    pygame.init()

    rows, cols = cfg["maze_size"]

    screen = pygame.display.set_mode(
        (
            cols * CELL + 2 * MARGIN,
            rows * CELL + 2 * MARGIN,
        )
    )

    pygame.display.set_caption("Q-Learning Maze Solver")

    obs, info = env.reset(seed=cfg["maze_seed"])

    running = True

    while running:

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

        action = q.best_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        draw(
            screen,
            env,
            info["agent_pos"],
            info["current_goal"],
        )

        if terminated:
            print("GOAL REACHED")
            time.sleep(2)

            obs, info = env.reset(seed=cfg["maze_seed"])

        if truncated:
            print("FAILED")
            time.sleep(1)

            obs, info = env.reset(seed=cfg["maze_seed"])
        time.sleep(0.12)
    pygame.quit()
    env.close()


if __name__ == "__main__":
    main()