import pygame

import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces

import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces


class DroneNavigationEnv(gym.Env):
    """
    A hybrid 2D grid environment with Pygame rendering and coordinate-based observations.
    The agent moves in a grid to reach a goal with minimal steps.
    """

    def __init__(self, grid_size=5, cell_size=100):
        super(DroneNavigationEnv, self).__init__()

        self.grid_size = grid_size
        self.cell_size = cell_size
        self.window_size = grid_size * cell_size

        self.start_pos = np.array([1, 1], dtype=np.int32)
        self.goal_pos = np.array([3, 2], dtype=np.int32)
        self.agent_pos = self.start_pos.copy()

        self.action_space = spaces.Discrete(4)  # up, down, left, right
        self.observation_space = spaces.Box(low=0, high=grid_size - 1, shape=(2,), dtype=np.int32)

        # Colors
        self.bg_color = (255, 255, 255)
        self.agent_color = (255, 0, 0)
        self.goal_color = (0, 255, 0)
        self.grid_color = (200, 200, 200)

        # Pygame initialization
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("Hybrid Grid Environment")
        self.clock = pygame.time.Clock()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = self.start_pos.copy()
        return self.agent_pos.copy(), {}

    def step(self, action):
        # Movement logic
        if action == 0 and self.agent_pos[1] > 0:  # up
            self.agent_pos[1] -= 1
        elif action == 1 and self.agent_pos[1] < self.grid_size - 1:  # down
            self.agent_pos[1] += 1
        elif action == 2 and self.agent_pos[0] > 0:  # left
            self.agent_pos[0] -= 1
        elif action == 3 and self.agent_pos[0] < self.grid_size - 1:  # right
            self.agent_pos[0] += 1

        done = np.array_equal(self.agent_pos, self.goal_pos)
        reward = 10.0 if done else -0.1

        distance_to_goal = np.linalg.norm(self.goal_pos - self.agent_pos)
        info = {"distance_to_goal": distance_to_goal}

        return self.agent_pos.copy(), reward, done, False, info

    def render(self):
        self.screen.fill(self.bg_color)

        # Draw grid
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                rect = pygame.Rect(j * self.cell_size, i * self.cell_size,
                                   self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, self.grid_color, rect, 1)

        # Draw goal
        gx, gy = self.goal_pos
        pygame.draw.rect(
            self.screen, self.goal_color,
            pygame.Rect(gx * self.cell_size, gy * self.cell_size,
                        self.cell_size, self.cell_size)
        )

        # Draw agent
        ax, ay = self.agent_pos
        padding = 10
        pygame.draw.rect(
            self.screen, self.agent_color,
            pygame.Rect(ax * self.cell_size + padding, ay * self.cell_size + padding,
                        self.cell_size - 2 * padding, self.cell_size - 2 * padding)
        )

        pygame.display.flip()
        self.clock.tick(10)

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    env = DroneNavigationEnv()
    obs, _ = env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        env.render()
        print(f"Obs: {obs}, Reward: {reward}, Done: {done}, Info: {info}")
        if done:
            print("Reached the goal!")
            break
    env.close()



"""
class DroneNavigationEnv(gym.Env):
    #
    A discrete 2D grid environment with pygame rendering for a drone to reach a goal.

    The drone moves in a 5x5 grid. It can move up, down, left, or right. The environment
    terminates when the drone reaches the goal.
    #

    def __init__(self, grid_size=5, cell_size=100):
        super(DroneNavigationEnv, self).__init__()

        # Grid and state
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.window_size = grid_size * cell_size

        self.start_pos = np.array([0, 0], dtype=np.int32)
        self.goal_pos = np.array([grid_size - 1, grid_size - 1], dtype=np.int32)
        self.drone_pos = self.start_pos.copy()

        # Gym spaces
        self.action_space = spaces.Discrete(4)  # up, down, left, right
        self.observation_space = spaces.Discrete(grid_size * grid_size)

        # Colors
        self.bg_color = (230, 230, 230)
        self.drone_color = (0, 102, 255)
        self.goal_color = (50, 168, 82)
        self.grid_color = (180, 180, 180)

        # Pygame init
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("Drone Navigation")
        self.clock = pygame.time.Clock()

    def reset(self):
        #Resets environment and returns starting observation.
        self.drone_pos = self.start_pos.copy()
        return self._get_obs()

    def step(self, action):
        #Executes action and returns (obs, reward, done, info).
        if action == 0 and self.drone_pos[0] > 0:      # up
            self.drone_pos[0] -= 1
        elif action == 1 and self.drone_pos[0] < self.grid_size - 1:  # down
            self.drone_pos[0] += 1
        elif action == 2 and self.drone_pos[1] > 0:     # left
            self.drone_pos[1] -= 1
        elif action == 3 and self.drone_pos[1] < self.grid_size - 1:  # right
            self.drone_pos[1] += 1

        done = np.array_equal(self.drone_pos, self.goal_pos)
        reward = 1.0 if done else -0.1
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        #Returns current position as a single integer (flattened index).
        return int(self.drone_pos[0] * self.grid_size + self.drone_pos[1])

    def render(self, mode='human'):
        #Renders the environment using pygame.
        self.screen.fill(self.bg_color)

        # Draw grid
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                rect = pygame.Rect(j * self.cell_size, i * self.cell_size,
                                   self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, self.grid_color, rect, 1)

        # Draw goal
        gx, gy = self.goal_pos[1], self.goal_pos[0]
        goal_rect = pygame.Rect(gx * self.cell_size, gy * self.cell_size,
                                self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, self.goal_color, goal_rect)

        # Draw drone
        dx, dy = self.drone_pos[1], self.drone_pos[0]
        drone_rect = pygame.Rect(dx * self.cell_size + 10, dy * self.cell_size + 10,
                                 self.cell_size - 20, self.cell_size - 20)
        pygame.draw.rect(self.screen, self.drone_color, drone_rect)

        pygame.display.flip()
        self.clock.tick(10)  # Limit to 10 FPS

    def close(self):
        #Closes the pygame window.
        pygame.quit()
"""