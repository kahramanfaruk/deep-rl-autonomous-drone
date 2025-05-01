import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os


class DroneNavigationEnv(gym.Env):
    """
    Custom Gym environment for simulating drone navigation over a terrain.

    The environment consists of a grid where each cell can either be a valley or a mountain.
    The goal of the drone is to navigate through the grid from a start position to a goal position.
    The environment also includes terrain images and an animation of the drone's movement.
    """

    def __init__(self, grid_size=(10, 15), cell_size=100):
        """
        Initializes the environment.

        Parameters:
        -----------
        grid_size : tuple of int (default: (10, 15))
            The size of the grid in terms of rows and columns.
        cell_size : int (default: 100)
            The size of each grid cell in pixels for rendering.
        """
        super(DroneNavigationEnv, self).__init__()

        self.grid_size = grid_size  # Now a tuple (rows, columns)
        self.cell_size = cell_size
        self.window_size = (grid_size[1] * cell_size, grid_size[0] * cell_size)  # width, height

        # Example terrain for grid_size (10x15), customize as needed
        self.terrain = np.random.choice([0, 1], size=self.grid_size, p=[0.3, 0.7])

        self.start_pos = np.array([0, 0], dtype=np.int32)
        self.goal_pos = np.array([grid_size[0] - 1, grid_size[1] - 1], dtype=np.int32)
        self.agent_pos = self.start_pos.copy()

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=np.array(self.grid_size) - 1, shape=(2,), dtype=np.int32)

        # Pygame init
        pygame.init()
        self.screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption("Drone Terrain Navigation with Images")
        self.clock = pygame.time.Clock()

        # Load images
        self.mountain_img = pygame.image.load("/home/faruk/Desktop/DRL_AutonomousDrone/env_images/mountain_2.png")
        self.valley_img = pygame.image.load("/home/faruk/Desktop/DRL_AutonomousDrone/env_images/flat_terrain.png")
        self.corner_img = pygame.image.load("/home/faruk/Desktop/DRL_AutonomousDrone/env_images/fire.png")
        self.drone_img = pygame.image.load("/home/faruk/Desktop/DRL_AutonomousDrone/env_images/drone_3.png")

        # Resize images
        self.mountain_img = pygame.transform.scale(self.mountain_img, (cell_size, cell_size))
        self.valley_img = pygame.transform.scale(self.valley_img, (cell_size, cell_size))
        self.corner_img = pygame.transform.scale(self.corner_img, (100, 100))

        # Add red border to corner_img
        corner_with_border = pygame.Surface((100, 100), pygame.SRCALPHA)
        corner_with_border.blit(self.corner_img, (0, 0))
        pygame.draw.rect(corner_with_border, (255, 0, 0), (0, 0, 100, 100), 10)  # 5px red border size
        self.corner_img = corner_with_border

        self.drone_img = pygame.transform.scale(self.drone_img, (cell_size - 30, cell_size - 30))

        # Colors
        self.goal_color = (255, 0, 0)

    def animate_agent_move(self, start_pos, end_pos, duration=0.3, steps=10):
        """
        Animate the movement of the agent (drone) from the start position to the end position.

        Parameters:
        -----------
        start_pos : tuple of int
            The starting position of the agent (x, y).
        end_pos : tuple of int
            The ending position of the agent (x, y).
        duration : float (default: 0.3)
            Duration of the movement animation in seconds.
        steps : int (default: 10)
            Number of animation steps.

        Returns:
        --------
        None
        """
        sx, sy = start_pos
        ex, ey = end_pos

        for step in range(1, steps + 1):
            t = step / steps
            current_x = sx + (ex - sx) * t
            current_y = sy + (ey - sy) * t

            # Clear and redraw everything
            for y in range(self.grid_size[0]):
                for x in range(self.grid_size[1]):
                    img = self.valley_img if self.terrain[y, x] == 1 else self.mountain_img
                    self.screen.blit(img, (x * self.cell_size, y * self.cell_size))

            # Draw drone at interpolated position
            padding = 15
            self.screen.blit(
                self.drone_img,
                (current_x * self.cell_size + padding, current_y * self.cell_size + padding)
            )

            # Draw goal
            gx, gy = self.goal_pos
            pygame.draw.rect(
                self.screen, self.goal_color,
                pygame.Rect(gx * self.cell_size, gy * self.cell_size, self.cell_size, self.cell_size),
                width=8
            )

            # Draw overlay (now with red border)
            self.screen.blit(self.corner_img, (self.window_size[0] - 100, self.window_size[1] - 100))

            pygame.display.flip()
            self.clock.tick(1 / (duration / steps))

    def reset(self, seed=None, options=None):
        """
        Resets the environment to the initial state.

        Parameters:
        -----------
        seed : int or None, optional (default=None)
            Random seed for reproducibility.
        options : dict, optional (default=None)
            Additional options to configure the environment.

        Returns:
        --------
        obs : np.ndarray
            The initial observation of the environment.
        info : dict
            Additional information (empty dictionary here).
        """
        super().reset(seed=seed)
        self.agent_pos = self.start_pos.copy()
        return self.agent_pos.copy(), {}

    def step(self, action):
        """
        Takes an action and returns the next state, reward, and other information.

        Parameters:
        -----------
        action : int
            The action taken by the agent (0=left, 1=right, 2=up, 3=down).

        Returns:
        --------
        obs : np.ndarray
            The new position of the agent after the action.
        reward : float
            The reward after taking the action.
        done : bool
            Whether the agent has reached the goal.
        truncated : bool
            Whether the environment was truncated (unused in this implementation).
        info : dict
            Additional information such as the distance to the goal.
        """
        next_pos = self.agent_pos.copy()
        if action == 0 and self.agent_pos[1] > 0:
            next_pos[1] -= 1
        elif action == 1 and self.agent_pos[1] < self.grid_size[1] - 1:
            next_pos[1] += 1
        elif action == 2 and self.agent_pos[0] > 0:
            next_pos[0] -= 1
        elif action == 3 and self.agent_pos[0] < self.grid_size[0] - 1:
            next_pos[0] += 1

        if self.terrain[next_pos[1], next_pos[0]] == 1:
            self.animate_agent_move(self.agent_pos, next_pos)
            self.agent_pos = next_pos

        done = np.array_equal(self.agent_pos, self.goal_pos)
        reward = 10.0 if done else -0.1
        info = {"distance_to_goal": np.linalg.norm(self.goal_pos - self.agent_pos)}
        return self.agent_pos.copy(), reward, done, False, info

    def render(self):
        """
        Renders the current state of the environment.

        Returns:
        --------
        None
        """
        for y in range(self.grid_size[0]):
            for x in range(self.grid_size[1]):
                img = self.valley_img if self.terrain[y, x] == 1 else self.mountain_img
                self.screen.blit(img, (x * self.cell_size, y * self.cell_size))

        # Draw drone
        ax, ay = self.agent_pos
        padding = 15
        self.screen.blit(self.drone_img, (ax * self.cell_size + padding, ay * self.cell_size + padding))

        # Draw goal
        gx, gy = self.goal_pos
        pygame.draw.rect(
            self.screen, self.goal_color,
            pygame.Rect(gx * self.cell_size, gy * self.cell_size, self.cell_size, self.cell_size), width=8
        )

        # Overlay (now with red border)
        self.screen.blit(self.corner_img, (self.window_size[0] - 100, self.window_size[1] - 100))

        pygame.display.flip()
        self.clock.tick(10)

    def close(self):
        """
        Closes the environment and quits Pygame.

        Returns:
        --------
        None
        """
        pygame.quit()


if __name__ == "__main__":
    pygame.init()
    info = pygame.display.Info()
    screen_height = info.current_h  # typically 1080 on laptops

    grid_size = (10, 15)  # Now (rows, columns)
    usable_height = int(screen_height * 0.9)
    cell_size = usable_height // grid_size[0]

    env = DroneNavigationEnv(grid_size=grid_size, cell_size=cell_size)
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
