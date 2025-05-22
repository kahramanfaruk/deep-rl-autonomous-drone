import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


class PadmEnv(gym.Env):
    """
    Custom grid-based environment for reinforcement learning.

    This environment simulates a 2D grid where an agent starts at a predefined position
    and attempts to reach a goal position. The agent can move up, down, left, or right
    in the grid, and the task is to reach the goal position.

    Attributes
    ----------
    grid_size : int
        The size of the grid (default is 5x5).
    agent_state : np.ndarray
        The current position of the agent on the grid (2D coordinates).
    goal_state : np.ndarray
        The position of the goal on the grid (2D coordinates).
    action_space : gym.spaces.Discrete
        The action space defining 4 discrete actions: up, down, left, right.
    observation_space : gym.spaces.Box
        The observation space defining a continuous 2D space for the agent's position.
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        Used for rendering and visualizing the environment.

    Methods
    -------
    __init__(grid_size=5)
        Initializes the environment with the given grid size.
    reset()
        Resets the environment to the initial state.
    step(action)
        Takes an action and updates the environment's state, returning the new state, reward, done flag, and info.
    render()
        Renders the current state of the environment.
    close()
        Closes the visualization window.
    """

    def __init__(self, grid_size=5):
        """
        Initializes the environment.

        Parameters
        ----------
        grid_size : int, optional
            The size of the grid (default is 5).
        """
        super().__init__()
        self.grid_size = grid_size
        self.agent_state = np.array([1, 1])
        self.goal_state = np.array([3, 2])
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0, high=self.grid_size, shape=(2,))
        self.fig, self.ax = plt.subplots()
        plt.show(block=False)

    def reset(self):
        """
        Resets the environment to the initial state.

        Returns
        -------
        np.ndarray
            The initial state of the environment (the agent's starting position).
        """
        self.agent_state = np.array([1, 1])
        return self.agent_state

    def step(self, action):
        """
        Takes an action and updates the environment's state.

        Parameters
        ----------
        action : int
            The action taken by the agent (0: up, 1: down, 2: left, 3: right).

        Returns
        -------
        state : np.ndarray
            The new state after the action (the agent's new position).
        reward : int
            The reward for taking the action (10 if the goal is reached, otherwise 0).
        done : bool
            A flag indicating whether the episode is over (True if the goal is reached).
        info : dict
            Additional information about the environment, such as the distance to the goal.
        """
        if action == 0 and self.agent_state[1] < self.grid_size:  # up
            self.agent_state[1] += 1
        elif action == 1 and self.agent_state[1] > 0:  # down
            self.agent_state[1] -= 1
        elif action == 2 and self.agent_state[0] > 0:  # left
            self.agent_state[0] -= 1
        elif action == 3 and self.agent_state[0] < self.grid_size:  # right
            self.agent_state[0] += 1

        reward = 0
        done = np.array_equal(self.agent_state, self.goal_state)
        if done:
            reward = 10

        # Calculate the Euclidean distance to the goal
        distance_to_goal = np.linalg.norm(self.goal_state - self.agent_state)

        info = {"Distance to Goal": distance_to_goal}

        return self.agent_state, reward, done, info

    def render(self):
        """
        Renders the current state of the environment.

        This method displays the agent's position and the goal on a 2D grid.
        """
        self.ax.clear()
        self.ax.plot(self.agent_state[0], self.agent_state[1], "ro")  # Agent as a red dot
        self.ax.plot(self.goal_state[0], self.goal_state[1], "g+")  # Goal as a green cross
        self.ax.set_xlim(-1, self.grid_size)
        self.ax.set_ylim(-1, self.grid_size)
        self.ax.set_aspect("equal")
        plt.pause(0.1)

    def close(self):
        """
        Closes the visualization window.

        This method is called when the environment is done.
        """
        plt.close()


if __name__ == "__main__":
    env = PadmEnv()
    state = env.reset()
    for _ in range(500):
        action = env.action_space.sample()  # action = 0 or 1 or 2 or 3
        state, reward, done, info = env.step(action)
        env.render()
        print(f"State:{state}, Reward:{reward}, Done:{done}, Info:{info}")
        if done:
            print("I reached the goal")
            break
    env.close()
