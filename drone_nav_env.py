# Import required libraries
import pygame  # For rendering
import numpy as np  # For matrix and numerical operations
import gymnasium as gym  # For creating custom RL environments
from gymnasium import spaces  # For defining action and observation spaces
import os  # Not used here, but may be useful for file paths

# Define the custom drone navigation environment
class DroneNavigationEnv(gym.Env):
    """Custom Gym environment for drone navigation over terrain with full image support."""
    
    def __init__(self, grid_size=(10, 15), cell_size=100, fixed_seed=42):
        super(DroneNavigationEnv, self).__init__()

        # Grid and cell configuration
        self.grid_size = grid_size  # Environment size in (rows, columns)
        self.cell_size = cell_size  # Size of each cell in pixels
        self.window_size = (grid_size[1] * cell_size, grid_size[0] * cell_size)  # Window size in pixels (width, height)

        # Seed for reproducibility
        self.fixed_seed = fixed_seed
        np.random.seed(self.fixed_seed)  # Ensures terrain generation is repeatable

        # Generate terrain: 1 = valley (walkable), 0 = mountain (higher penalty)
        # self.terrain = np.random.choice([0, 1], size=self.grid_size, p=[0.3, 0.7])
        
        # All hell_state_coordonates: [(1,0), (2, 0), (3, 0), (1, 1), (4, 1), (4, 2), (2, 3), (0, 4), (2, 4), (3, 4)]
        self.terrain = np.ones(self.grid_size, dtype=np.int32)  # Default all valleys

        # Set specific mountains
        mountain_coords = [(0, 2), (1, 4), (2, 1), (3, 3), (4, 0)]
        for r, c in mountain_coords:
            self.terrain[r, c] = 0


        # Define agent and goal positions
        self.start_pos = np.array([0, 0])  # Start at top-left corner
        self.goal_pos = np.array([grid_size[0]-1, grid_size[1]-1])  # Goal at bottom-right
        self.agent_pos = self.start_pos.copy()  # Initialize agent position

        print(f"Start: {self.start_pos}, Goal: {self.goal_pos}")  # Print positions for debugging

        # Define the action and observation space
        self.action_space = spaces.Discrete(4)  # Actions: 0=left, 1=right, 2=up, 3=down
        self.observation_space = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([grid_size[0]-1, grid_size[1]-1]),
            dtype=np.int32
        )

        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode(self.window_size)  # Create game window
        pygame.display.set_caption("Drone Navigation Environment")
        self.clock = pygame.time.Clock()  # For controlling frame rate

        # Load terrain and drone images
        self.load_images()

        # Set goal color (used for drawing goal cell)
        self.goal_color = (255, 0, 0)  # Red
    
    def load_images(self):
        """Load and scale all four images needed for rendering."""
        try:
            # Load terrain images
            self.mountain_img = pygame.image.load("/home/faruk/Desktop/HDD_Projects/DRL_AutonomousDrone/env_images/mountain_2.png")
            self.valley_img = pygame.image.load("/home/faruk/Desktop/HDD_Projects/DRL_AutonomousDrone/env_images/flat_terrain.png")

            # Load drone and fire/corner images
            self.corner_img = pygame.image.load("/home/faruk/Desktop/HDD_Projects/DRL_AutonomousDrone/env_images/fire.png")
            self.drone_img = pygame.image.load("/home/faruk/Desktop/HDD_Projects/DRL_AutonomousDrone/env_images/drone_3.png")

            # Resize images to match cell size
            self.mountain_img = pygame.transform.scale(self.mountain_img, (self.cell_size, self.cell_size))
            self.valley_img = pygame.transform.scale(self.valley_img, (self.cell_size, self.cell_size))
            self.drone_img = pygame.transform.scale(self.drone_img, (self.cell_size-20, self.cell_size-20))  # Slightly smaller for padding

            # Prepare fire image with red border
            self.corner_img = pygame.transform.scale(self.corner_img, (100, 100))
            corner_with_border = pygame.Surface((100, 100), pygame.SRCALPHA)
            corner_with_border.blit(self.corner_img, (0, 0))
            pygame.draw.rect(corner_with_border, (255, 0, 0), (0, 0, 100, 100), 5)  # Draw red border
            self.corner_img = corner_with_border

        except pygame.error as e:
            # If image loading fails, fallback to color surfaces
            print(f"Error loading images: {e}")
            self.create_fallback_surfaces()
    
    def create_fallback_surfaces(self):
        """Create simple colored surfaces if image loading fails."""
        # Create gray surface for mountains
        self.mountain_img = pygame.Surface((self.cell_size, self.cell_size))
        self.mountain_img.fill((139, 137, 137))  # Gray

        # Create green surface for valleys
        self.valley_img = pygame.Surface((self.cell_size, self.cell_size))
        self.valley_img.fill((34, 139, 34))  # Green

        # Create blue surface for drone
        self.drone_img = pygame.Surface((self.cell_size-20, self.cell_size-20))
        self.drone_img.fill((0, 0, 255))  # Blue

        # Create orange surface with red border for fire/corner
        self.corner_img = pygame.Surface((100, 100))
        self.corner_img.fill((255, 165, 0))  # Orange
        pygame.draw.rect(self.corner_img, (255, 0, 0), (0, 0, 100, 100), 5)
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        # Regenerate terrain with same seed (ensures consistency)
        #np.random.seed(self.fixed_seed)
        #self.terrain = np.random.choice([0, 1], size=self.grid_size, p=[0.3, 0.7])

        # Reset agent to start position
        self.agent_pos = self.start_pos.copy()
        return self.agent_pos.copy(), {}
    
    def step(self, action):
        """Execute one time step within the environment."""
        new_pos = self.agent_pos.copy()  # Copy current position

        # Apply action
        if action == 0:  # Move left
            new_pos[1] = max(0, new_pos[1] - 1)
        elif action == 1:  # Move right
            new_pos[1] = min(self.grid_size[1] - 1, new_pos[1] + 1)
        elif action == 2:  # Move up
            new_pos[0] = max(0, new_pos[0] - 1)
        elif action == 3:  # Move down
            new_pos[0] = min(self.grid_size[0] - 1, new_pos[0] + 1)

        # Animate movement if position has changed
        if not np.array_equal(new_pos, self.agent_pos):
            self.animate_agent_move(self.agent_pos, new_pos)

        # Update agent position
        self.agent_pos = new_pos

        # Determine reward and done condition
        terrain_type = self.terrain[new_pos[0], new_pos[1]]
        done = np.array_equal(self.agent_pos, self.goal_pos)

        if done:
            reward = 10.0  # Reward for reaching goal
        elif terrain_type == 0:
            reward = -1.0  # Penalty for mountain
        else:
            reward = -0.1  # Small penalty for valley (encourages faster path)

        # Return observation, reward, done, truncated, info
        info = {
            "distance_to_goal": np.linalg.norm(self.goal_pos - self.agent_pos),
            "position": self.agent_pos.copy(),
            "terrain": "mountain" if terrain_type == 0 else "valley"
        }

        return self.agent_pos.copy(), reward, done, False, info
    
    def animate_agent_move(self, start_pos, end_pos, duration=0.2, steps=1):
        """Animate smooth movement between positions (1 step)."""
        start_row, start_col = start_pos
        end_row, end_col = end_pos

        for step in range(1, steps + 1):
            t = step / steps
            current_row = start_row + (end_row - start_row) * t
            current_col = start_col + (end_col - start_col) * t

            # Render intermediate frame
            self.render_frame(current_row, current_col)
            pygame.display.flip()
            self.clock.tick(30)  # Fixed to 30 FPS
    
    def render_frame(self, drone_row=None, drone_col=None):
        """Render a single frame of the environment."""
        # Use current position if not provided
        if drone_row is None or drone_col is None:
            drone_row, drone_col = self.agent_pos

        # Draw terrain grid
        for row in range(self.grid_size[0]):
            for col in range(self.grid_size[1]):
                terrain_type = self.terrain[row, col]
                img = self.valley_img if terrain_type == 1 else self.mountain_img
                self.screen.blit(img, (col * self.cell_size, row * self.cell_size))

        # Draw goal cell as red rectangle
        goal_row, goal_col = self.goal_pos
        pygame.draw.rect(
            self.screen,
            self.goal_color,
            pygame.Rect(
                goal_col * self.cell_size,
                goal_row * self.cell_size,
                self.cell_size,
                self.cell_size
            ),
            width=5
        )

        # Draw drone image at current position (with padding)
        drone_x = drone_col * self.cell_size + 10
        drone_y = drone_row * self.cell_size + 10
        self.screen.blit(self.drone_img, (drone_x, drone_y))

        # Draw fire/corner image at bottom-right corner
        self.screen.blit(
            self.corner_img,
            (self.window_size[0] - 100, self.window_size[1] - 100)
        )

        # Update the display
        pygame.display.flip()
    
    def render(self):
        """Render the current state (alias for render_frame)."""
        self.render_frame()
    
    def close(self):
        """Close the environment and cleanup."""
        pygame.quit()

# Main test block to run the environment manually
if __name__ == "__main__":
    # Initialize Pygame and get screen size
    pygame.init()
    info = pygame.display.Info()
    screen_height = info.current_h

    # Define smaller grid and resize cells to fit on screen
    grid_size = (5, 5)
    cell_size = min(100, int(screen_height * 0.8 / grid_size[0]))

    # Create environment
    env = DroneNavigationEnv(grid_size=grid_size, cell_size=cell_size)
    obs, _ = env.reset()

    # Main game loop
    running = True
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False  # Exit on window close

        # Take random actions
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # Render frame
        env.render()
        print(f"Position: {obs}, Reward: {reward}, Done: {terminated}")

        # Exit if goal reached
        if terminated:
            print("Goal reached!")
            pygame.time.delay(2000)
            running = False

        clock.tick(10)  # Run at 10 FPS

    # Close the environment
    env.close()
