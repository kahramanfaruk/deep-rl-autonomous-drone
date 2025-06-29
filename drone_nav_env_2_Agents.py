import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random

class DroneNavigationEnv(gym.Env):
    """Custom Gym environment for drone navigation over terrain with full image support."""
    
    def __init__(self, grid_size=(10, 15), cell_size=100, fixed_seed=42):
        super(DroneNavigationEnv, self).__init__()
        
        # Environment setup
        self.grid_size = grid_size  # (rows, columns)
        self.cell_size = cell_size
        self.window_size = (grid_size[1] * cell_size, grid_size[0] * cell_size)  # (width, height)
        
        # Set a fixed random seed for reproducibility
        self.fixed_seed = fixed_seed
        np.random.seed(self.fixed_seed)  # This will make np.random.choice deterministic
        random.seed(self.fixed_seed)
        
        # Positions (using numpy arrays for easy manipulation)
        self.start_pos_1 = np.array([0, 0])  # First agent position (top-left)
        self.start_pos_2 = np.array([2, 2])  # Second agent position (2, 2)

        # Fixed goal position at (5, 5)
        self.goal_pos = np.array([5, 5])

        self.agent_pos_1 = self.start_pos_1.copy()
        self.agent_pos_2 = self.start_pos_2.copy()
        
        print(f"Start: {self.start_pos_1}, {self.start_pos_2}, Goal (Fire): {self.goal_pos}")
        
        # Generate terrain after agent positions are set
        self.generate_terrain()

        # Gym spaces
        self.action_space = spaces.Discrete(4)  # 0:left, 1:right, 2:up, 3:down
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0]),
            high=np.array([grid_size[0]-1, grid_size[1]-1, grid_size[0]-1, grid_size[1]-1]),
            dtype=np.int32
        )
        
        # Pygame initialization
        pygame.init()
        self.screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption("Drone Navigation Environment")
        self.clock = pygame.time.Clock()
        
        # Load all four images with error handling
        self.load_images()

    def generate_terrain(self):
        """Generate terrain with a clear path of valleys (1) from start to goal."""
        # Initialize terrain with all mountains (0)
        self.terrain = np.zeros(self.grid_size, dtype=int)
        
        # Create a valley path from start to goal
        self.create_valley_path(self.start_pos_1, self.goal_pos)
        
        # Optionally, randomly scatter some mountains and valleys (but make it deterministic)
        self.add_random_terrain()

    def create_valley_path(self, start, goal):
        """Create a clear path of valleys from the start position to the goal."""
        # Use a simple greedy algorithm to create a path from start to goal
        current_pos = start.copy()
        path = [current_pos]

        while not np.array_equal(current_pos, goal):
            possible_moves = []
            
            # Look at the neighboring cells (right, down, up, left)
            for direction in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                new_pos = current_pos + direction
                if 0 <= new_pos[0] < self.grid_size[0] and 0 <= new_pos[1] < self.grid_size[1]:
                    possible_moves.append(new_pos)
            
            # Move to the next position that brings us closer to the goal (greedy approach)
            current_pos = min(possible_moves, key=lambda x: np.linalg.norm(x - goal))
            path.append(current_pos)

        # Set the terrain to valley (1) along the path
        for pos in path:
            self.terrain[pos[0], pos[1]] = 1

    def add_random_terrain(self):
        """Randomly scatter mountains and valleys on the remaining grid."""
        for row in range(self.grid_size[0]):
            for col in range(self.grid_size[1]):
                if self.terrain[row, col] == 0:  # Only modify the mountain cells
                    # Randomly place some valleys in the remaining mountain area
                    if random.random() < 0.2:  # 20% chance of creating a valley
                        self.terrain[row, col] = 1

    def load_images(self):
        """Load and scale all four images needed for rendering."""
        try:
            # Base terrain images
            self.mountain_img = pygame.image.load("/home/faruk/Desktop/HDD_Projects/DRL_AutonomousDrone/env_images/mountain_2.png")
            self.valley_img = pygame.image.load("/home/faruk/Desktop/HDD_Projects/DRL_AutonomousDrone/env_images/flat_terrain.png")
            
            # Special images
            self.corner_img = pygame.image.load("/home/faruk/Desktop/HDD_Projects/DRL_AutonomousDrone/env_images/fire.png")
            self.drone_img = pygame.image.load("/home/faruk/Desktop/HDD_Projects/DRL_AutonomousDrone/env_images/drone_3.png")
            

            # Scale images to cell size
            self.mountain_img = pygame.transform.scale(self.mountain_img, (self.cell_size, self.cell_size))
            self.valley_img = pygame.transform.scale(self.valley_img, (self.cell_size, self.cell_size))
            self.drone_img = pygame.transform.scale(self.drone_img, (self.cell_size-20, self.cell_size-20))
            
            # Prepare corner image with red border
            self.corner_img = pygame.transform.scale(self.corner_img, (100, 100))
            corner_with_border = pygame.Surface((100, 100), pygame.SRCALPHA)
            corner_with_border.blit(self.corner_img, (0, 0))
            pygame.draw.rect(corner_with_border, (255, 0, 0), (0, 0, 100, 100), 5)
            self.corner_img = corner_with_border
            
        except pygame.error as e:
            print(f"Error loading images: {e}")
            # Fallback to colored rectangles if images fail to load
            self.create_fallback_surfaces()
    
    def create_fallback_surfaces(self):
        """Create simple colored surfaces if image loading fails."""
        self.mountain_img = pygame.Surface((self.cell_size, self.cell_size))
        self.mountain_img.fill((139, 137, 137))  # Gray for mountains
        
        self.valley_img = pygame.Surface((self.cell_size, self.cell_size))
        self.valley_img.fill((34, 139, 34))  # Green for valleys
        
        self.drone_img = pygame.Surface((self.cell_size-20, self.cell_size-20))
        self.drone_img.fill((0, 0, 255))  # Blue for drone
        
        self.corner_img = pygame.Surface((100, 100))
        self.corner_img.fill((255, 165, 0))  # Orange for corner
        pygame.draw.rect(self.corner_img, (255, 0, 0), (0, 0, 100, 100), 5)
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Reset terrain with the same fixed seed (if you want to re-use it on every reset)
        np.random.seed(self.fixed_seed)
        random.seed(self.fixed_seed)
        self.generate_terrain()
        
        self.agent_pos_1 = self.start_pos_1.copy()
        self.agent_pos_2 = self.start_pos_2.copy()
        
        print(f"Reset: Goal Position: {self.goal_pos}")
        
        return np.concatenate([self.agent_pos_1, self.agent_pos_2]), {}
    
    def step(self, action):
        """Execute one time step within the environment."""
        new_pos_2 = self.agent_pos_2.copy()

        # Action mapping for agent 2 (second agent only)
        if action == 0:  # Left
            new_pos_2[1] -= 1
        elif action == 1:  # Right
            new_pos_2[1] += 1
        elif action == 2:  # Up
            new_pos_2[0] -= 1
        elif action == 3:  # Down
            new_pos_2[0] += 1

        # Restrict agent 2 to move within the 2-5 range for both rows and columns
        new_pos_2[0] = np.clip(new_pos_2[0], 2, 5)  # Rows 2-5
        new_pos_2[1] = np.clip(new_pos_2[1], 2, 5)  # Columns 2-5

        # Check if new position for agent 2 is valid (1 = valley/flat terrain)
        if self.terrain[new_pos_2[0], new_pos_2[1]] == 1:
            self.agent_pos_2 = new_pos_2
        
        # Check if agent 2 has reached the goal (fire)
        done = np.array_equal(self.agent_pos_2, self.goal_pos)
        reward = 10.0 if done else -0.1
        info = {
            "distance_to_goal_2": np.linalg.norm(self.goal_pos - self.agent_pos_2),
            "position_2": self.agent_pos_2.copy()
        }
        
        return np.concatenate([self.agent_pos_1, self.agent_pos_2]), reward, done, False, info
    
    def render_frame(self, drone_row_1=None, drone_col_1=None, drone_row_2=None, drone_col_2=None):
        """Render a single frame of the environment."""
        # Use current positions if none provided
        if drone_row_1 is None or drone_col_1 is None:
            drone_row_1, drone_col_1 = self.agent_pos_1
        if drone_row_2 is None or drone_col_2 is None:
            drone_row_2, drone_col_2 = self.agent_pos_2
        
        # Draw terrain
        for row in range(self.grid_size[0]):
            for col in range(self.grid_size[1]):
                terrain_type = self.terrain[row, col]
                img = self.valley_img if terrain_type == 1 else self.mountain_img
                self.screen.blit(img, (col * self.cell_size, row * self.cell_size))
        
        # Draw goal (red rectangle) for fire position
        fire_row, fire_col = self.goal_pos
        pygame.draw.rect(
            self.screen, 
            (255, 0, 0), 
            pygame.Rect(fire_col * self.cell_size, fire_row * self.cell_size, self.cell_size, self.cell_size)
        )
        
        # Draw drones
        self.screen.blit(self.drone_img, (drone_col_1 * self.cell_size + 10, drone_row_1 * self.cell_size + 10))
        self.screen.blit(self.drone_img, (drone_col_2 * self.cell_size + 10, drone_row_2 * self.cell_size + 10))
        
        # Draw fire at goal position
        self.screen.blit(self.corner_img, (fire_col * self.cell_size, fire_row * self.cell_size))
        
        pygame.display.flip()

    def render(self):
        """Render the current state (alias for render_frame)."""
        self.render_frame()
    
    def close(self):
        """Close the environment and cleanup."""
        pygame.quit()

# Main loop to run the environment
if __name__ == "__main__":
    # Initialize environment
    pygame.init()
    info = pygame.display.Info()
    screen_height = info.current_h
    
    grid_size = (6, 6)  # rows, columns
    cell_size = min(100, int(screen_height * 0.8 / grid_size[0]))
    
    env = DroneNavigationEnv(grid_size=grid_size, cell_size=cell_size)
    obs, _ = env.reset()
    
    # Main game loop
    running = True
    clock = pygame.time.Clock()
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Take random actions for the second agent (first agent stays stationary)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        env.render()
        print(f"Position: {obs}, Reward: {reward}, Done: {terminated}")
        
        if terminated:
            print("Goal reached!")
            # Keep displaying the final state for 2 seconds
            pygame.time.delay(2000)
            running = False
        
        clock.tick(10)  # Limit to 10 FPS
    
    env.close()
