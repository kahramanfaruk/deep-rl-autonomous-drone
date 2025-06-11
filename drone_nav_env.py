import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os

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
        
        # Generate terrain (1=valley, 0=mountain) with a fixed seed
        self.terrain = np.random.choice([0, 1], size=self.grid_size, p=[0.3, 0.7])

        # Positions (using numpy arrays for easy manipulation)
        self.start_pos = np.array([0, 0])  # (row, column)
        self.goal_pos = np.array([grid_size[0]-1, grid_size[1]-1])  # (row, column)
        self.agent_pos = self.start_pos.copy()
        
        print(f"Start: {self.start_pos}, Goal: {self.goal_pos}")
        
        # Gym spaces
        self.action_space = spaces.Discrete(4)  # 0:left, 1:right, 2:up, 3:down
        self.observation_space = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([grid_size[0]-1, grid_size[1]-1]),
            dtype=np.int32
        )
        
        # Pygame initialization
        pygame.init()
        self.screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption("Drone Navigation Environment")
        self.clock = pygame.time.Clock()
        
        # Load all four images with error handling
        self.load_images()
        
        # Colors
        self.goal_color = (255, 0, 0)  # Red for goal indication
    
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
        self.terrain = np.random.choice([0, 1], size=self.grid_size, p=[0.3, 0.7])
        
        self.agent_pos = self.start_pos.copy()
        return self.agent_pos.copy(), {}
    
    def step(self, action):
        """Execute one time step within the environment."""
        new_pos = self.agent_pos.copy()
        
        # Action mapping:
        # 0: left (column - 1)
        # 1: right (column + 1)
        # 2: up (row - 1)
        # 3: down (row + 1)
        
        if action == 0:  # Left
            new_pos[1] = max(0, new_pos[1] - 1)
        elif action == 1:  # Right
            new_pos[1] = min(self.grid_size[1] - 1, new_pos[1] + 1)
        elif action == 2:  # Up
            new_pos[0] = max(0, new_pos[0] - 1)
        elif action == 3:  # Down
            new_pos[0] = min(self.grid_size[0] - 1, new_pos[0] + 1)
        
        # Always allow movement, but give different rewards based on terrain
        if not np.array_equal(new_pos, self.agent_pos):
            self.animate_agent_move(self.agent_pos, new_pos)
        self.agent_pos = new_pos
        
        # Calculate reward based on terrain type
        terrain_type = self.terrain[new_pos[0], new_pos[1]]
        done = np.array_equal(self.agent_pos, self.goal_pos)
        
        if done:
            reward = 10.0  # Large reward for reaching goal
        elif terrain_type == 0:  # Mountain
            reward = -1.0  # Higher penalty for being on mountain
        else:  # Valley
            reward = -0.1  # Small penalty for normal movement
        
        info = {
            "distance_to_goal": np.linalg.norm(self.goal_pos - self.agent_pos),
            "position": self.agent_pos.copy(),
            "terrain": "mountain" if terrain_type == 0 else "valley"
        }
        
        return self.agent_pos.copy(), reward, done, False, info
    
    def animate_agent_move(self, start_pos, end_pos, duration=0.2, steps=1):
        """Animate smooth movement between positions."""
        start_row, start_col = start_pos
        end_row, end_col = end_pos
        
        for step in range(1, steps + 1):
            t = step / steps
            current_row = start_row + (end_row - start_row) * t
            current_col = start_col + (end_col - start_col) * t
            
            # Redraw everything
            self.render_frame(current_row, current_col)
            
            pygame.display.flip()
            #self.clock.tick(1 / (duration / steps))
            self.clock.tick(30)  # Increase from 1/(duration/steps) to 30 FPS
    
    def render_frame(self, drone_row=None, drone_col=None):
        """Render a single frame of the environment."""
        # Use current position if none provided
        if drone_row is None or drone_col is None:
            drone_row, drone_col = self.agent_pos
        
        # Draw terrain
        for row in range(self.grid_size[0]):
            for col in range(self.grid_size[1]):
                terrain_type = self.terrain[row, col]
                img = self.valley_img if terrain_type == 1 else self.mountain_img
                self.screen.blit(img, (col * self.cell_size, row * self.cell_size))
        
        # Draw goal (red rectangle)
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
        
        # Draw drone
        drone_x = drone_col * self.cell_size + 10  # +10 for centering
        drone_y = drone_row * self.cell_size + 10
        self.screen.blit(self.drone_img, (drone_x, drone_y))
        
        # Draw corner image (fire.png with red border)
        self.screen.blit(
            self.corner_img,
            (self.window_size[0] - 100, self.window_size[1] - 100)
        )
        
        # Update display
        pygame.display.flip()
    
    def render(self):
        """Render the current state (alias for render_frame)."""
        self.render_frame()
    
    def close(self):
        """Close the environment and cleanup."""
        pygame.quit()

# TODO: Add a new drone agent (helper agent) which detects the fire and then transfers the location (goal position) to the main agent
if __name__ == "__main__":
    # Initialize environment
    pygame.init()
    info = pygame.display.Info()
    screen_height = info.current_h
    
    grid_size = (5, 5)  # rows, columns
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
        
        # Take random actions
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
