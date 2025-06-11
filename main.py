from drone_nav_env import DroneNavigationEnv
from q_learning import train_q_learning, visualize_q_table

# --- User-configurable Flags ---
train = True
visualize_results = True

# --- Hyperparameters ---
learning_rate = 0.1         # Alpha: learning rate
gamma = 0.95                # Gamma: discount factor
epsilon = 1.0               # Initial exploration rate
epsilon_min = 0.1           # Minimum exploration
epsilon_decay = 0.995       # Epsilon decay rate
no_episodes = 500           # Number of episodes to train

# --- Environment Setup ---
grid_size = (5, 5)
cell_size = 100
random_seed = 42
goal_coordinates = (grid_size[0] - 1, grid_size[1] - 1)  # Bottom-right corner

# --- Execute Training & Visualization ---
if train:
    # Initialize the environment
    env = DroneNavigationEnv(grid_size=grid_size, cell_size=cell_size, fixed_seed=random_seed)

    # Train Q-learning agent
    train_q_learning(env=env,
                     no_episodes=no_episodes,
                     epsilon=epsilon,
                     epsilon_min=epsilon_min,
                     epsilon_decay=epsilon_decay,
                     alpha=learning_rate,
                     gamma=gamma,
                     q_table_save_path="q_table.npy")

if visualize_results:
    # Display Q-values heatmaps
    visualize_q_table(goal_coordinates=goal_coordinates,
                      q_values_path="q_table.npy")
