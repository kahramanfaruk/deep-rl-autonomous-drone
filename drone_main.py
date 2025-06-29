from drone_nav_env import DroneNavigationEnv
from drone_q_learning import train_q_learning, visualize_q_table
import os

# --- Config ---
train = True
visualize_results = True
num_runs = 5  # Run training 5 times

# --- Hyperparameters ---
learning_rate = 0.1
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.995
no_episodes = 1000

# --- Environment ---
grid_size = (5, 5)
cell_size = 100
random_seed = 42
goal_coordinates = (grid_size[0] - 1, grid_size[1] - 1)

# --- Run Training and Visualization for Multiple Runs ---
for run_id in range(num_runs):
    print(f"\n=== 🚀 Run {run_id + 1}/{num_runs} ===")

    env = DroneNavigationEnv(grid_size=grid_size, 
                             cell_size=cell_size, 
                             fixed_seed=random_seed)

    q_table_path = f"logs/run_{run_id}/q_table.npy"
    log_dir = f"logs/run_{run_id}/tensorboard"

    os.makedirs(os.path.dirname(q_table_path), exist_ok=True)

    if train:
        train_q_learning(env=env,
                         no_episodes=no_episodes,
                         epsilon=epsilon,
                         epsilon_min=epsilon_min,
                         epsilon_decay=epsilon_decay,
                         alpha=learning_rate,
                         gamma=gamma,
                         q_table_save_path=q_table_path,
                         tensorboard_log_dir=log_dir)

    if visualize_results:
        visualize_q_table(
            goal_coordinates=goal_coordinates,
            q_values_path=q_table_path,
            save_path_prefix=f"logs/run_{run_id}/q_heatmap"
        )
