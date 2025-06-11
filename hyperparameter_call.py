from drone_nav_env import DroneNavigationEnv
from hyperparameter_search_grid import hyperparameter_search
from hyperparameter_parallel import hyperparameter_search_parallel
import numpy as np

import itertools

# TODO: Also try the adaptive exploration strategy instead of fixed epsilon


param_grid = {
    'alpha': np.round(np.linspace(0.05, 0.5, 5), 3).tolist(),           # [0.05, 0.1625, 0.275, 0.3875, 0.5]
    'gamma': np.round(np.linspace(0.85, 0.99, 5), 3).tolist(),          # [0.85, 0.8875, 0.925, 0.9625, 0.99]
    'epsilon': [1.0],
    'epsilon_min': [0.1],
    'epsilon_decay': np.round(np.linspace(0.95, 0.999, 5), 3).tolist()  # [0.95, 0.96225, 0.9745, 0.98675, 0.999]
}

print(param_grid)


def env_factory():
    return DroneNavigationEnv(grid_size=(5, 5), cell_size=100, fixed_seed=42)

# Sequential hyperparameter search
"""
results = hyperparameter_search(param_grid, env_factory, episodes=1)

# Sort and show best
results.sort(key=lambda r: -r['q_sum'])
for i, r in enumerate(results[:3]):
    print(f"🏅 Top {i+1}: {r}")
"""


# n_jobs=-1 uses all CPU cores; adjust as needed
# tqdm wraps over the combinations for a neat progress bar
#Each trial is done serially per combination (could be parallelized deeper if needed)
# If your train_q_learning supports interruption, you can integrate early stopping inside run_single_trial

#  Parallel hyperparameter search
results = hyperparameter_search_parallel(
    param_grid=param_grid,
    env_factory=env_factory,
    episodes=1,
    trials_per_combination=3,
    early_stop_q_sum=-500,
    n_jobs=-1  # Use all CPUs; change to e.g. 4 for 4 parallel jobs
)


"""
# Parallel hyperparameter search with early stopping
results = hyperparameter_search_parallel(
    param_grid=param_grid,
    env_factory=env_factory,
    episodes=1,
    trials_per_combination=3,
    early_stop_q_sum=-500,
    use_early_stopping=True,             # Enable moving average early stopping
    early_stop_threshold=0.01,            # Threshold for improvement (tune as needed)
    early_stop_window=10,                 # Window size for moving average (must match train_q_learning)
    n_jobs=-1                            # Use all CPUs; change if needed
)
"""