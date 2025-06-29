from drone_nav_env import DroneNavigationEnv
from hyperparameter_sequential import hyperparameter_search_sequential
from hyperparameter_parallel import hyperparameter_search_parallel
import numpy as np
import itertools


# TODO: Also try the adaptive exploration strategy instead of fixed epsilon
# TODO: Remove the lists there
param_grid = {
    'alpha': np.round(np.linspace(0.05, 0.5, 4), 3).tolist(),           # [0.05, 0.1625, 0.275, 0.3875, 0.5]
    'gamma': np.round(np.linspace(0.85, 0.99, 4), 3).tolist(),          # [0.85, 0.8875, 0.925, 0.9625, 0.99]
    'epsilon': [1.0],
    'epsilon_min': [0.1],
    'epsilon_decay': np.round(np.linspace(0.95, 0.999, 4), 3).tolist()  # [0.95, 0.96225, 0.9745, 0.98675, 0.999]
}

print(param_grid)


def env_factory():
    return DroneNavigationEnv(grid_size=(5, 5), cell_size=100, fixed_seed=42)


# Sequential hyperparameter search
"""
results = hyperparameter_search_sequential(param_grid, env_factory, episodes=1)

# Sort and show best
results.sort(key=lambda r: -r['q_sum'])
for i, r in enumerate(results[:3]):
    print(f"🏅 Top {i+1}: {r}")
"""


#  Parallel hyperparameter search
results = hyperparameter_search_parallel(
    param_grid=param_grid,
    env_factory=env_factory,
    episodes=250,
    trials_per_combination=3,
    early_stop_q_sum=None,  # Set to None to disable early stopping
    n_jobs=-1  # Use all CPUs; change to e.g. 4 for 4 parallel jobs
)

# ✅ Find and print the best hyperparameters
best_result = max(results, key=lambda x: x['avg_q_sum'])

print("\n🏆 Best Hyperparameters Found:")
for key, value in best_result.items():
    print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")


# Some explanation
# n_jobs=-1 uses all CPU cores; adjust as needed
# tqdm wraps over the combinations for a neat progress bar
#Each trial is done serially per combination (could be parallelized deeper if needed)
# If your train_q_learning supports interruption, you can integrate early stopping inside run_single_trial
