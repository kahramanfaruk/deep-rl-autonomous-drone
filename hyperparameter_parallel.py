import numpy as np
import csv
from joblib import Parallel, delayed
from tqdm import tqdm
import os
import itertools

def run_single_trial(idx, trial, params, episodes, env_factory, early_stop_q_sum):
    from q_learning import train_q_learning

    env = env_factory()
    q_table_path = f"q_table_run_{idx}_trial_{trial}.npy"

    train_q_learning(
        env=env,
        no_episodes=episodes,
        epsilon=params['epsilon'],
        epsilon_min=params['epsilon_min'],
        epsilon_decay=params['epsilon_decay'],
        alpha=params['alpha'],
        gamma=params['gamma'],
        q_table_save_path=q_table_path
    )

    try:
        q_table = np.load(q_table_path)
        total_q = np.sum(q_table)

        if early_stop_q_sum is not None and total_q < early_stop_q_sum:
            # You could add logic to stop early here if your train_q_learning supports it
            pass

    except Exception as e:
        print(f"⚠️ Failed to load Q-table for idx {idx}, trial {trial}: {e}")
        total_q = float('-inf')

    return total_q

def hyperparameter_search_parallel(param_grid, env_factory, episodes=500, trials_per_combination=3, early_stop_q_sum=None, n_jobs=-1):
    from collections import defaultdict

    keys = list(param_grid.keys())
    combinations = list(itertools.product(*(param_grid[key] for key in keys)))
    results = []
    log_file = "hyperparameter_log.csv"

    # Prepare CSV log file
    with open(log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(keys + ['avg_q_sum'])

    # Function to run all trials for one hyperparameter combo
    def run_combination(idx, values):
        params = dict(zip(keys, values))
        q_sums = []

        for trial in range(trials_per_combination):
            total_q = run_single_trial(idx, trial, params, episodes, env_factory, early_stop_q_sum)
            q_sums.append(total_q)

        avg_q_sum = np.mean(q_sums)
        return {**params, "avg_q_sum": avg_q_sum}

    # Run in parallel with tqdm progress bar
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_combination)(idx, values) for idx, values in tqdm(enumerate(combinations), total=len(combinations))
    )

    # Append results to CSV
    with open(log_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        for res in results:
            writer.writerow([res[key] for key in keys] + [res['avg_q_sum']])

    return results





# Early stopping implemented
"""
import itertools
import numpy as np
import csv
from joblib import Parallel, delayed
from tqdm import tqdm
import os

def run_single_trial(idx, trial, params, episodes, env_factory, early_stop_q_sum,
                     use_early_stopping=False, early_stop_threshold=None, early_stop_window=10):
    from q_learning import train_q_learning

    env = env_factory()
    q_table_path = f"q_table_run_{idx}_trial_{trial}.npy"

    train_q_learning(
        env=env,
        no_episodes=episodes,
        epsilon=params['epsilon'],
        epsilon_min=params['epsilon_min'],
        epsilon_decay=params['epsilon_decay'],
        alpha=params['alpha'],
        gamma=params['gamma'],
        q_table_save_path=q_table_path,
        use_early_stopping=use_early_stopping,
        early_stop_threshold=early_stop_threshold,
        early_stop_window=early_stop_window
    )

    try:
        q_table = np.load(q_table_path)
        total_q = np.sum(q_table)

        if early_stop_q_sum is not None and total_q < early_stop_q_sum:
            # You could add logic to stop early here if your train_q_learning supports it
            pass

    except Exception as e:
        print(f"⚠️ Failed to load Q-table for idx {idx}, trial {trial}: {e}")
        total_q = float('-inf')

    return total_q

def hyperparameter_search_parallel(param_grid, env_factory, episodes=500, trials_per_combination=3,
                                   early_stop_q_sum=None,
                                   use_early_stopping=False,
                                   early_stop_threshold=None,
                                   early_stop_window=10,
                                   n_jobs=-1):
    from collections import defaultdict

    keys = list(param_grid.keys())
    combinations = list(itertools.product(*(param_grid[key] for key in keys)))
    results = []
    log_file = "hyperparameter_log.csv"

    # Prepare CSV log file
    with open(log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(keys + ['avg_q_sum'])

    # Function to run all trials for one hyperparameter combo
    def run_combination(idx, values):
        params = dict(zip(keys, values))
        q_sums = []

        for trial in range(trials_per_combination):
            total_q = run_single_trial(
                idx,
                trial,
                params,
                episodes,
                env_factory,
                early_stop_q_sum,
                use_early_stopping=use_early_stopping,
                early_stop_threshold=early_stop_threshold,
                early_stop_window=early_stop_window
            )
            q_sums.append(total_q)

        avg_q_sum = np.mean(q_sums)
        return {**params, "avg_q_sum": avg_q_sum}

    # Run in parallel with tqdm progress bar
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_combination)(idx, values) for idx, values in tqdm(enumerate(combinations), total=len(combinations))
    )

    # Append results to CSV
    with open(log_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        for res in results:
            writer.writerow([res[key] for key in keys] + [res['avg_q_sum']])

    return results
"""