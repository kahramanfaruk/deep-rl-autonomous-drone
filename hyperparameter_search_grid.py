import itertools
import numpy as np
import csv
import os
import itertools

def hyperparameter_search(param_grid, env_factory, episodes=500, trials_per_combination=3, early_stop_q_sum=None):
    """
    Performs a grid search with finer resolution, epsilon_decay sweep,
    multiple trials per combination, and optional early stopping.

    Parameters
    ----------
    param_grid : dict
        Dictionary of hyperparameters with values to search.
    env_factory : function
        Returns a new instance of the environment.
    episodes : int
        Number of training episodes per trial.
    trials_per_combination : int
        Number of runs per parameter combination to average results.
    early_stop_q_sum : float or None
        If provided, stops a trial early if Q-table sum is below this value.

    Returns
    -------
    results : list of dict
        Each dict includes hyperparameters and average q_sum.
    """
    from q_learning import train_q_learning

    keys = list(param_grid.keys())
    combinations = list(itertools.product(*(param_grid[key] for key in keys)))
    results = []

    log_file = "hyperparameter_log.csv"
    with open(log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(keys + ['avg_q_sum'])

    print(f"\n🔍 Running {len(combinations)} hyperparameter combinations with {trials_per_combination} trials each...")

    for idx, values in enumerate(combinations):
        params = dict(zip(keys, values))
        print(f"\n[{idx+1}/{len(combinations)}] Testing parameters: {params}")

        q_sums = []

        for trial in range(trials_per_combination):
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
                    print(f"⚠️ Early stopping trial due to low Q-table sum ({total_q:.2f})")
                    break

            except Exception as e:
                print(f"⚠️ Failed to load Q-table: {e}")
                total_q = float('-inf')

            q_sums.append(total_q)

        avg_q_sum = np.mean(q_sums)
        result = {**params, "avg_q_sum": avg_q_sum}
        results.append(result)

        with open(log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([params[key] for key in keys] + [avg_q_sum])

        print(f"✅ Avg Q-table sum over {len(q_sums)} trial(s): {avg_q_sum:.2f}")

    return results
