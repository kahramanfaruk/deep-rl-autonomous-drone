import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def train_q_learning(env, no_episodes, epsilon, epsilon_min, epsilon_decay, alpha, gamma, q_table_save_path="q_table.npy"):
    def to_native_tuple(np_array):
        return tuple(int(x) for x in np_array)
    """
    Train a Q-learning agent on a custom Gym environment.

    Parameters
    ----------
    env : gym.Env
        The custom drone navigation environment.
    no_episodes : int
        Number of episodes for training.
    epsilon : float
        Initial exploration rate.
    epsilon_min : float
        Minimum value for epsilon after decay.
    epsilon_decay : float
        Factor by which epsilon is decayed each episode.
    alpha : float
        Learning rate (step size).
    gamma : float
        Discount factor for future rewards.
    q_table_save_path : str, optional
        Path to save the learned Q-table (default is "q_table.npy").

    Returns
    -------
    None
    """
    rows, cols = env.grid_size
    q_table = np.zeros((rows, cols, env.action_space.n))  # Initialize Q-table to zeros

    for episode in range(no_episodes):
        state, _ = env.reset()
        #state = tuple(state)  # Convert to tuple for indexing
        state = to_native_tuple(state)
        total_reward = 0
        step_counter = 0
        print(f"\n--- Episode {episode + 1} ---")
        print(f"Starting at position: {state}")

        while True:
            # ε-greedy action selection
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Explore
                decision = "exploration"
            else:
                action = np.argmax(q_table[state])  # Exploit
                decision = "exploration"

            next_state, reward, done, _, _ = env.step(action)
            
            env.render()

            #next_state = tuple(next_state)
            next_state = to_native_tuple(next_state)
            total_reward += reward

            # Q-learning update rule
            q_table[state][action] += alpha * (
                reward + gamma * np.max(q_table[next_state]) - q_table[state][action]
            )
            print(f"Step {step_counter:3d} | Pos: {state} -> {next_state} | Action: {action} ({decision}) | "
                  f"Reward: {reward:+.2f} | ε: {epsilon:.3f}")
            step_counter += 1
            state = next_state

            if done:
                print(f"✔️ Reached goal at {state} in {step_counter} steps. Total reward: {total_reward:.2f}")
                break

        # Decay epsilon after each episode
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        print(f"Episode {episode + 1}/{no_episodes}, Total Reward: {total_reward:.2f}")

    env.close()
    np.save(q_table_save_path, q_table)
    print(f"Training complete. Q-table saved to '{q_table_save_path}'.")


def visualize_q_table(hell_state_coordinates=[(2, 1), (0, 4)],
                      goal_coordinates=(4, 4),
                      actions=["Up", "Down", "Right", "Left"],
                      q_values_path="q_table.npy"):

    # Load the Q-table:
    # -----------------
    try:
        q_table = np.load(q_values_path)

        # Create subplots for each action:
        # --------------------------------
        _, axes = plt.subplots(1, 4, figsize=(20, 5))

        for i, action in enumerate(actions):
            ax = axes[i]
            heatmap_data = q_table[:, :, i].copy()

            # Mask the goal state's Q-value for visualization:
            # ------------------------------------------------
            mask = np.zeros_like(heatmap_data, dtype=bool)
            mask[goal_coordinates] = True
            for hell in hell_state_coordinates:
                mask[hell] = True

            # Create a custom colormap that shows negative values in red
            sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="coolwarm",
                        ax=ax, cbar=False, mask=mask, annot_kws={"size": 9},
                        center=0)  # Center color map at 0

            # Denote Goal and Hell states:
            # ----------------------------
            ax.text(goal_coordinates[1] + 0.5, goal_coordinates[0] + 0.5, 'G', color='green',
                    ha='center', va='center', weight='bold', fontsize=14)
            for hell in hell_state_coordinates:
                ax.text(hell[1] + 0.5, hell[0] + 0.5, 'H', color='red',
                        ha='center', va='center', weight='bold', fontsize=14)

            ax.set_title(f'Action: {action}')

        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print("No saved Q-table was found. Please train the Q-learning agent first or check your path.")

