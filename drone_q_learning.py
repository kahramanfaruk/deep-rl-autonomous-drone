import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
import os
import matplotlib.pyplot as plt
import seaborn as sns

def train_q_learning(env, no_episodes, epsilon, epsilon_min, epsilon_decay, alpha, gamma,
                     q_table_save_path="q_table.npy", tensorboard_log_dir="runs/default"):

    writer = SummaryWriter(log_dir=tensorboard_log_dir)

    def to_native_tuple(np_array):
        return tuple(int(x) for x in np_array)

    rows, cols = env.grid_size
    q_table = np.zeros((rows, cols, env.action_space.n))

    for episode in range(no_episodes):
        state, _ = env.reset()
        state = to_native_tuple(state)
        total_reward = 0
        step_counter = 0

        while True:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
                decision = "exploration"
            else:
                action = np.argmax(q_table[state])
                decision = "exploitation"

            next_state, reward, done, _, _ = env.step(action)
            # env.render()  # Comment this out if running headless

            next_state = to_native_tuple(next_state)
            total_reward += reward

            q_table[state][action] += alpha * (
                reward + gamma * np.max(q_table[next_state]) - q_table[state][action]
            )

            writer.add_scalar("Step/Reward", reward, episode * 1000 + step_counter)
            step_counter += 1
            state = next_state

            if done:
                break

        writer.add_scalar("Episode/Total_Reward", total_reward, episode)
        writer.add_scalar("Episode/Epsilon", epsilon, episode)
        writer.add_scalar("Episode/Step_Count", step_counter, episode)

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    env.close()
    np.save(q_table_save_path, q_table)
    writer.close()
    print(f"✅ Q-table saved: {q_table_save_path}")


def visualize_q_table(
    hell_state_coordinates=[(0, 2), (1, 4), (2, 1), (3, 3), (4, 0)],
    goal_coordinates=(4, 4),
    actions=["Left", "Right", "Up", "Down"],
    q_values_path="q_table.npy",
    save_path_prefix="q_visual"
):
    q_table = np.load(q_values_path)
    rows, cols, n_actions = q_table.shape

    os.makedirs(os.path.dirname(save_path_prefix), exist_ok=True)

    # Plot 1: Q-values per action
    _, axes = plt.subplots(1, n_actions, figsize=(20, 5))
    for i, action in enumerate(actions):
        ax = axes[i]
        heatmap_data = q_table[:, :, i].copy()

        mask = np.zeros_like(heatmap_data, dtype=bool)
        mask[goal_coordinates] = True
        for hell in hell_state_coordinates:
            mask[hell] = True

        sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="coolwarm",
                    ax=ax, cbar=False, mask=mask, annot_kws={"size": 9},
                    center=0)

        ax.text(goal_coordinates[1] + 0.5, goal_coordinates[0] + 0.5, 'G', color='green',
                ha='center', va='center', weight='bold', fontsize=14)
        for hell in hell_state_coordinates:
            ax.text(hell[1] + 0.5, hell[0] + 0.5, 'H', color='red',
                    ha='center', va='center', weight='bold', fontsize=14)

        ax.set_title(f'Action: {action}')
    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_actions.png")
    plt.close()

    # Plot 2: Best action arrows
    best_actions = np.argmax(q_table, axis=2)
    mask = np.zeros((rows, cols), dtype=bool)
    for hell in hell_state_coordinates:
        mask[hell] = True
    mask[goal_coordinates] = True

    action_symbols = {0: "←", 1: "→", 2: "↑", 3: "↓"}
    max_q_values = np.max(q_table, axis=2)

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(max_q_values, annot=False, fmt=".2f",
                     cmap="YlGnBu", mask=mask, cbar=True,
                     linewidths=0.3, linecolor='gray')

    for i in range(rows):
        for j in range(cols):
            if not mask[i, j]:
                symbol = action_symbols[best_actions[i, j]]
                ax.text(j + 0.5, i + 0.5, symbol,
                        ha='center', va='center',
                        color='black', fontsize=12, weight='bold')

    for hell in hell_state_coordinates:
        ax.add_patch(plt.Rectangle((hell[1], hell[0]), 1, 1,
                                   fill=True, color='lightgray', lw=1))
        ax.text(hell[1] + 0.5, hell[0] + 0.5, 'H',
                ha='center', va='center', color='red', weight='bold', fontsize=14)

    g_row, g_col = goal_coordinates
    ax.add_patch(plt.Rectangle((g_col, g_row), 1, 1,
                               fill=True, color='green', lw=1))
    ax.text(g_col + 0.5, g_row + 0.5, 'G',
            ha='center', va='center', color='white', weight='bold', fontsize=14)

    plt.title("Best Actions & Max Q-values")
    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_best_actions.png")
    plt.close()
