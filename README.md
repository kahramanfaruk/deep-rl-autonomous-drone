# 🚁 Drone Navigation with Q-Learning

This project implements a reinforcement learning agent using **Q-Learning** to navigate a custom grid-based terrain environment. The goal is for a drone to reach a target location while avoiding hazardous terrain.

---

## 📁 Project Structure



- drone_main.py # Main script to run training and visualization
- drone_q_learning.py # Q-learning algorithm and Q-table visualization
- drone_nav_env.py # Custom drone navigation environment using Pygame
- logs/ # Output directory for Q-tables, heatmaps, TensorBoard logs
- env_images/ # Required images for rendering the environment



---

## 🚀 Features

- Custom grid environment with valleys, mountains, and goal cell
- Q-Learning with ε-greedy strategy
- Visualizations: heatmaps for Q-values and policy arrows
- Multiple training runs with saved logs and metrics
- TensorBoard support for training insights
- Pygame rendering for step-by-step simulation

---

## 🔧 Requirements

Install dependencies using pip:

```bash
pip install numpy matplotlib seaborn pygame torch gymnasium
