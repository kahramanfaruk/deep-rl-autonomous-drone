import time
from drone_nav_env import DroneNavigationEnv

env = DroneNavigationEnv()
obs = env.reset()
done = False

while not done:
    env.render()
    action = env.action_space.sample()  # Random movement
    #obs, reward, done, info = env.step(action)
    obs, reward, terminated, truncated, info = env.step(action)

    time.sleep(0.3)  # Add a delay to see the movement

env.render()
time.sleep(1)
env.close()
print("Drone reached the goal!")
