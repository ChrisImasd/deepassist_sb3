import gymnasium as gym
from stable_baselines3 import DQN

env = gym.make("LunarLander-v3")

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=5000)

env.close()

eval_env = gym.make("LunarLander-v3", render_mode="human")

for episode in range(3):
    obs, _ = eval_env.reset()
    total_reward = 0.0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = eval_env.step(action)
        total_reward += reward
        done = terminated or truncated
    print(f"Episode {episode + 1}: total reward = {total_reward:.2f}")

eval_env.close()
