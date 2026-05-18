from pathlib import Path

import numpy as np
from stable_baselines3 import DQN

from shared_autonomy_env import SharedAutonomyEnv

TOTAL_TIMESTEPS = 500_000
MODEL_PATH = Path("models/full_pilot.zip")


def evaluate(model, env, n_episodes: int = 5):
    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
        print(f"  Episode {ep + 1}: {total_reward:.2f}")
    print(f"  Mean: {np.mean(rewards):.2f}")
    return rewards


def main():
    MODEL_PATH.parent.mkdir(exist_ok=True)
    env = SharedAutonomyEnv()

    retrain = True
    if MODEL_PATH.exists():
        resp = input(f"{MODEL_PATH} found. [e]valuate only or [r]etrain from scratch? ").strip().lower()
        retrain = not resp.startswith("e")

    if retrain:
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=1e-3,
            buffer_size=50_000,
            learning_starts=1_000,
            batch_size=32,
            gamma=0.99,
            train_freq=1,
            target_update_interval=1_500,
            exploration_fraction=0.1,
            exploration_final_eps=0.02,
            policy_kwargs=dict(net_arch=[64, 64]),
            verbose=1,
        )
        print(f"Training for {TOTAL_TIMESTEPS:,} timesteps...")
        model.learn(total_timesteps=TOTAL_TIMESTEPS, log_interval=10)
        model.save(MODEL_PATH)
        print(f"\nModel saved to {MODEL_PATH}")
    else:
        model = DQN.load(MODEL_PATH, env=env)
        print(f"Loaded {MODEL_PATH}")

    print("\nEvaluating (5 episodes, deterministic):")
    evaluate(model, env)
    env.close()


if __name__ == "__main__":
    main()
