from pathlib import Path

import numpy as np
from stable_baselines3 import DQN

from copilot_env import CopilotEnv
from full_pilot import FullPilot
from pilot_policies import LaggyPilot, NoisyPilot, noop_pilot, sensor_pilot

TOTAL_TIMESTEPS = 500000
MODEL_PATH = Path("models/copilot.zip")


def make_training_env():
    full = FullPilot()
    return CopilotEnv([
        LaggyPilot(full, lag_prob=0.8),
        NoisyPilot(full, noise_prob=0.15),
        noop_pilot,
        sensor_pilot,
    ])


def evaluate(model, pilot, n_episodes: int = 20) -> float:
    env = CopilotEnv([pilot])
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        rewards.append(total)
    env.close()
    return float(np.mean(rewards))


def main():
    MODEL_PATH.parent.mkdir(exist_ok=True)

    retrain = True
    if MODEL_PATH.exists():
        resp = input(f"{MODEL_PATH} found. [e]valuate only or [r]etrain from scratch? ").strip().lower()
        retrain = not resp.startswith("e")

    env = make_training_env()

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

    env.close()

    # Evaluate against all 5 pilots — FullPilot is unseen during training
    full = FullPilot()
    eval_pilots = [
        ("LaggyPilot  (trained on)", LaggyPilot(full, lag_prob=0.8)),
        ("NoisyPilot  (trained on)", NoisyPilot(full, noise_prob=0.15)),
        ("noop_pilot  (trained on)", noop_pilot),
        ("sensor_pilot(trained on)", sensor_pilot),
        ("FullPilot   (unseen)    ", FullPilot()),
    ]

    print("\nEvaluation — 20 episodes per pilot (deterministic copilot):")
    print(f"  {'Pilot':<35} {'Mean reward':>12}")
    print(f"  {'-'*35} {'-'*12}")
    for name, pilot in eval_pilots:
        mean = evaluate(model, pilot)
        print(f"  {name:<35} {mean:12.2f}")


if __name__ == "__main__":
    main()
