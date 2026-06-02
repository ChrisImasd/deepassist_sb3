import numpy as np
from stable_baselines3 import DQN

from shared_autonomy_env import SharedAutonomyEnv
from full_pilot import FullPilot
from risk_aware_policy import risk_aware_action, build_copilot_obs


def run_episode(env, pilot, copilot_model, mode: str):
    obs, _ = env.reset()

    if hasattr(pilot, "reset"):
        pilot.reset()

    total_reward = 0.0
    done = False

    interventions = 0
    steps = 0
    final_reward = 0.0

    while not done:
        pilot_action = int(pilot(obs))

        if mode == "pilot":
            action = pilot_action

        elif mode == "copilot":
            copilot_obs = build_copilot_obs(obs, pilot_action)
            action, _ = copilot_model.predict(copilot_obs, deterministic=True)
            action = int(action)

        elif mode == "risk_aware":
            action, info = risk_aware_action(obs, pilot_action, copilot_model)
            interventions += info["intervened"]

        else:
            raise ValueError(f"Unknown mode: {mode}")

        obs, reward, terminated, truncated, _ = env.step(action)

        total_reward += reward
        final_reward = reward
        done = terminated or truncated
        steps += 1

    return {
        "reward": total_reward,
        "intervention_rate": interventions / max(steps, 1),
        "final_reward": final_reward,
    }


def evaluate(mode: str, n_episodes: int = 50):
    env = SharedAutonomyEnv()
    pilot = FullPilot("models/full_pilot.zip")
    copilot_model = DQN.load("models/copilot.zip")

    results = []

    for _ in range(n_episodes):
        results.append(run_episode(env, pilot, copilot_model, mode))

    env.close()

    rewards = np.array([r["reward"] for r in results])
    intervention_rates = np.array([r["intervention_rate"] for r in results])
    final_rewards = np.array([r["final_reward"] for r in results])

    # Gymnasium Lunar Lander gives an extra +100 for landing safely
    # and -100 for crashing, so terminal reward is a useful rough classifier.
    crash_rate = np.mean(final_rewards < -50)
    success_rate = np.mean(final_rewards > 50)

    print(f"\nMode: {mode}")
    print(f"Mean reward:        {np.mean(rewards):8.2f}")
    print(f"Std reward:         {np.std(rewards):8.2f}")
    print(f"Success rate:       {success_rate:8.2%}")
    print(f"Crash rate:         {crash_rate:8.2%}")
    print(f"Intervention rate:  {np.mean(intervention_rates):8.2%}")


if __name__ == "__main__":
    for mode in ["pilot", "copilot", "risk_aware"]:
        evaluate(mode, n_episodes=50)