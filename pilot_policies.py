import numpy as np


class RandomPilot:
    """Uniformly random action from Discrete(6). Placeholder for a trained pilot."""

    def __call__(self, obs: np.ndarray) -> int:
        return int(np.random.randint(6))


class LaggyPilot:
    """Wraps a base policy with sticky action behavior.

    On each step, returns the previous action with probability lag_prob instead of
    querying the base policy. Call reset() at episode boundaries to clear the cache.

    Args:
        base_policy: any callable (obs) -> int.
        lag_prob: probability of repeating the last action (default 0.8).
    """

    def __init__(self, base_policy, lag_prob: float = 0.8):
        self.base_policy = base_policy
        self.lag_prob = lag_prob
        self._last_action = None

    def reset(self):
        self._last_action = None

    def __call__(self, obs: np.ndarray) -> int:
        if self._last_action is None or np.random.random() >= self.lag_prob:
            self._last_action = self.base_policy(obs)
        return self._last_action


class NoisyPilot:
    """Wraps a base policy with independent per-step action corruption.

    After querying the base policy, applies two independent noise passes:
      - with probability noise_prob: flip the main engine state (actions 0-2 <-> 3-5)
      - with probability noise_prob: shift steering to one of the other two options

    Args:
        base_policy: any callable (obs) -> int.
        noise_prob: corruption probability for each pass (default 0.15).
    """

    def __init__(self, base_policy, noise_prob: float = 0.15):
        self.base_policy = base_policy
        self.noise_prob = noise_prob

    def __call__(self, obs: np.ndarray) -> int:
        action = self.base_policy(obs)
        if np.random.random() < self.noise_prob:
            action = (action + 3) % 6
        if np.random.random() < self.noise_prob:
            action = action // 3 * 3 + (action + np.random.randint(1, 3)) % 3
        return int(action)


def noop_pilot(obs: np.ndarray) -> int:
    """Always returns action 1: main engine off, no steering."""
    return 1


def sensor_pilot(obs: np.ndarray, thresh: float = 0.1) -> int:
    """Steers toward the goal using obs[8] (goal x) and obs[0] (lander x).

    Computes signed horizontal distance to goal, then returns a main-engine-off
    action: steer left (0), no steer (1), or steer right (2).
    """
    d = obs[8] - obs[0]  # positive means goal is to the right of the lander
    if d < -thresh:
        return 0  # steer left
    elif d > thresh:
        return 2  # steer right
    return 1  # close enough, no steer


if __name__ == "__main__":
    from shared_autonomy_env import SharedAutonomyEnv
    from full_pilot import FullPilot

    env = SharedAutonomyEnv()

    try:
        full = FullPilot()
        print("FullPilot loaded from models/full_pilot.zip")
    except FileNotFoundError:
        full = None
        print("models/full_pilot.zip not found — run train_full_pilot.py first. Using RandomPilot as base.")

    base = full if full is not None else RandomPilot()

    pilots = [
        ("random", RandomPilot()),
        ("laggy",  LaggyPilot(base)),
        ("noisy",  NoisyPilot(base)),
        ("noop",   noop_pilot),
        ("sensor", sensor_pilot),
    ]
    if full is not None:
        pilots.append(("full", full))

    for name, pilot in pilots:
        obs, _ = env.reset()
        if hasattr(pilot, "reset"):
            pilot.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = pilot(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        print(f"{name:8s}  total reward: {total_reward:8.2f}")

    env.close()
