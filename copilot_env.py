import numpy as np
import gymnasium as gym
from gymnasium import spaces

from shared_autonomy_env import SharedAutonomyEnv


def _one_hot(action: int, n: int = 6) -> np.ndarray:
    v = np.zeros(n, dtype=np.float32)
    v[action] = 1.0
    return v


class CopilotEnv(gym.Wrapper):
    """SharedAutonomyEnv extended with a pilot's proposed action in the observation.

    Observation (15-dim):
      [0:9]  world state from SharedAutonomyEnv (includes obs[8] goal x)
      [9:15] one-hot encoding of the current pilot's most-recent proposed action

    The copilot chooses Discrete(6) actions that go directly to the underlying env.
    The pilot's action is information only — it never reaches the underlying env.

    A pilot is randomly sampled from the provided list at the start of each episode,
    enabling a single copilot policy to generalize across impaired pilot types.

    Args:
        pilots: list of callables (obs: np.ndarray) -> int (action 0-5).
        render_mode: forwarded to SharedAutonomyEnv.
        reward_shaping: forwarded to SharedAutonomyEnv.
    """

    def __init__(self, pilots: list, render_mode=None, reward_shaping: bool = False):
        env = SharedAutonomyEnv(render_mode=render_mode, using_lander_reward_shaping=reward_shaping)
        super().__init__(env)

        self.pilots = pilots
        self.current_pilot = None
        self._last_pilot_action: int = 1  # noop until first reset

        base = env.observation_space  # Box(9,)
        self.observation_space = spaces.Box(
            low=np.concatenate([base.low, np.zeros(6, dtype=np.float32)]),
            high=np.concatenate([base.high, np.ones(6, dtype=np.float32)]),
            dtype=np.float32,
        )
        # action_space remains Discrete(6) from SharedAutonomyEnv

    def _build_obs(self, world_obs: np.ndarray, pilot_action: int) -> np.ndarray:
        return np.concatenate([world_obs, _one_hot(pilot_action)]).astype(np.float32)

    def reset(self, seed=None, options=None):
        world_obs, info = self.env.reset(seed=seed, options=options)

        # Sample one pilot for this episode
        idx = int(self.np_random.integers(0, len(self.pilots)))
        self.current_pilot = self.pilots[idx]
        if hasattr(self.current_pilot, "reset"):
            self.current_pilot.reset()

        self._last_pilot_action = int(self.current_pilot(world_obs))
        return self._build_obs(world_obs, self._last_pilot_action), info

    def step(self, copilot_action: int):
        world_obs, reward, terminated, truncated, info = self.env.step(copilot_action)
        self._last_pilot_action = int(self.current_pilot(world_obs))
        return self._build_obs(world_obs, self._last_pilot_action), reward, terminated, truncated, info


if __name__ == "__main__":
    from full_pilot import FullPilot
    from pilot_policies import LaggyPilot, NoisyPilot, noop_pilot, sensor_pilot

    full = FullPilot()
    pilots = [
        LaggyPilot(full, lag_prob=0.8),
        NoisyPilot(full, noise_prob=0.15),
        noop_pilot,
        sensor_pilot,
    ]
    pilot_names = ["LaggyPilot", "NoisyPilot", "noop_pilot", "sensor_pilot"]

    env = CopilotEnv(pilots)

    assert env.observation_space.shape == (15,), \
        f"Expected (15,), got {env.observation_space.shape}"
    assert env.action_space.n == 6, \
        f"Expected Discrete(6), got {env.action_space}"
    print("Spaces OK — obs (15,), action Discrete(6)")

    obs, _ = env.reset()
    episode = 1
    print(f"\nEpisode {episode}: pilot = {pilot_names[pilots.index(env.current_pilot)]}")

    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
            episode += 1
            name = pilot_names[pilots.index(env.current_pilot)]
            print(f"Episode {episode}: pilot = {name}")

    print(f"\nFinal obs shape: {obs.shape}")
    print("Smoke test passed.")
    env.close()
