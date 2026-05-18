import numpy as np
import gymnasium as gym
from gymnasium import spaces

THROTTLE_MAG = 0.75
_STEERING = [-THROTTLE_MAG, 0.0, THROTTLE_MAG]


def disc_to_cont(action: int) -> np.ndarray:
    """Discrete(6) → continuous [main_engine, steering] at throttle_mag=0.75.

    action // 3 → main engine: 0-2 = off (-0.75),  3-5 = on (+0.75)
    action  % 3 → steering:    0 = left (-0.75), 1 = center (0.0), 2 = right (+0.75)

      0 → [-0.75, -0.75]   3 → [+0.75, -0.75]
      1 → [-0.75,  0.00]   4 → [+0.75,  0.00]
      2 → [-0.75, +0.75]   5 → [+0.75, +0.75]
    """
    m = THROTTLE_MAG if action >= 3 else -THROTTLE_MAG
    s = _STEERING[action % 3]
    return np.array([m, s], dtype=np.float32)


class SharedAutonomyEnv(gym.Wrapper):
    """LunarLanderContinuous-v3 with Discrete(6) actions and a 9-dim observation.

    Observation layout:
      [0] x position        [4] angle
      [1] y position        [5] angular velocity
      [2] x velocity        [6] left leg contact
      [3] y velocity        [7] right leg contact
                            [8] goal x-coordinate, sampled each episode from [-0.8, 0.8]

    The goal x in obs[8] communicates the target landing zone to the agent.
    It is sampled fresh in reset() and held constant for the episode.

    Args:
        render_mode: forwarded to LunarLanderContinuous-v3.
        using_lander_reward_shaping: stored for future use; currently has no effect.
    """

    def __init__(self, render_mode=None, using_lander_reward_shaping: bool = False):
        env = gym.make("LunarLanderContinuous-v3", render_mode=render_mode)
        super().__init__(env)

        self.using_lander_reward_shaping = using_lander_reward_shaping
        self._goal_x = 0.0

        self.action_space = spaces.Discrete(6)

        base = self.env.observation_space
        self.observation_space = spaces.Box(
            low=np.append(base.low, -1.0).astype(np.float32),
            high=np.append(base.high, 1.0).astype(np.float32),
            dtype=np.float32,
        )

    def _augment(self, obs: np.ndarray) -> np.ndarray:
        return np.append(obs, self._goal_x).astype(np.float32)

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._goal_x = float(self.np_random.uniform(-0.8, 0.8))
        return self._augment(obs), info

    def step(self, action: int):
        cont = disc_to_cont(int(action))
        obs, reward, terminated, truncated, info = self.env.step(cont)
        return self._augment(obs), reward, terminated, truncated, info


if __name__ == "__main__":
    env = SharedAutonomyEnv()
    obs, info = env.reset()

    print("Action space:     ", env.action_space)
    print("Observation space:", env.observation_space)
    print("First obs:        ", obs)
    print(f"  obs[8] goal x  = {obs[8]:.4f}  (sampled from [-0.8, 0.8])")

    assert isinstance(env.action_space, spaces.Discrete) and env.action_space.n == 6, \
        f"Expected Discrete(6), got {env.action_space}"
    assert env.observation_space.shape == (9,), \
        f"Expected observation_space.shape == (9,), got {env.observation_space.shape}"

    total_reward = 0.0
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            obs, info = env.reset()

    print(f"\nTotal reward over 100 steps: {total_reward:.2f}")
    print(f"Final obs shape:  {obs.shape}")
    print("\nAll assertions passed.")
    env.close()
