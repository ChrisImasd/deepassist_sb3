import numpy as np
from stable_baselines3 import DQN


class FullPilot:
    """Trained DQN pilot. Loads from models/full_pilot.zip.

    Exposes the same callable interface as the other pilots in pilot_policies.py:
    takes a (9,) observation, returns an integer action 0-5.
    reset() is a no-op — DQN is stateless across episodes — but is included for
    API consistency with LaggyPilot.
    """

    def __init__(self, model_path: str = "models/full_pilot.zip"):
        self.model = DQN.load(model_path)

    def reset(self):
        pass

    def __call__(self, obs: np.ndarray) -> int:
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)
