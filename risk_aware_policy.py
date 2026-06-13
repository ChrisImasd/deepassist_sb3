from collections import deque

import numpy as np
import torch as th

from shared_autonomy_env import disc_to_cont


def one_hot(action: int, n: int = 6) -> np.ndarray:
    v = np.zeros(n, dtype=np.float32)
    v[int(action)] = 1.0
    return v


def build_copilot_obs(world_obs: np.ndarray, pilot_action: int) -> np.ndarray:
    """
    world_obs is the 9D observation from SharedAutonomyEnv.
    pilot_action is the pilot's proposed discrete action.
    Returns the 15D observation expected by the trained copilot.
    """
    return np.concatenate([world_obs, one_hot(pilot_action)]).astype(np.float32)


def get_q_values(copilot_model, copilot_obs: np.ndarray) -> np.ndarray:
    """
    Extract Q-values from the trained SB3 DQN copilot.
    Returns a length-6 array: Q(z, a) for a = 0,...,5.
    """
    obs_tensor, _ = copilot_model.policy.obs_to_tensor(copilot_obs)

    q_net = getattr(copilot_model, "q_net", copilot_model.policy.q_net)

    with th.no_grad():
        q_values = q_net(obs_tensor)

    return q_values.cpu().numpy().squeeze()


def landing_risk(world_obs: np.ndarray) -> float:
    """
    Heuristic risk score r_t in [0, 1].

    Observation layout:
    0: x position
    1: y position
    2: x velocity
    3: y velocity
    4: angle
    5: angular velocity
    6: left leg contact
    7: right leg contact
    8: goal x
    """
    x = float(world_obs[0])
    y = float(world_obs[1])
    vx = float(world_obs[2])
    vy = float(world_obs[3])
    theta = float(world_obs[4])
    omega = float(world_obs[5])
    goal_x = float(world_obs[8])

    # Normalize each risk feature.
    low_altitude = np.clip((0.6 - y) / 0.6, 0.0, 1.0)
    fast_descent = np.clip(max(0.0, -vy) / 1.0, 0.0, 1.0)
    large_tilt = np.clip(abs(theta) / 0.6, 0.0, 1.0)
    high_spin = np.clip(abs(omega) / 1.5, 0.0, 1.0)
    # target_error extends the four-feature slide formulation (vy, theta, omega,
    # altitude); it is a deliberate refinement, not part of the original spec.
    target_error = np.clip(abs(x - goal_x) / 0.8, 0.0, 1.0)

    risk = (
        0.25 * low_altitude
        + 0.30 * fast_descent
        + 0.25 * large_tilt
        + 0.10 * high_spin
        + 0.10 * target_error
    )

    return float(np.clip(risk, 0.0, 1.0))


def _q_prime(q_values: np.ndarray) -> np.ndarray:
    """Q'(s, a) = Q(s, a) - min_{a'} Q(s, a'), so min(Q') = 0 and max(Q') = Q'(a*)."""
    return q_values - float(np.min(q_values))


def confidence_indicator(q_values: np.ndarray, pilot_action: int, tau: float = 0.5) -> float:
    """
    Single-step indicator for the windowed confidence estimator:

        indicator[ Q'(s, a_pilot) / Q'(s, a_star) >= tau ]

    Returns 1.0 if the pilot's action is within a fraction tau of the best
    action's advantage, else 0.0. When all Q-values are equal (a_star advantage
    is ~0) the pilot is trivially within tolerance, so we return 1.0.
    """
    q_prime = _q_prime(q_values)
    q_prime_star = float(np.max(q_prime))

    if q_prime_star < 1e-8:
        return 1.0

    ratio = float(q_prime[int(pilot_action)]) / q_prime_star
    return 1.0 if ratio >= tau else 0.0


def adaptive_alpha(
    risk: float,
    confidence: float,
    alpha_min: float = 0.0,
    alpha_max: float = 0.7,
) -> float:
    """
    Pilot tolerance alpha_t (Reddy et al. 2018 convention): higher alpha means a
    wider feasible set and more pilot freedom.

        alpha_t = alpha_min + (alpha_max - alpha_min) * (1 - r_t) * rho_t

    Note the asymmetry: with alpha_max = 0.7 < 1, even at zero risk and full
    confidence the feasible set excludes the worst actions, so the copilot can
    still override hazardous pilot inputs.
    """
    raw = alpha_min + (alpha_max - alpha_min) * (1.0 - risk) * confidence
    return float(np.clip(raw, alpha_min, alpha_max))


def action_distance(a: int, b: int) -> float:
    """
    Distance between two discrete actions after mapping them to continuous
    [main_engine, steering] commands.
    """
    ua = disc_to_cont(int(a))
    ub = disc_to_cont(int(b))
    return float(np.linalg.norm(ua - ub))


class RiskAwareController:
    """
    Risk-aware shared-autonomy controller using Reddy et al. (2018)'s tolerance
    convention with an adaptive, risk- and confidence-modulated alpha.

    Pipeline per step:
      r_t   = landing_risk(world_obs)
      rho_t = windowed mean of indicator[ Q'(s, a_pilot)/Q'(s, a_star) >= tau ]
      alpha_t = alpha_min + (alpha_max - alpha_min) * (1 - r_t) * rho_t
      A_feasible = { a : Q'(s, a) >= (1 - alpha_t) * Q'(s, a*) }
      a_executed = argmin_{a in A_feasible} d(a, a_pilot)

    alpha = 0 -> only a* is feasible (maximum copilot intervention).
    alpha = 1 -> all actions feasible (zero intervention; pilot action selected,
                 being trivially closest to itself).

    The confidence estimator is stateful: it maintains a rolling window of the
    most recent indicator values via a deque(maxlen=window). Call reset() at
    episode boundaries to clear it.
    """

    def __init__(
        self,
        window: int = 10,
        tau: float = 0.5,
        alpha_min: float = 0.0,
        alpha_max: float = 0.7,
    ):
        self.window = window
        self.tau = tau
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self._history: deque = deque(maxlen=window)

    def reset(self) -> None:
        """Clear the confidence window at an episode boundary."""
        self._history.clear()

    def update_confidence(self, q_values: np.ndarray, pilot_action: int) -> float:
        """
        Append this step's indicator to the rolling window and return rho_t, the
        windowed confidence: (1/N) * sum of indicators over the recent window.
        """
        self._history.append(confidence_indicator(q_values, pilot_action, self.tau))
        return float(np.mean(self._history))

    def select_action(self, world_obs, pilot_action: int, copilot_model, alpha_override=None):
        """
        Run the full risk-aware selection for one step.

        If alpha_override is given, that constant alpha is used in place of the
        adaptive blend (this is exactly the fixed_shared / Reddy-constant policy,
        reusing the same feasible-set + argmin-distance logic). The confidence
        window is still updated so rho_t remains available in the info dict.

        Returns:
            final_action, diagnostic_info
        """
        copilot_obs = build_copilot_obs(world_obs, pilot_action)
        q_values = get_q_values(copilot_model, copilot_obs)

        risk = landing_risk(world_obs)
        confidence = self.update_confidence(q_values, pilot_action)
        if alpha_override is not None:
            alpha = float(np.clip(alpha_override, 0.0, 1.0))
        else:
            alpha = adaptive_alpha(
                risk, confidence, alpha_min=self.alpha_min, alpha_max=self.alpha_max
            )

        # Reddy tolerance feasible set: A = { a : Q'(s,a) >= (1 - alpha) Q'(s,a*) }.
        q_prime = _q_prime(q_values)
        q_prime_star = float(np.max(q_prime))
        threshold = (1.0 - alpha) * q_prime_star

        candidate_actions = [a for a in range(6) if q_prime[a] >= threshold]

        if len(candidate_actions) == 0:
            candidate_actions = [int(np.argmax(q_values))]

        final_action = min(
            candidate_actions,
            key=lambda a: action_distance(a, pilot_action),
        )

        info = {
            "risk": risk,
            # "reliability" retained for backward compatibility; now holds rho_t.
            "reliability": confidence,
            "confidence": confidence,
            "alpha": alpha,
            "pilot_action": int(pilot_action),
            "copilot_greedy_action": int(np.argmax(q_values)),
            "final_action": int(final_action),
            "intervened": int(final_action != int(pilot_action)),
            "q_values": q_values,
        }

        return int(final_action), info


# Module-level default controller so the existing functional API keeps working.
# evaluate_risk_aware.py calls risk_aware_action(...) directly; this singleton
# carries the confidence window across calls. Call reset() at episode boundaries
# to clear it.
_DEFAULT_CONTROLLER = RiskAwareController()


def reset() -> None:
    """Reset the module-level default controller's confidence window."""
    _DEFAULT_CONTROLLER.reset()


def pilot_reliability(q_values: np.ndarray, pilot_action: int, tau: float = 0.5) -> float:
    """
    Backward-compatible wrapper. NOTE: this is now stateful — it updates the
    module-level controller's rolling window and returns the windowed confidence
    rho_t (not a single-step value). Prefer RiskAwareController for new code.
    """
    _DEFAULT_CONTROLLER.tau = tau
    return _DEFAULT_CONTROLLER.update_confidence(q_values, pilot_action)


def risk_aware_action(
    world_obs: np.ndarray,
    pilot_action: int,
    copilot_model,
) -> tuple[int, dict]:
    """
    Final risk-aware shared-autonomy selector (thin wrapper around the module's
    default RiskAwareController instance).

    Returns:
        final_action, diagnostic_info
    """
    return _DEFAULT_CONTROLLER.select_action(world_obs, pilot_action, copilot_model)


if __name__ == "__main__":
    from stable_baselines3 import DQN

    from copilot_env import CopilotEnv
    from full_pilot import FullPilot
    from pilot_policies import NoisyPilot

    # 1. Load the trained models.
    full = FullPilot("models/full_pilot.zip")
    copilot_model = DQN.load("models/copilot.zip")

    # 2. CopilotEnv with a single pilot (NoisyPilot wrapping FullPilot).
    pilot = NoisyPilot(full, noise_prob=0.15)
    env = CopilotEnv([pilot])

    # 3. Run one episode step-by-step through risk_aware_action().
    reset()  # clear the confidence window at the episode boundary
    obs, _ = env.reset()

    print(f"{'step':>4}  {'r_t':>5}  {'rho_t':>5}  {'alpha':>5}  "
          f"{'a_pilot':>7}  {'a_cop*':>6}  {'a_exec':>6}")
    print("-" * 52)

    required_keys = {"intervened"}  # the only key evaluate_risk_aware.py reads
    step = 0
    done = False
    while not done:
        world_obs = obs[:9]
        pilot_action = int(np.argmax(obs[9:15]))

        executed_action, info = risk_aware_action(world_obs, pilot_action, copilot_model)

        # 5. Assert the info dict still carries the keys the eval script needs.
        assert required_keys.issubset(info.keys()), \
            f"info dict missing required keys: {required_keys - set(info.keys())}"

        if step < 20:
            print(f"{step:>4}  {info['risk']:>5.2f}  {info['reliability']:>5.2f}  "
                  f"{info['alpha']:>5.2f}  {info['pilot_action']:>7}  "
                  f"{info['copilot_greedy_action']:>6}  {info['final_action']:>6}")

        obs, reward, terminated, truncated, _ = env.step(executed_action)
        done = terminated or truncated
        step += 1

    env.close()
    print("-" * 52)
    print(f"Episode finished after {step} steps. "
          f"info dict contains required keys {sorted(required_keys)}: OK")
