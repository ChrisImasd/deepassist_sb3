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
    Heuristic risk score in [0, 1].

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
    target_error = np.clip(abs(x - goal_x) / 0.8, 0.0, 1.0)

    risk = (
        0.25 * low_altitude
        + 0.30 * fast_descent
        + 0.25 * large_tilt
        + 0.10 * high_spin
        + 0.10 * target_error
    )

    return float(np.clip(risk, 0.0, 1.0))


def pilot_reliability(q_values: np.ndarray, pilot_action: int) -> float:
    """
    Reliability in [0, 1] based on how valuable the pilot's action looks
    according to the trained copilot.
    """
    q_min = float(np.min(q_values))
    q_max = float(np.max(q_values))
    q_pilot = float(q_values[int(pilot_action)])

    denom = q_max - q_min

    if denom < 1e-8:
        return 1.0

    rho = (q_pilot - q_min) / denom
    return float(np.clip(rho, 0.0, 1.0))


def adaptive_alpha(
    risk: float,
    reliability: float,
    alpha_min: float = 0.05,
    alpha_max: float = 0.95,
    lam: float = 0.65,
) -> float:
    """
    Higher alpha means stronger copilot intervention.
    """
    raw = alpha_min + (alpha_max - alpha_min) * (
        lam * risk + (1.0 - lam) * (1.0 - reliability)
    )

    return float(np.clip(raw, alpha_min, alpha_max))


def action_distance(a: int, b: int) -> float:
    """
    Distance between two discrete actions after mapping them to continuous
    [main_engine, steering] commands.
    """
    ua = disc_to_cont(int(a))
    ub = disc_to_cont(int(b))
    return float(np.linalg.norm(ua - ub))


def risk_aware_action(
    world_obs: np.ndarray,
    pilot_action: int,
    copilot_model,
) -> tuple[int, dict]:
    """
    Final risk-aware shared-autonomy selector.

    Returns:
        final_action, diagnostic_info
    """
    copilot_obs = build_copilot_obs(world_obs, pilot_action)
    q_values = get_q_values(copilot_model, copilot_obs)

    risk = landing_risk(world_obs)
    reliability = pilot_reliability(q_values, pilot_action)
    alpha = adaptive_alpha(risk, reliability)

    q_min = float(np.min(q_values))
    q_max = float(np.max(q_values))

    threshold = q_min + alpha * (q_max - q_min)

    candidate_actions = [
        a for a in range(6)
        if q_values[a] >= threshold
    ]

    if len(candidate_actions) == 0:
        candidate_actions = [int(np.argmax(q_values))]

    final_action = min(
        candidate_actions,
        key=lambda a: action_distance(a, pilot_action)
    )

    info = {
        "risk": risk,
        "reliability": reliability,
        "alpha": alpha,
        "pilot_action": int(pilot_action),
        "copilot_greedy_action": int(np.argmax(q_values)),
        "final_action": int(final_action),
        "intervened": int(final_action != int(pilot_action)),
        "q_values": q_values,
    }

    return int(final_action), info