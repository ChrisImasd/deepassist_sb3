"""
Unified evaluation harness for the four shared-autonomy modes (slide 14).

Modes:
  human_only       — execute the pilot's action directly; no copilot.
  autonomous       — execute the copilot's greedy action; pilot ignored.
  fixed_shared     — Reddy alpha-tolerance policy with constant alpha (default 0.6).
  adaptive_shared  — risk-aware policy with adaptive alpha_t = g(r_t, rho_t).

fixed_shared and adaptive_shared share the exact feasible-set / argmin-distance
logic in RiskAwareController.select_action; fixed_shared just passes a constant
alpha_override.

Run a tiny smoke test:
  python evaluate_all_modes.py --test

Run the full 800-episode grid:
  python evaluate_all_modes.py
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
from stable_baselines3 import DQN

from shared_autonomy_env import SharedAutonomyEnv
from pilot_policies import LaggyPilot, NoisyPilot, sensor_pilot
from full_pilot import FullPilot
from risk_aware_policy import (
    RiskAwareController,
    build_copilot_obs,
    get_q_values,
)

MODES = ["human_only", "autonomous", "fixed_shared", "adaptive_shared"]
PILOTS = ["FullPilot", "LaggyPilot", "NoisyPilot", "sensor_pilot"]

FIXED_ALPHA = 0.6
# LunarLander hands out exactly +100 on a successful landing step and -100 on a
# crash step. We split on the sign with a +/-50 margin so the terminal bonus is
# captured cleanly while ordinary shaping steps never trip the classifier.
SUCCESS_REWARD = 50.0    # terminal step reward above this == landed
CRASH_REWARD = -50.0     # terminal step reward below this == crashed

RESULTS_DIR = "results"


def make_pilots(full: FullPilot) -> dict:
    """Build the pilot lookup. LaggyPilot/NoisyPilot wrap the shared FullPilot."""
    return {
        "FullPilot": full,
        "LaggyPilot": LaggyPilot(full, lag_prob=0.8),
        "NoisyPilot": NoisyPilot(full, noise_prob=0.15),
        "sensor_pilot": sensor_pilot,
    }


def run_episode(env, pilot, copilot_model, controller, mode, episode_seed):
    """
    Run one evaluation episode and return the five slide-14 metrics:
        reward, success, crash, intervention_rate, mean_alpha
    """
    # Seed both the env and the pilot's (global np.random) stochasticity.
    np.random.seed(episode_seed)
    obs, _ = env.reset(seed=episode_seed)

    controller.reset()
    if hasattr(pilot, "reset"):
        pilot.reset()

    total_reward = 0.0
    last_reward = 0.0
    terminated = False
    steps = 0
    interventions = 0
    alphas = []

    done = False
    while not done:
        pilot_action = int(pilot(obs))

        if mode == "human_only":
            action = pilot_action
            intervened = 0

        elif mode == "autonomous":
            copilot_obs = build_copilot_obs(obs, pilot_action)
            q_values = get_q_values(copilot_model, copilot_obs)
            action = int(np.argmax(q_values))
            intervened = 1

        elif mode == "fixed_shared":
            action, info = controller.select_action(
                obs, pilot_action, copilot_model, alpha_override=FIXED_ALPHA
            )
            intervened = info["intervened"]
            alphas.append(info["alpha"])

        elif mode == "adaptive_shared":
            action, info = controller.select_action(obs, pilot_action, copilot_model)
            intervened = info["intervened"]
            alphas.append(info["alpha"])

        else:
            raise ValueError(f"Unknown mode: {mode}")

        obs, reward, terminated, truncated, _ = env.step(action)

        total_reward += reward
        last_reward = reward
        interventions += int(intervened)
        steps += 1
        done = terminated or truncated

    success = bool(terminated and last_reward > SUCCESS_REWARD)
    crash = bool(terminated and last_reward < CRASH_REWARD)
    intervention_rate = interventions / max(steps, 1)
    mean_alpha = float(np.mean(alphas)) if alphas else float("nan")

    return {
        "reward": float(total_reward),
        "success": success,
        "crash": crash,
        "intervention_rate": float(intervention_rate),
        "mean_alpha": mean_alpha,
    }


def run_grid(pilots, modes, seeds, episodes_per_seed):
    """Iterate over (pilot, mode, seed) cells, collecting per-episode rows."""
    full = FullPilot("models/full_pilot.zip")
    pilot_lookup = make_pilots(full)
    copilot_model = DQN.load("models/copilot.zip")

    env = SharedAutonomyEnv()
    controller = RiskAwareController()

    n_cells = len(pilots) * len(modes)
    per_cell = len(seeds) * episodes_per_seed
    total = n_cells * per_cell

    rows = []
    cell = 0
    start = time.time()
    for pilot_name in pilots:
        pilot = pilot_lookup[pilot_name]
        for mode in modes:
            for seed in seeds:
                for episode in range(episodes_per_seed):
                    # Distinct, reproducible seed per episode within the cell.
                    episode_seed = seed * 100 + episode
                    metrics = run_episode(
                        env, pilot, copilot_model, controller, mode, episode_seed
                    )
                    rows.append({
                        "pilot": pilot_name,
                        "mode": mode,
                        "seed": seed,
                        "episode": episode,
                        **metrics,
                    })

            # One-line progress update after each (pilot, mode) cell.
            cell += 1
            done_eps = cell * per_cell
            elapsed = time.time() - start
            rate = done_eps / elapsed if elapsed > 0 else 0.0
            eta = (total - done_eps) / rate if rate > 0 else 0.0
            cell_df = pd.DataFrame(rows[-per_cell:])
            print(
                f"[{cell:2d}/{n_cells}] {pilot_name:<12} {mode:<16} "
                f"reward={cell_df['reward'].mean():8.2f}  "
                f"success={cell_df['success'].mean():6.1%}  "
                f"| {done_eps}/{total} eps  elapsed={elapsed:5.1f}s  eta={eta:5.1f}s",
                flush=True,
            )

    env.close()
    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame):
    """Summary grouped by (pilot, mode) with reward, success/crash, intervention, alpha."""
    present_modes = [m for m in MODES if m in df["mode"].unique()]
    present_pilots = [p for p in PILOTS if p in df["pilot"].unique()]

    g = df.groupby(["pilot", "mode"], sort=False)
    summary = pd.DataFrame({
        "mean_reward": g["reward"].mean(),
        "std_reward": g["reward"].std(),
        "success_rate": g["success"].mean(),
        "crash_rate": g["crash"].mean(),
        "mean_intervention_rate": g["intervention_rate"].mean(),
        "mean_alpha": g["mean_alpha"].mean(),
    }).reset_index()

    # Order rows by the canonical pilot / mode ordering.
    summary["pilot"] = pd.Categorical(summary["pilot"], categories=present_pilots, ordered=True)
    summary["mode"] = pd.Categorical(summary["mode"], categories=present_modes, ordered=True)
    summary = summary.sort_values(["pilot", "mode"]).reset_index(drop=True)

    # Report-friendly formatted columns.
    display = pd.DataFrame({
        "pilot": summary["pilot"].astype(str),
        "mode": summary["mode"].astype(str),
        "reward (mean ± std)": [
            f"{m:.1f} ± {s:.1f}" for m, s in zip(summary["mean_reward"], summary["std_reward"])
        ],
        "success": [f"{v:.1%}" for v in summary["success_rate"]],
        "crash": [f"{v:.1%}" for v in summary["crash_rate"]],
        "intervention": [f"{v:.1%}" for v in summary["mean_intervention_rate"]],
        "mean_alpha": [
            "NaN" if pd.isna(v) else f"{v:.3f}" for v in summary["mean_alpha"]
        ],
    })

    print("\n=== Summary by (pilot, mode), aggregated over all seeds & episodes ===\n")
    try:
        print(display.to_markdown(index=False))
    except ImportError:
        print(display.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true",
                        help="Tiny smoke grid: 1 pilot x 4 modes x 1 seed x 1 episode.")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--out", default=None,
                        help="Output CSV path (overrides the default for the chosen grid).")
    args = parser.parse_args()

    if args.test:
        pilots = ["FullPilot"]
        modes = MODES
        seeds = [0]
        episodes = 1
        out_path = os.path.join(RESULTS_DIR, "eval_all_modes_test.csv")
    else:
        pilots = PILOTS
        modes = MODES
        seeds = list(range(args.seeds))
        episodes = args.episodes
        out_path = os.path.join(RESULTS_DIR, "eval_all_modes.csv")

    if args.out is not None:
        out_path = args.out

    total = len(pilots) * len(modes) * len(seeds) * episodes
    print(f"Running {total} episodes: "
          f"{len(pilots)} pilots x {len(modes)} modes x {len(seeds)} seeds x {episodes} episodes")

    df = run_grid(pilots, modes, seeds, episodes)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows to {out_path}")

    print_summary(df)


if __name__ == "__main__":
    main()
