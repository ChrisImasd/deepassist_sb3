"""
Publication-quality trace plot of the adaptive controller's internal dynamics
over one representative sensor_pilot episode (the slide-15 figure).

Outputs:
  results/alpha_trace_sensor.pdf / .png        — full two-panel figure
  results/alpha_trace_sensor_simple.pdf        — top panel only, half width

Run:  python plot_alpha_trace.py
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from stable_baselines3 import DQN

import risk_aware_policy
from risk_aware_policy import risk_aware_action
from copilot_env import CopilotEnv
from pilot_policies import sensor_pilot

RESULTS_DIR = "results"
FIXED_ALPHA = 0.6
SUCCESS_REWARD = 50.0
CRASH_REWARD = -50.0

ACTION_NAMES = [
    "off·left", "off·center", "off·right",
    "on·left", "on·center", "on·right",
]


def collect_episode(env, copilot_model, seed):
    """Run one sensor_pilot episode through risk_aware_action; return per-step trace."""
    risk_aware_policy.reset()                 # clear the confidence window
    np.random.seed(seed)
    obs, _ = env.reset(seed=seed)             # sensor_pilot has no reset()

    risk, rho, alpha = [], [], []
    a_pilot, a_exec, intervened = [], [], []
    last_reward = 0.0
    terminated = False
    done = False

    while not done:
        world_obs = obs[:9]
        pilot_action = int(np.argmax(obs[9:15]))

        action, info = risk_aware_action(world_obs, pilot_action, copilot_model)

        risk.append(info["risk"])
        rho.append(info["reliability"])
        alpha.append(info["alpha"])
        a_pilot.append(info["pilot_action"])
        a_exec.append(info["final_action"])
        intervened.append(bool(info["intervened"]))

        obs, reward, terminated, truncated, _ = env.step(action)
        last_reward = reward
        done = terminated or truncated

    return {
        "seed": seed,
        "risk": np.array(risk),
        "rho": np.array(rho),
        "alpha": np.array(alpha),
        "a_pilot": np.array(a_pilot),
        "a_exec": np.array(a_exec),
        "intervened": np.array(intervened),
        "length": len(risk),
        "success": bool(terminated and last_reward > SUCCESS_REWARD),
        "crash": bool(terminated and last_reward < CRASH_REWARD),
    }


def draw_top_panel(ax, ep, legend=True):
    """Draw the r_t / rho_t / alpha_t panel with fixed-baseline shading + markers."""
    t = np.arange(ep["length"])
    alpha = ep["alpha"]

    # Shade adaptive-vs-fixed difference.
    ax.fill_between(t, alpha, FIXED_ALPHA, where=(alpha < FIXED_ALPHA),
                    interpolate=True, color="red", alpha=0.15)
    ax.fill_between(t, alpha, FIXED_ALPHA, where=(alpha > FIXED_ALPHA),
                    interpolate=True, color="green", alpha=0.15)

    ax.plot(t, ep["risk"], color="tab:blue", solid_capstyle="round",
            label=r"Landing risk $r_t$")
    ax.plot(t, ep["rho"], color="tab:orange", solid_capstyle="round",
            label=r"Input confidence $\rho_t$")
    ax.plot(t, alpha, color="tab:green", linewidth=2,
            label=r"Adaptive tolerance $\alpha_t$")
    ax.axhline(FIXED_ALPHA, color="grey", linestyle="--", linewidth=1,
               label=r"Fixed $\alpha = 0.6$")

    # Intervention markers along the top.
    interv_t = t[ep["intervened"]]
    ax.plot(interv_t, np.full_like(interv_t, 0.97, dtype=float),
            linestyle="None", marker="^", color="red", markersize=4,
            alpha=0.6, label="Intervention")

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Normalized value")
    if legend:
        ax.legend(loc="upper right", fontsize=9, ncol=1)


def draw_action_panel(ax, ep):
    """Draw pilot command vs executed action, red where the copilot overrides."""
    t = np.arange(ep["length"])
    interv = ep["intervened"]

    ax.step(t, ep["a_pilot"], where="post", color="0.55", linewidth=1.6,
            label="Pilot command")

    exec_override = np.where(interv, ep["a_exec"].astype(float), np.nan)
    ax.step(t, exec_override, where="post", color="tab:red", linewidth=1.8,
            label="Copilot override")
    ax.plot(t[interv], ep["a_exec"][interv], linestyle="None", marker="o",
            color="tab:red", markersize=3, alpha=0.7)

    ax.set_ylim(-0.5, 5.5)
    ax.set_yticks(range(6))
    ax.set_yticklabels(ACTION_NAMES)
    ax.set_ylabel("Action")
    ax.set_xlabel("Timestep")
    ax.legend(loc="upper right", fontsize=9)


def make_full_figure(ep, path_pdf, path_png):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 2]},
    )
    draw_top_panel(ax1, ep, legend=True)
    # Plain underscore renders cleanly in matplotlib's default (non-usetex) title;
    # the LaTeX-style "\_" would show a literal backslash here.
    ax1.set_title(
        "Adaptive Intervention During One Representative Episode (sensor_pilot)"
    )
    draw_action_panel(ax2, ep)
    fig.tight_layout()
    fig.savefig(path_pdf)
    fig.savefig(path_png, dpi=300)
    plt.close(fig)


def make_simple_figure(ep, path_pdf):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    draw_top_panel(ax, ep, legend=True)
    ax.set_xlabel("Timestep")
    fig.tight_layout()
    fig.savefig(path_pdf)
    plt.close(fig)


def main():
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3

    copilot_model = DQN.load("models/copilot.zip")
    env = CopilotEnv([sensor_pilot])

    # Episode selection: longest-surviving sensor_pilot episode over seeds 0..19,
    # lowest seed as tiebreaker (only replace on a strictly longer episode).
    best = None
    for seed in range(20):
        ep = collect_episode(env, copilot_model, seed)
        if best is None or ep["length"] > best["length"]:
            best = ep
    env.close()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    pdf = os.path.join(RESULTS_DIR, "alpha_trace_sensor.pdf")
    png = os.path.join(RESULTS_DIR, "alpha_trace_sensor.png")
    simple_pdf = os.path.join(RESULTS_DIR, "alpha_trace_sensor_simple.pdf")

    make_full_figure(best, pdf, png)
    make_simple_figure(best, simple_pdf)

    n_interv = int(best["intervened"].sum())
    print(f"Selected seed:        {best['seed']} "
          f"(longest survival over seeds 0-19, lowest-seed tiebreak)")
    print(f"Episode length:       {best['length']} timesteps")
    print(f"Intervention steps:   {n_interv} "
          f"(rate {n_interv / best['length']:.1%})")
    print(f"Mean r_t:             {best['risk'].mean():.3f}")
    print(f"Mean rho_t:           {best['rho'].mean():.3f}")
    print(f"Mean alpha_t:         {best['alpha'].mean():.3f}")
    outcome = "success" if best["success"] else ("crash" if best["crash"] else "neither")
    print(f"Episode outcome:      {outcome}")

    print("\nFiles written:")
    for p in (pdf, png, simple_pdf):
        print(f"  {p:42s} exists={os.path.exists(p)}  size={os.path.getsize(p):,d}B")


if __name__ == "__main__":
    main()
