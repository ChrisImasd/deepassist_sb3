# deepassist-sb3: Codebase Review and Modification Analysis

## Overview and Project Context

This codebase is a complete modernization port of the original **deepassist** shared-autonomy research project (circa 2018) to current Python RL tooling. The original code was built on:

- **OpenAI Gym** (`gym`) — the old RL environment interface
- **TF1 Baselines** — DeepMind/OpenAI's original TensorFlow 1.x RL library
- **Python 3.6-era** conventions

The port targets:

- **Gymnasium** (`gymnasium`) — the Farama Foundation's maintained fork of Gym, with a cleaner and stricter API
- **Stable-Baselines3 (SB3)** — a PyTorch rewrite of Baselines, actively maintained and well-documented
- **Modern Python 3.10+** idioms

The scientific goal is a **shared-autonomy Lunar Lander**: a setting where a human pilot (modeled by imperfect heuristic policies) gives action proposals, and an AI copilot (trained via RL) takes over the actual control, ideally compensating for the pilot's imperfections while respecting their intent.

---

## Critical API-Level Changes Across All Files

Before discussing individual scripts, it is worth explaining the library-level migrations that touched every file. These are the foundation of what makes the old code incompatible with modern libraries.

### 1. `gym` → `gymnasium` (namespace and API contract)

The Farama Foundation forked OpenAI Gym into `gymnasium` and hardened the API. All imports were updated:

```python
# Old
import gym
from gym import spaces

# New
import gymnasium as gym
from gymnasium import spaces
```

More importantly, the **step and reset contracts changed**:

| API call | Old `gym` return | New `gymnasium` return |
|---|---|---|
| `env.reset()` | `obs` | `(obs, info)` |
| `env.step(action)` | `(obs, reward, done, info)` | `(obs, reward, terminated, truncated, info)` |

The `done` flag was split into two separate booleans:
- `terminated`: the episode ended due to a natural terminal condition (e.g., the lander crashed or landed successfully)
- `truncated`: the episode was cut short by a time limit

Every loop in every file was updated from:

```python
# Old
obs = env.reset()
obs, reward, done, info = env.step(action)
```

to:

```python
# New
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
done = terminated or truncated
```

This matters because conflating time-limit truncation with true termination can bias value function estimates in RL. SB3 handles this correctly internally but requires the proper split signals.

### 2. Environment ID versioning (`-v2` → `-v3`)

Environment registrations changed between Gym and Gymnasium versions:

```python
# Old
gym.make("LunarLanderContinuous-v2")
gym.make("LunarLander-v2")

# New
gym.make("LunarLanderContinuous-v3")
gym.make("LunarLander-v3")
```

The `-v3` variants use Gymnasium's updated physics and seeding infrastructure. The physics behavior is largely identical but the environment class internals differ.

### 3. TF1 Baselines → Stable-Baselines3

The original baselines API was stateful, TensorFlow-based, and had a very different calling convention. SB3 is PyTorch-based and uses a cleaner object-oriented model. Key substitutions:

| Concern | Old TF1 Baselines | Stable-Baselines3 |
|---|---|---|
| Algorithm import | `from baselines import deepq` | `from stable_baselines3 import DQN` |
| Model creation | Functional, many positional args | `DQN("MlpPolicy", env, ...)` |
| Training | `deepq.learn(...)` | `model.learn(total_timesteps=...)` |
| Saving | Custom checkpoint system | `model.save("path.zip")` |
| Loading | Custom restore | `DQN.load("path.zip")` |
| Inference | Custom `act` function | `model.predict(obs, deterministic=True)` |
| Network arch | `hiddens=[64, 64]` | `policy_kwargs=dict(net_arch=[64, 64])` |

The `model.predict()` call returns a tuple `(action, state)` — the action must be extracted from index 0, which is a subtle difference that was applied throughout.

### 4. `np_random` seeding

Gymnasium's `Wrapper` base class provides a `self.np_random` attribute — a `numpy.random.Generator` instance that is properly seeded when `env.reset(seed=...)` is called. The old codebase used bare `np.random.*` calls, which are not seed-controlled through the environment interface. All stochastic sampling in the wrappers was migrated to use `self.np_random`.

---

## File-by-File Analysis

---

### 1. `smoke_test.py`

**Role:** Initial integration test. Validates that Gymnasium and SB3 are installed correctly and communicate with each other before any project-specific code is written.

**What the code does:**

```python
import gymnasium as gym
from stable_baselines3 import DQN

env = gym.make("LunarLander-v3")
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=5000)
env.close()

eval_env = gym.make("LunarLander-v3", render_mode="human")
for episode in range(3):
    obs, _ = eval_env.reset()
    ...
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = eval_env.step(action)
    done = terminated or truncated
```

This script:
1. Creates the standard discrete `LunarLander-v3` environment (not the continuous variant, not the custom wrapper)
2. Instantiates a DQN model with SB3's string-based policy specifier `"MlpPolicy"`
3. Trains for 5,000 timesteps (not enough to produce a good policy, but sufficient to confirm the training loop runs without errors)
4. Opens a second environment with `render_mode="human"` for visual playback
5. Runs 3 deterministic evaluation episodes

**Significance of changes:**

This file exists precisely because of the migration. The old codebase had no such test, and verifying that `gymnasium` + `SB3` work end-to-end without any project-specific code was an essential first step. Several things that could go wrong were checked here:

- The Gymnasium step API (`terminated, truncated` unpacking) is used correctly
- `env.reset()` returns `(obs, info)` — the `_` discard is explicit
- `model.predict()` returns `(action, _)` — the state output is discarded
- `render_mode="human"` is the correct Gymnasium parameter name (not `render=True` as in old Gym)

The choice to use `LunarLander-v3` (discrete) rather than `LunarLanderContinuous-v3` here is intentional: it lets DQN work directly with no action-space adapter. The continuous environment is introduced only in the next file.

---

### 2. `shared_autonomy_env.py`

**Role:** The core environment wrapper. This is the most important file in the codebase — it defines the fundamental observation and action interface that everything else builds on.

**What the code does:**

`SharedAutonomyEnv` is a `gymnasium.Wrapper` around `LunarLanderContinuous-v3`. It does two things:

**A. Discretizes the action space**

The underlying continuous environment expects actions in `Box([-1, -1], [1, 1])` — two floats representing main engine throttle and side thruster steering. DQN cannot operate on a continuous action space directly, so the wrapper maps `Discrete(6)` integer actions to fixed continuous vectors via `disc_to_cont`:

```
action // 3 → main engine:   0-2 = off (-0.75),  3-5 = on (+0.75)
action  % 3 → steering:      0 = left (-0.75), 1 = center (0.0), 2 = right (+0.75)

  0 → [-0.75, -0.75]   3 → [+0.75, -0.75]
  1 → [-0.75,  0.00]   4 → [+0.75,  0.00]
  2 → [-0.75, +0.75]   5 → [+0.75, +0.75]
```

The throttle magnitude `THROTTLE_MAG = 0.75` was chosen to be strong enough for effective control while avoiding full saturation. Action 1 (main off, no steer) serves as "noop" — it was not set to 0 because action 0 fires the left thruster.

**B. Extends the observation with a goal coordinate**

The standard LunarLander observation is 8-dimensional. This wrapper appends a 9th element, `obs[8]`, which is a target landing x-coordinate sampled uniformly from `[-0.8, 0.8]` at the start of each episode:

```python
def reset(self, seed=None, options=None):
    obs, info = self.env.reset(seed=seed, options=options)
    self._goal_x = float(self.np_random.uniform(-0.8, 0.8))
    return self._augment(obs), info
```

The 9-dim observation layout is:

| Index | Meaning |
|---|---|
| 0 | Lander x position |
| 1 | Lander y position |
| 2 | Lander x velocity |
| 3 | Lander y velocity |
| 4 | Lander angle |
| 5 | Lander angular velocity |
| 6 | Left leg contact (bool) |
| 7 | Right leg contact (bool) |
| 8 | Goal x-coordinate (constant within episode) |

The `observation_space` is updated to a 9-element `Box` matching the augmented observation. The physics helipad is always at x=0, so the goal x and the actual landing target are intentionally decoupled — this is a feature of the shared-autonomy setting, not a bug. The `using_lander_reward_shaping` flag is stored for future reward-shaping that would penalize landing far from `obs[8]`.

**Significance of changes:**

This file represents the deepest architectural decision of the port. In the original codebase:
- The environment was likely using `gym.make("LunarLanderContinuous-v2")`
- The wrapper concept was either informal or not a proper `gym.Wrapper` subclass
- The reset/step contracts used the old 4-tuple and single-return forms

The key modernizations:
1. **Proper `gymnasium.Wrapper` inheritance** — gives the wrapper `self.np_random`, proper space propagation, and correct `render()` passthrough automatically
2. **`self.np_random.uniform(-0.8, 0.8)` instead of `np.random.uniform(...)`** — goal sampling is now seed-controlled through the environment interface, enabling reproducible experiments
3. **`reset(seed=None, options=None)` signature** — Gymnasium requires these keyword arguments to be accepted and forwarded; old `gym` wrappers often didn't need them
4. **Explicit `observation_space` and `action_space` overrides** — replacing whatever informal duck-typing the old code used with proper `spaces.Box` and `spaces.Discrete` declarations that SB3 reads to configure its networks

---

### 3. `pilot_policies.py`

**Role:** A library of simulated human pilot behaviors. These model different kinds of imperfect human operators for use in shared-autonomy training and evaluation.

**What the code does:**

All pilots share the same callable interface: `(obs: np.ndarray) -> int` (returning an action 0–5 from `Discrete(6)`).

**`RandomPilot`**
Returns a uniformly random action. Represents a completely uninformed operator. Used as a baseline and as the base policy for other wrappers.

**`LaggyPilot(base_policy, lag_prob=0.8)`**
Wraps any base policy and introduces **action persistence** (stickiness). On each step, with probability `lag_prob=0.8`, it repeats its last action instead of querying the base policy. This models a human with slow reaction time. The critical design decision: `reset()` must be called at episode boundaries to clear `_last_action = None`. If it is not called, the previous episode's final action bleeds into the new episode's first step.

```python
def __call__(self, obs):
    if self._last_action is None or np.random.random() >= self.lag_prob:
        self._last_action = self.base_policy(obs)
    return self._last_action
```

**`NoisyPilot(base_policy, noise_prob=0.15)`**
Applies two **independent noise passes** to the base policy's action:
1. With probability `noise_prob`, flip the main engine state: `action = (action + 3) % 6` (toggles between the `0-2` block and the `3-5` block)
2. With probability `noise_prob`, randomize steering: `action = action // 3 * 3 + (action + randint(1,3)) % 3` (preserves the engine state, picks a different steering direction)

The two-pass independence is important: both can fire simultaneously, which can result in a doubly-corrupted action (wrong engine AND wrong steering).

**`noop_pilot`**
A plain function returning `1` always. Action 1 is `[-0.75, 0.0]` — main engine off, no steering. This represents an unresponsive or passive operator. Action 0 was deliberately avoided as "noop" because action 0 fires the left thruster.

**`sensor_pilot`**
A simple heuristic that uses the goal coordinate:

```python
def sensor_pilot(obs, thresh=0.1):
    d = obs[8] - obs[0]   # goal x minus lander x
    if d < -thresh: return 0   # steer left
    elif d > thresh: return 2  # steer right
    return 1                   # close enough
```

This pilot only steers — it never fires the main engine — modeling a human who understands where to go but has no intuition for altitude management.

**Significance of changes:**

The original codebase used different abstractions for its pilot policies, tied to TF1 or gym-specific types. The modernization stripped those dependencies entirely and replaced them with:
- **Pure NumPy callables** — no gym, no TF, no SB3 dependency in this file
- **Consistent `(obs: np.ndarray) -> int` interface** — allows any pilot to be dropped into any environment wrapper without adaptation
- **Stateless vs stateful distinction is explicit** — `LaggyPilot` has `reset()`, stateless pilots do not. Code that iterates over pilots can check `hasattr(pilot, "reset")` to handle boundaries correctly
- **No `gym.Space.sample()` usage** — the original `RandomPilot` likely used `env.action_space.sample()`, coupling it to a live environment object. Now it uses `np.random.randint(6)` with no environment dependency

---

### 4. `full_pilot.py`

**Role:** A wrapper that loads a trained DQN checkpoint and exposes it through the same callable interface as the handwritten pilot policies.

**What the code does:**

```python
class FullPilot:
    def __init__(self, model_path="models/full_pilot.zip"):
        self.model = DQN.load(model_path)

    def reset(self):
        pass

    def __call__(self, obs: np.ndarray) -> int:
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)
```

`DQN.load()` is SB3's checkpoint restoration method. It reads the `.zip` archive written by `model.save()` and reconstructs the full model — policy network weights, optimizer state, hyperparameters, and observation/action space metadata.

`reset()` is a no-op because DQN is Markovian: it has no recurrent state or episode-level memory to clear between episodes. The method exists only for API consistency with `LaggyPilot`, so calling code can uniformly call `pilot.reset()` without checking which pilot type it has.

`model.predict(obs, deterministic=True)` returns `(action_array, state)`. The `deterministic=True` flag disables exploration: the policy takes the greedy action under the current Q-values rather than epsilon-greedy sampling. The `int()` cast is required because SB3 returns a NumPy scalar, not a Python int, and downstream code expects a plain int.

**Significance of changes:**

The original deepassist used a TF1 Baselines `act` function returned from `deepq.learn()`. That function was a TensorFlow computation graph node — it was not serializable, had to be kept alive alongside the TF session, and could not be cleanly reloaded in a separate process. Replacing it with SB3's `DQN.load()` + `model.predict()` gives:

- **Portable checkpoints** — `.zip` files can be copied, versioned, and loaded in any Python environment with SB3 installed
- **Clean separation of training and inference** — `FullPilot` has no knowledge of how the model was trained
- **Identical interface to hand-coded pilots** — the RL-trained pilot and the heuristic pilots are interchangeable in all downstream code

---

### 5. `train_full_pilot.py`

**Role:** Trains the `FullPilot` DQN model on `SharedAutonomyEnv`. This is the first stage of the two-stage training pipeline.

**What the code does:**

The script has two modes — if a checkpoint already exists, the user is prompted to evaluate-only or retrain. This prevents accidental overwrites.

**DQN configuration:**

```python
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
```

Parameter-by-parameter explanation:

| Parameter | Value | Rationale |
|---|---|---|
| `"MlpPolicy"` | — | Two-layer fully-connected network. SB3 infers input/output sizes from the environment spaces. |
| `learning_rate` | `1e-3` | Standard Adam learning rate for DQN on medium-complexity tasks |
| `buffer_size` | `50_000` | Replay buffer capacity. Keeps memory manageable while retaining enough diversity |
| `learning_starts` | `1_000` | Steps of pure random exploration before any gradient updates. Ensures the buffer is non-trivially populated before training starts |
| `batch_size` | `32` | Minibatch size drawn from the replay buffer each gradient step |
| `gamma` | `0.99` | Discount factor. High value is appropriate for episodic tasks where terminal reward matters |
| `train_freq` | `1` | Update the online network every step (not every N steps). Aggressive but works for this environment |
| `target_update_interval` | `1_500` | How many steps between hard-copying the online network weights to the target network. Too-frequent updates destabilize training; 1500 is a conservative choice |
| `exploration_fraction` | `0.1` | Fraction of total training steps spent linearly decaying epsilon from 1.0 to `exploration_final_eps` |
| `exploration_final_eps` | `0.02` | Final epsilon. 2% random actions retained forever to prevent policy collapse |
| `net_arch` | `[64, 64]` | Two hidden layers of 64 neurons each. Sufficient for this 9-dim observation |

**Training:**
```python
model.learn(total_timesteps=TOTAL_TIMESTEPS, log_interval=10)
model.save(MODEL_PATH)
```

`TOTAL_TIMESTEPS = 500_000` was chosen to give the agent enough experience to develop a competent landing policy on the 9-dim observation space.

**Evaluation:**
```python
def evaluate(model, env, n_episodes=5):
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
```

Note `deterministic=True` during evaluation — this gives a fair measurement of the greedy policy, not the exploratory one. The 5-episode evaluation prints per-episode and mean rewards.

**Significance of changes:**

The original training script used TF1 Baselines' functional interface:

```python
# Old (approximate reconstruction)
act = deepq.learn(
    env,
    network="mlp",
    hiddens=[64, 64],
    lr=1e-3,
    total_timesteps=500_000,
    ...
)
```

This interface:
- Required TensorFlow 1.x sessions to be active during the entire training
- Returned an `act` function that was not persistable across processes
- Had different parameter names and conventions (`hiddens` vs `net_arch`, etc.)
- Did not have the `terminated`/`truncated` distinction — the environment returned the old 4-tuple

The SB3 replacement:
- Runs on PyTorch with no session management
- Saves a portable `.zip` checkpoint at any point
- Accepts the Gymnasium 5-tuple contract natively
- Provides a `verbose=1` progress log that prints rolling statistics during training
- The `evaluate()` helper function is new — the original likely didn't have a clean evaluation path separate from training

---

### 6. `watch_pilot.py`

**Role:** Visual evaluation and video recording tool for the trained `FullPilot`.

**What the code does:**

The script has two modes selected at runtime via command-line arguments:

**Live rendering mode:**
```python
env = SharedAutonomyEnv(render_mode="human")
```
Opens an SDL/OpenGL window showing the simulation in real-time.

**Recording mode:**
```python
env = RecordVideo(
    SharedAutonomyEnv(render_mode="rgb_array"),
    video_folder=video_dir,
    episode_trigger=lambda _: True,
    name_prefix="full_pilot",
)
```
`RecordVideo` is a Gymnasium wrapper that captures frames from `render_mode="rgb_array"` and encodes them to MP4. `episode_trigger=lambda _: True` records every episode (not just periodic ones). Each episode produces a file named `full_pilot-episode-NNN.mp4`.

**Video merging:**
```python
from moviepy import VideoFileClip, concatenate_videoclips  # v2
# fallback:
from moviepy.editor import VideoFileClip, concatenate_videoclips  # v1
```

After all episodes are recorded, `merge_recordings()` uses MoviePy to concatenate all per-episode MP4s into a single `full_pilot-merged.mp4`. The `try/except` import handles both MoviePy v1 (import path `moviepy.editor`) and v2 (flat `moviepy` namespace), since the package changed its import structure between major versions.

**CLI interface:**
```python
parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--record", action="store_true")
parser.add_argument("--video-dir", default="recordings")
```

Usage examples:
```
python watch_pilot.py                          # 10 live episodes
python watch_pilot.py --episodes 3            # 3 live episodes
python watch_pilot.py --record                # record to recordings/
python watch_pilot.py --record --video-dir out # record to out/
```

**Significance of changes:**

The original deepassist had a Jupyter notebook (`1.0-lunarlander-sim.ipynb`) for visualization, which is not suitable for headless or automated evaluation. The modernization:
- Replaces notebook cells with a proper CLI script
- Uses Gymnasium's `RecordVideo` wrapper (the old `gym.wrappers.Monitor` was deprecated in Gym 0.26 and removed in Gymnasium; `RecordVideo` is its replacement)
- Adds MoviePy for video merging — useful for producing clean demo videos from multi-episode evaluation runs
- `render_mode` is now a constructor parameter (`gym.make(..., render_mode="human")`), not a runtime `.render()` call — this is a Gymnasium API change from old Gym where rendering mode was set separately

The MoviePy fallback import is a specific defensive measure: MoviePy broke its import path between v1 and v2, and this two-branch try/except makes the script work regardless of which version is installed.

---

### 7. `copilot_env.py`

**Role:** The second-layer environment wrapper. Extends `SharedAutonomyEnv` with the pilot's proposed action included in the observation, creating the full shared-autonomy observation space for copilot training.

**What the code does:**

`CopilotEnv` wraps `SharedAutonomyEnv` (which in turn wraps `LunarLanderContinuous-v3`). It adds a one-hot encoding of the pilot's most recent proposed action to the observation:

**Observation space (15-dim):**

| Indices | Content |
|---|---|
| `[0:9]` | World state from `SharedAutonomyEnv` (8 lander states + goal x) |
| `[9:15]` | One-hot encoding of the pilot's proposed action (6 possible actions) |

The one-hot encoding converts the pilot's integer action (0–5) into a 6-element vector with a single `1.0` and five `0.0`s:

```python
def _one_hot(action: int, n: int = 6) -> np.ndarray:
    v = np.zeros(n, dtype=np.float32)
    v[action] = 1.0
    return v
```

Using one-hot rather than a raw integer is important: it prevents the network from inferring spurious ordinal relationships between actions (e.g., action 3 is not "3 times as much" as action 1).

**Pilot randomization per episode:**

```python
def reset(self, seed=None, options=None):
    world_obs, info = self.env.reset(seed=seed, options=options)
    idx = int(self.np_random.integers(0, len(self.pilots)))
    self.current_pilot = self.pilots[idx]
    if hasattr(self.current_pilot, "reset"):
        self.current_pilot.reset()
    self._last_pilot_action = int(self.current_pilot(world_obs))
    return self._build_obs(world_obs, self._last_pilot_action), info
```

At the start of each episode, one pilot is drawn at random from the provided list. The copilot therefore trains against a mixture of pilot behaviors rather than specializing to a single type. `self.np_random.integers(0, len(self.pilots))` uses the environment's seeded RNG for reproducible pilot sampling.

**Step execution:**

```python
def step(self, copilot_action: int):
    world_obs, reward, terminated, truncated, info = self.env.step(copilot_action)
    self._last_pilot_action = int(self.current_pilot(world_obs))
    return self._build_obs(world_obs, self._last_pilot_action), reward, terminated, truncated, info
```

Note that `copilot_action` goes directly to the underlying environment — the pilot's proposed action is **information only** and does not influence the physics. The copilot is in full control; it sees what the pilot would have done but chooses its own action.

The pilot sees the post-step world observation to decide its next proposed action. This creates the correct temporal ordering: the copilot acts, the world updates, the pilot proposes its next action based on the new state, and that proposal is included in the observation the copilot will see on the next step.

**Significance of changes:**

This file has no direct equivalent in the original codebase — the shared-autonomy copilot training infrastructure was either incomplete or structured very differently. The key design choices made here:

1. **Stacked wrappers rather than a monolithic environment** — `CopilotEnv → SharedAutonomyEnv → LunarLanderContinuous-v3` is a clean chain. Each layer adds exactly one concern. This is idiomatic Gymnasium wrapper design and makes each layer independently testable.

2. **Pilot sampling is seeded through `self.np_random`** — consistent with Gymnasium's seeding contract. If `env.reset(seed=42)` is called, both the world physics and the pilot assignment are deterministic.

3. **`hasattr(pilot, "reset")` check** — stateless pilots (functions like `noop_pilot`) and stateful ones (`LaggyPilot`) are handled uniformly without requiring a common base class.

4. **Observation space declared as `Box`** — SB3 reads this declaration to set the input size of the policy network's first layer. Getting this wrong would silently produce a network that crashes or produces garbage on the first forward pass.

5. **Action space unchanged** — the copilot's action space is still `Discrete(6)` inherited from `SharedAutonomyEnv`. The wrapper does not need to modify it.

---

### 8. `train_copilot.py`

**Role:** Trains the copilot DQN on `CopilotEnv`. This is the second and final stage of the training pipeline.

**What the code does:**

**Training environment construction:**

```python
def make_training_env():
    full = FullPilot()
    return CopilotEnv([
        LaggyPilot(full, lag_prob=0.8),
        NoisyPilot(full, noise_prob=0.15),
        noop_pilot,
        sensor_pilot,
    ])
```

Four pilot types are used during training, all based on the trained `FullPilot` as the underlying rational agent (except `noop_pilot` and `sensor_pilot` which are pure heuristics). The copilot therefore learns to compensate for:
- Slow, sticky actions (laggy)
- Random corruptions to engine and steering (noisy)
- Complete passivity (noop)
- Steering-only control with no altitude management (sensor)

**DQN configuration** is identical to `train_full_pilot.py`, which is intentional — the same network architecture and hyperparameters are used for both stages, making the comparison between pilot and copilot performance fair and the hyperparameter set minimal.

**Evaluation strategy:**

```python
eval_pilots = [
    ("LaggyPilot  (trained on)", LaggyPilot(full, lag_prob=0.8)),
    ("NoisyPilot  (trained on)", NoisyPilot(full, noise_prob=0.15)),
    ("noop_pilot  (trained on)", noop_pilot),
    ("sensor_pilot(trained on)", sensor_pilot),
    ("FullPilot   (unseen)    ", FullPilot()),
]
```

The copilot is evaluated against all four training pilots **plus** a perfect `FullPilot` that was never seen during training. This tests generalization: does the copilot learn to recognize and complement competent input as well as impaired input? Each pilot gets 20 evaluation episodes with `deterministic=True` to remove exploration noise.

The evaluation uses **separate `CopilotEnv` instances** per pilot (each wrapping a `[pilot]` singleton list), not the training environment. This isolates the measurement from any environment state accumulated during training.

**Significance of changes:**

This file is the culmination of the entire modernization effort. Key points:

1. **Two-stage training pipeline** — `train_full_pilot.py` produces a competent pilot model; `train_copilot.py` uses that model to generate impaired pilot behaviors to train the copilot against. This separation mirrors the research design: the copilot should assist pilots that have the right *intention* (modeled by FullPilot's Q-values) but impaired *execution* (lag, noise, passivity).

2. **`FullPilot` is an out-of-distribution test** — including an unimpaired `FullPilot` in the evaluation (but not training) is a deliberate probe of generalization. A well-trained copilot should either defer to or complement a perfect pilot, not fight it.

3. **`make_training_env()` factory function** — creates a fresh environment with fresh pilot instances each call. This is important because `LaggyPilot` is stateful; reusing the same instance across training and evaluation runs would create cross-contamination.

4. **Evaluation results are printed in a formatted table** — a small but useful quality-of-life addition that makes reading results fast:

```
  Pilot                               Mean reward
  ----------------------------------- ------------
  LaggyPilot  (trained on)              ...
  NoisyPilot  (trained on)              ...
  noop_pilot  (trained on)              ...
  sensor_pilot(trained on)              ...
  FullPilot   (unseen)                  ...
```

---

## Architecture Summary

The full system forms a clean three-layer hierarchy:

```
LunarLanderContinuous-v3          ← Gymnasium physics engine
         ↓
SharedAutonomyEnv                 ← Discrete(6) actions, 9-dim obs (+ goal x)
         ↓
CopilotEnv                        ← 15-dim obs (+ pilot one-hot), pilot mixture
```

Training proceeds in two stages:

```
Stage 1: train_full_pilot.py
  → Trains DQN on SharedAutonomyEnv (9-dim obs)
  → Saves models/full_pilot.zip
  → Produces FullPilot — a competent autonomous lander

Stage 2: train_copilot.py
  → Constructs impaired pilots from FullPilot (Laggy, Noisy, Noop, Sensor)
  → Trains DQN on CopilotEnv (15-dim obs)
  → Saves models/copilot.zip
  → Produces a copilot that generalizes across pilot impairment types
```

Evaluation and visualization:
```
watch_pilot.py    → Watch / record FullPilot episodes
pilot_policies.py → Benchmark all pilot types in one run
smoke_test.py     → Verify SB3 + Gymnasium installation
```

---

## Summary of All Modernization Changes

| Category | Old | New |
|---|---|---|
| Import | `import gym` | `import gymnasium as gym` |
| Env ID | `LunarLander-v2`, `LunarLanderContinuous-v2` | `LunarLander-v3`, `LunarLanderContinuous-v3` |
| Reset return | `obs = env.reset()` | `obs, info = env.reset()` |
| Step return | `obs, reward, done, info = env.step(a)` | `obs, reward, terminated, truncated, info = env.step(a)` |
| Done logic | `if done:` | `done = terminated or truncated` |
| RL library | TF1 Baselines (`deepq.learn`) | SB3 (`DQN("MlpPolicy", env, ...)`) |
| Model save | Custom TF checkpoint | `model.save("path.zip")` |
| Model load | TF session restore | `DQN.load("path.zip")` |
| Inference | `act(obs)` graph node | `model.predict(obs, deterministic=True)` |
| Network arch | `hiddens=[64, 64]` | `policy_kwargs=dict(net_arch=[64, 64])` |
| Seeded RNG | `np.random.*` directly | `self.np_random.*` (Gymnasium seeding) |
| Visualization | Jupyter notebook | CLI (`watch_pilot.py`) + `RecordVideo` wrapper |
| Recording | `gym.wrappers.Monitor` (deprecated) | `gymnasium.wrappers.RecordVideo` |
| Action discretization | Ad-hoc or manual | `disc_to_cont()` + `Discrete(6)` action space override |
| Pilot interface | TF-coupled | Pure NumPy callables, framework-independent |
