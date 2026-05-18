"""Watch the trained FullPilot fly the Lunar Lander.

Usage:
    python watch_pilot.py                          # 10 episodes, live window
    python watch_pilot.py --episodes 3             # 3 episodes, live window
    python watch_pilot.py --record                 # record + merge to recordings/full_pilot-merged.mp4
    python watch_pilot.py --record --video-dir out # record to out/
"""

import argparse
from pathlib import Path

import numpy as np
from gymnasium.wrappers import RecordVideo

from shared_autonomy_env import SharedAutonomyEnv
from full_pilot import FullPilot

try:
    from moviepy import VideoFileClip, concatenate_videoclips  # moviepy v2
except ImportError:
    from moviepy.editor import VideoFileClip, concatenate_videoclips  # moviepy v1


def merge_recordings(video_dir: str, out_name: str = "full_pilot-merged.mp4"):
    clip_paths = sorted(Path(video_dir).glob("full_pilot-episode-*.mp4"))
    if not clip_paths:
        print("No episode files found to merge.")
        return

    out_path = Path(video_dir) / out_name
    clips = [VideoFileClip(str(p)) for p in clip_paths]
    merged = concatenate_videoclips(clips)
    merged.write_videofile(str(out_path), logger=None)
    for c in clips:
        c.close()
    merged.close()
    print(f"Merged video saved to {out_path}")


def watch(n_episodes: int = 10, record: bool = False, video_dir: str = "recordings"):
    if record:
        Path(video_dir).mkdir(exist_ok=True)
        env = RecordVideo(
            SharedAutonomyEnv(render_mode="rgb_array"),
            video_folder=video_dir,
            episode_trigger=lambda _: True,
            name_prefix="full_pilot",
        )
        print(f"Recording to {video_dir}/")
    else:
        env = SharedAutonomyEnv(render_mode="human")

    pilot = FullPilot()
    rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        pilot.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = pilot(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
        print(f"Episode {ep + 1:>3}: {total_reward:8.2f}")

    print(f"\nMean over {n_episodes} episodes: {np.mean(rewards):.2f}")
    env.close()

    if record:
        merge_recordings(video_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--record", action="store_true", help="save MP4s and merge into one file")
    parser.add_argument("--video-dir", default="recordings", help="folder for saved videos (default: recordings/)")
    args = parser.parse_args()
    watch(args.episodes, args.record, args.video_dir)
