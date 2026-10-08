"""Collect demonstration episodes for FetchPickAndPlace-v4 using the
scripted expert in scripted_expert.py -- no teleop hardware, no human
operator required. Saves one .npz per successful episode (via
utils/recording.save_episode_npz), which format_dataset.py then converts
into a real LeRobotDataset.

Two-step design (collect -> format) deliberately, not one script: this step
has NO dependency on lerobot/torch at all, only mujoco/gymnasium/numpy/
Pillow -- so a member can generate demonstrations even before installing the
(much heavier) training stack, and can re-run format_dataset.py repeatedly
against the same raw demos without re-collecting.

VERIFIED: this script was run end-to-end during development and produced
usable episodes fed successfully into format_dataset.py -> lerobot-train
(see RESEARCH_NOTES.md "Verification log").

Usage:
    python approaches/imitation_act/collect_demos.py --num_episodes 20
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root

import numpy as np
from PIL import Image

from approaches.imitation_act.scripted_expert import scripted_expert_action
from envs.arm_pick_place import make_env
from utils.recording import save_episode_npz
from utils.repo_paths import OUTPUTS_DIR

IMAGE_SIZE = 96  # downsampled from the env's native 480x480 -- plenty for a
                 # small demo task and much lighter for CPU training/storage.
MAX_STEPS_PER_EPISODE = 100


def resize(frame: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    return np.array(Image.fromarray(frame).resize((size, size)))


def collect_one_episode(env, seed: int):
    obs, info = env.reset(seed=seed)
    observations, actions, rewards, images = [], [], [], []
    success = False
    for _ in range(MAX_STEPS_PER_EPISODE):
        frame = resize(env.render())
        action = scripted_expert_action(obs)

        observations.append(obs["observation"].copy())
        actions.append(action.copy())
        images.append(frame)

        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        if info.get("is_success"):
            success = True
        if terminated or truncated:
            break
    return observations, actions, rewards, images, success


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num_episodes", type=int, default=20,
                        help="number of SUCCESSFUL episodes to collect (failed ones are retried with a new seed)")
    parser.add_argument("--out_dir", default=None, help="default: outputs/demos/raw")
    parser.add_argument("--seed_start", type=int, default=0)
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir) if args.out_dir else (OUTPUTS_DIR / "demos" / "raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(render_mode="rgb_array")
    collected = 0
    seed = args.seed_start
    attempts = 0
    max_attempts = args.num_episodes * 3  # scripted expert succeeds ~100% (see README), so this is generous

    while collected < args.num_episodes and attempts < max_attempts:
        observations, actions, rewards, images, success = collect_one_episode(env, seed)
        attempts += 1
        seed += 1
        if not success:
            print(f"  seed {seed - 1}: episode failed, skipping")
            continue

        path = out_dir / f"episode_{collected:04d}.npz"
        save_episode_npz(path, observations, actions, rewards, images=images, success=True)
        print(f"  seed {seed - 1}: SUCCESS, {len(actions)} steps -> {path}")
        collected += 1

    env.close()
    print(f"\nCollected {collected}/{args.num_episodes} episodes in {out_dir} "
          f"({attempts} attempts, {100 * collected / attempts:.0f}% success rate)")
    if collected < args.num_episodes:
        print("WARNING: fewer episodes collected than requested -- check the scripted expert "
              "is still succeeding (see scripted_expert.py's own smoke test).")


if __name__ == "__main__":
    main()
