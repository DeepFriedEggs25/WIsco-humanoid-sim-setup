"""Load a trained ACT (or Diffusion -- this loader is policy-agnostic)
checkpoint and run it in the live FetchPickAndPlace-v4 environment, measuring
success rate over N episodes and saving a video of one episode.

VERIFIED against lerobot==0.6.1: the load-checkpoint -> build pre/post-
processors -> select_action loop below was run during development against a
real (undertrained, 5-step) checkpoint and produced a well-formed 4-dim
action -- see RESEARCH_NOTES.md "Verification log" for the exact pattern
this follows (lerobot.policies.make_pre_post_processors,
ACTPolicy.from_pretrained). Running it against a FULLY TRAINED policy (i.e.
actually measuring a meaningful success rate) was NOT done in development --
training to convergence takes hours; this script's job is to have a correct,
ready-to-use evaluation loop waiting for whenever a member finishes a real
training run.

Usage:
    python approaches/imitation_act/evaluate.py \\
        --checkpoint outputs/train/act/checkpoints/last/pretrained_model \\
        --episodes 10 --video outputs/eval_act.mp4
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root

import numpy as np
import torch
from PIL import Image

from envs.arm_pick_place import make_env
from utils.recording import VideoWriter
from utils.repo_paths import OUTPUTS_DIR

IMAGE_SIZE = 96  # must match format_dataset.py's IMAGE_SIZE for the checkpoint being evaluated
MAX_STEPS_PER_EPISODE = 100


def load_policy(checkpoint_dir: str):
    # Imported here (not at module top) so --help works without lerobot/torch
    # installed, matching format_dataset.py/train_act.py's pattern.
    from lerobot.policies.factory import get_policy_class
    from lerobot.policies import make_pre_post_processors
    from lerobot.configs.policies import PreTrainedConfig

    config = PreTrainedConfig.from_pretrained(checkpoint_dir)
    policy_cls = get_policy_class(config.type)
    policy = policy_cls.from_pretrained(checkpoint_dir)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(config, pretrained_path=checkpoint_dir)
    return policy, preprocessor, postprocessor


def build_observation(obs: dict, frame: np.ndarray):
    img = np.array(Image.fromarray(frame).resize((IMAGE_SIZE, IMAGE_SIZE)))
    return {
        "observation.state": torch.from_numpy(obs["observation"].astype(np.float32)).unsqueeze(0),
        "observation.images.top": torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True,
                        help="path to a pretrained_model/ directory, e.g. "
                             "outputs/train/act/checkpoints/last/pretrained_model")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--video", default=None, help="if given, save a video of the FIRST episode here")
    parser.add_argument("--seed_start", type=int, default=1000,
                        help="offset from collect_demos.py's seeds so eval doesn't reuse training episodes")
    args = parser.parse_args()

    policy, preprocessor, postprocessor = load_policy(args.checkpoint)
    env = make_env(render_mode="rgb_array")

    successes = 0
    writer = None
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed_start + ep)
        policy.reset()
        record_this_episode = (ep == 0 and args.video is not None)
        if record_this_episode:
            out = pathlib.Path(args.video) if args.video else (OUTPUTS_DIR / "eval_act.mp4")
            writer = VideoWriter(out, fps=10)

        ep_success = False
        for _ in range(MAX_STEPS_PER_EPISODE):
            frame = env.render()
            if record_this_episode:
                writer.add(frame)
            batch = preprocessor(build_observation(obs, frame))
            with torch.no_grad():
                action = policy.select_action(batch)
            action = postprocessor(action).squeeze(0).numpy()

            obs, reward, terminated, truncated, info = env.step(action)
            if info.get("is_success"):
                ep_success = True
            if terminated or truncated:
                break

        successes += int(ep_success)
        print(f"episode {ep}: {'SUCCESS' if ep_success else 'failure'}")
        if record_this_episode:
            writer.close()
            print(f"  video saved to {writer.path}")

    env.close()
    rate = 100 * successes / args.episodes
    print(f"\nSuccess rate: {successes}/{args.episodes} ({rate:.0f}%)")


if __name__ == "__main__":
    main()
