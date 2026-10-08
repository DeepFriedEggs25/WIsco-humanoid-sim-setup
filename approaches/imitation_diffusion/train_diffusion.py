"""Train Diffusion Policy on the SAME dataset format_dataset.py produces for
ACT -- no new data collection needed, just a different --policy.type. Thin
wrapper around `lerobot-train`, identical in structure to
../imitation_act/train_act.py (see that file's docstring for the general
rationale of wrapping rather than reimplementing).

VERIFIED against lerobot==0.6.1: this exact invocation shape (with
--policy.type=diffusion) was run during development -- see
RESEARCH_NOTES.md "Verification log". Two real gotchas were found and are
handled/documented here:

1. Requires the diffusion extra: `pip install "lerobot[diffusion]"`
   (installs the `diffusers` package). Without it you get a clear
   ImportError naming the fix, not a cryptic failure.
2. Diffusion Policy's default config needs longer episodes than ACT does --
   it failed outright ("No valid frames remain after applying
   drop_n_first_frames and drop_n_last_frames") on 5-frame test episodes but
   worked fine on 30+. format_dataset.py's demos (50 steps/episode from
   collect_demos.py) are long enough; if you shorten MAX_STEPS_PER_EPISODE
   there, re-check this.

Diffusion Policy is also much heavier than ACT: 263M params vs. ACT's 52M in
this same setup, and roughly 5x slower per step on CPU in development
testing (~1 step/sec vs. ~5-6 steps/sec). Expect training to take
noticeably longer for the same --steps.

Usage:
    python approaches/imitation_diffusion/train_diffusion.py \\
        --dataset_root outputs/datasets/fetch_pick_place \\
        --repo_id local/fetch_pick_place \\
        --steps 5000
"""
import argparse
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root
from utils.repo_paths import OUTPUTS_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_root", default=None, help="default: outputs/datasets/fetch_pick_place")
    parser.add_argument("--repo_id", default="local/fetch_pick_place")
    parser.add_argument("--output_dir", default=None, help="default: outputs/train/diffusion")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--save_freq", type=int, default=1000)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    dataset_root = args.dataset_root or str(OUTPUTS_DIR / "datasets" / "fetch_pick_place")
    output_dir = args.output_dir or str(OUTPUTS_DIR / "train" / "diffusion")

    cmd = [
        "lerobot-train",
        f"--dataset.repo_id={args.repo_id}",
        f"--dataset.root={dataset_root}",
        "--policy.type=diffusion",
        f"--policy.device={args.device}",
        "--policy.push_to_hub=false",
        f"--output_dir={output_dir}",
        "--job_name=diffusion_fetch_pick_place",
        f"--steps={args.steps}",
        f"--batch_size={args.batch_size}",
        "--save_checkpoint=true",
        f"--save_freq={args.save_freq}",
        f"--wandb.enable={'true' if args.wandb else 'false'}",
    ]
    print("Running:", " ".join(cmd))
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
