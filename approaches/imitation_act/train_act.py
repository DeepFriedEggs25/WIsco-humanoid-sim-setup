"""Train ACT (Action Chunking with Transformers) on the dataset produced by
format_dataset.py, via LeRobot's own `lerobot-train` CLI.

This is a thin wrapper, not a reimplementation -- it just fills in this
sandbox's sensible defaults (local dataset root, CPU device, small step
count for a first run) and shells out to the real `lerobot-train` entry
point, so you always get LeRobot's actual, current training loop, not a
copy of it that can drift out of sync.

VERIFIED against lerobot==0.6.1: this exact CLI invocation shape was run
during development (5-500 steps) against a dataset produced by
format_dataset.py, with `--policy.type=act` -- see RESEARCH_NOTES.md
"Verification log" for output and timing (52M params, ~5-6 steps/sec on an
Apple M-series CPU). Separately confirmed with a real 5000-step run on
Apple Silicon GPU (`--device mps`): ~17 steps/sec, loss 5.679 -> 0.223 over
the run (RESEARCH_NOTES.md section 3a). A FULL run to convergence (the
LeRobot team's own guidance: ~100k steps for real tasks) has not been run
-- that takes hours even on a GPU and is the natural next thing to try once
this pipeline is confirmed working, which it is.

Usage (local dataset, the default):
    python approaches/imitation_act/train_act.py \\
        --dataset_root outputs/datasets/fetch_pick_place \\
        --repo_id local/fetch_pick_place \\
        --steps 5000

Usage (dataset already pushed to the Hub -- downloads it instead of using
a local copy; requires `hf auth login` first if the dataset is private):
    python approaches/imitation_act/train_act.py \\
        --repo_id <your-hf-username>/fetch_pick_place --from_hub \\
        --steps 5000

Requires: pip install "lerobot[training]"  (see requirements.txt)
"""
import argparse
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root
from utils.repo_paths import OUTPUTS_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_root", default=None,
                        help="local dataset path (default: outputs/datasets/fetch_pick_place). "
                             "Ignored if --from_hub is set.")
    parser.add_argument("--repo_id", default="local/fetch_pick_place",
                        help="with --from_hub, this MUST be the real "
                             "'<your-hf-username>/dataset_name' you pushed to")
    parser.add_argument("--from_hub", action="store_true",
                        help="download --repo_id from the Hugging Face Hub instead of reading a "
                             "local --dataset_root (needs internet; needs `hf auth login` first if "
                             "the dataset is private)")
    parser.add_argument("--output_dir", default=None, help="default: outputs/train/act")
    parser.add_argument("--steps", type=int, default=5000,
                        help="LeRobot's own guidance: ~100k for a real task, a few thousand is enough "
                             "to sanity-check the pipeline is learning something on a tiny demo set")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"],
                        help="'mps' uses Apple Silicon GPU -- confirmed working directly, ~17 "
                             "steps/sec on a real 5000-step run vs ~5-6/sec on CPU, no extra flags "
                             "needed. See RESEARCH_NOTES.md section 3a.")
    parser.add_argument("--save_freq", type=int, default=1000)
    parser.add_argument("--wandb", action="store_true", help="enable Weights & Biases logging")
    args = parser.parse_args()

    output_dir = args.output_dir or str(OUTPUTS_DIR / "train" / "act")

    cmd = ["lerobot-train", f"--dataset.repo_id={args.repo_id}"]
    if args.from_hub:
        # No --dataset.root: LeRobot resolves purely from repo_id, downloading
        # into its own HF cache (~/.cache/huggingface/lerobot/) if not already there.
        print(f"Downloading dataset from https://huggingface.co/datasets/{args.repo_id} ...")
    else:
        dataset_root = args.dataset_root or str(OUTPUTS_DIR / "datasets" / "fetch_pick_place")
        cmd.append(f"--dataset.root={dataset_root}")

    cmd += [
        "--policy.type=act",
        f"--policy.device={args.device}",
        "--policy.push_to_hub=false",
        f"--output_dir={output_dir}",
        "--job_name=act_fetch_pick_place",
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
