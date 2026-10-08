"""Convert the raw .npz episodes from collect_demos.py into a real
LeRobotDataset (v3 format), ready for `lerobot-train --policy.type=act` (or
--policy.type=diffusion, see ../imitation_diffusion).

VERIFIED against lerobot==0.6.1: LeRobotDataset.create() / add_frame() /
save_episode() / finalize() were all exercised end-to-end during
development against a dataset shaped exactly like the one this script
produces, and the resulting dataset was then successfully consumed by
`lerobot-train --policy.type=act` AND `--policy.type=diffusion` for a few
steps on CPU. See RESEARCH_NOTES.md "Verification log" for the exact
commands and output. The 25/4/96x96x3 feature shapes below match
envs/arm_pick_place.py and collect_demos.py's IMAGE_SIZE exactly -- if you
change either, update this file's `features` dict too.

Usage:
    python approaches/imitation_act/format_dataset.py \\
        --demos_dir outputs/demos/raw \\
        --repo_id local/fetch_pick_place \\
        --root outputs/datasets/fetch_pick_place

`--root` keeps everything on your local disk -- no Hugging Face account or
internet access needed for this step. Use `--push_to_hub` if you do want to
share the dataset (requires `hf auth login` first -- `huggingface-cli` is
deprecated in current huggingface_hub versions and no longer works).
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root

import numpy as np

from utils.recording import load_episode_npz
from utils.repo_paths import OUTPUTS_DIR

TASK_DESCRIPTION = "pick up the block and move it to the target position"
FPS = 10  # matches the ~10Hz scripted-expert control rate used to collect the demos


def build_features(state_dim: int, action_dim: int, image_size: int) -> dict:
    return {
        "observation.state": {"dtype": "float32", "shape": (state_dim,),
                               "names": [f"s{i}" for i in range(state_dim)]},
        "action": {"dtype": "float32", "shape": (action_dim,),
                   "names": [f"a{i}" for i in range(action_dim)]},
        "observation.images.top": {"dtype": "video", "shape": (image_size, image_size, 3),
                                    "names": ["height", "width", "channel"]},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--demos_dir", default=None, help="default: outputs/demos/raw")
    parser.add_argument("--repo_id", default="local/fetch_pick_place",
                        help="dataset identifier, 'namespace/name' -- doesn't need to be a real HF user "
                             "unless you pass --push_to_hub")
    parser.add_argument("--root", default=None, help="default: outputs/datasets/<repo_id's name>")
    parser.add_argument("--push_to_hub", action="store_true",
                        help="also push to the Hugging Face Hub (requires `hf auth login` first)")
    args = parser.parse_args()

    # Imported here, not at module top, so `--help` and collect_demos.py's
    # workflow never require lerobot/torch to be installed -- only this
    # conversion step does.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    demos_dir = pathlib.Path(args.demos_dir) if args.demos_dir else (OUTPUTS_DIR / "demos" / "raw")
    episode_paths = sorted(demos_dir.glob("episode_*.npz"))
    if not episode_paths:
        raise FileNotFoundError(f"no episode_*.npz files found in {demos_dir} -- run collect_demos.py first")

    first = load_episode_npz(episode_paths[0])
    state_dim = first["observations"].shape[-1]
    action_dim = first["actions"].shape[-1]
    image_size = first["images"].shape[1]

    root = pathlib.Path(args.root) if args.root else (OUTPUTS_DIR / "datasets" / args.repo_id.split("/")[-1])
    if root.exists():
        raise FileExistsError(f"{root} already exists -- LeRobotDataset.create() refuses to overwrite. "
                               "Remove it or pass a different --root.")

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=FPS,
        features=build_features(state_dim, action_dim, image_size),
        root=root,
        use_videos=True,
    )

    for path in episode_paths:
        ep = load_episode_npz(path)
        for t in range(len(ep["actions"])):
            dataset.add_frame({
                "observation.state": ep["observations"][t].astype(np.float32),
                "action": ep["actions"][t].astype(np.float32),
                "observation.images.top": ep["images"][t],
                "task": TASK_DESCRIPTION,
            })
        dataset.save_episode()
        print(f"  {path.name}: {len(ep['actions'])} frames added")

    dataset.finalize()
    print(f"\nDataset created at {root}")
    print(f"  episodes: {dataset.num_episodes}, frames: {dataset.num_frames}")

    if args.push_to_hub:
        dataset.push_to_hub()
        print(f"pushed to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
