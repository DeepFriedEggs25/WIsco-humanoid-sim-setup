"""Helpers for saving demo trajectories and videos to disk. Deliberately
dependency-light: plain numpy .npz for trajectories (converted to a real
LeRobot dataset separately by approaches/imitation_act/format_dataset.py --
kept as two steps so the "collect" step never depends on lerobot/torch being
installed, only mujoco/gymnasium).
"""
import pathlib

import imageio.v2 as imageio
import numpy as np


class VideoWriter:
    """Accumulates rgb frames (H,W,3 uint8) and writes an mp4 on close().
    VERIFIED working during development (see RESEARCH_NOTES.md)."""

    def __init__(self, path, fps: int = 30):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.frames = []
        self.fps = fps

    def add(self, frame: np.ndarray):
        self.frames.append(frame)

    def close(self):
        if not self.frames:
            return
        # codec must be given explicitly -- a real bug found during
        # development: imageio's pyav backend does not reliably infer a
        # codec from the ".mp4" extension alone in imageio==2.37 and fails
        # with a confusing "expected bytes, NoneType found" deep in PyAV
        # instead of a clear error. libx264 is the widely-compatible choice.
        imageio.mimsave(self.path, self.frames, fps=self.fps, codec="libx264")


def save_episode_npz(path, observations: list, actions: list, rewards: list,
                      images: list | None = None, success: bool = False):
    """Saves one collected episode as a single .npz file. `observations` and
    `actions` are lists of per-step numpy arrays (ragged across episodes is
    fine, each episode is its own file); `images`, if given, is a list of
    (H,W,3) uint8 frames, one per step.

    format_dataset.py reads these back in to build a real LeRobotDataset --
    see approaches/imitation_act/README.md for the two-step reasoning.
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "success": np.array(success),
    }
    if images is not None:
        payload["images"] = np.asarray(images, dtype=np.uint8)
    np.savez_compressed(path, **payload)


def load_episode_npz(path) -> dict:
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}
