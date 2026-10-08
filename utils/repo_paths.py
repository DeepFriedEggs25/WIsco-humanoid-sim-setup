"""Shared path constants. Every runnable script in this repo starts with the
two-line bootstrap below (before importing anything from `utils`, since that
import itself needs the repo root on sys.path first):

    import pathlib, sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

...where `parents[N]` is however many directories up from the script to the
repo root (2 for a script in approaches/<name>/, 1 for a script directly in
envs/ or utils/). This lets every script be run directly
(`python approaches/scripted_ik/pick_place_scripted.py`) with no `pip
install -e .` / PYTHONPATH setup required -- deliberate, for a
learning-sandbox repo new members clone and run immediately.
"""
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
OUTPUTS_DIR = REPO_ROOT / "outputs"  # videos, saved demos, checkpoints -- gitignored
