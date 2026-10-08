"""Interactive launcher, fully driven from inside the MuJoCo render window:
pick a trained checkpoint, type exact block/target coordinates, and watch
the policy attempt pick-and-place live -- retrying up to X times and
stopping early on the first success. All input happens as on-screen
prompts + keystrokes captured by the same window, not the terminal.

The window is an OpenCV window (`cv2.imshow`) fed by MuJoCo's offscreen
renderer -- the same `env.render(render_mode="rgb_array")` path already
verified throughout this repo (collect_demos.py, evaluate.py,
pick_place_scripted.py's --render video). Deliberately NOT MuJoCo's own
interactive viewer (`mujoco.viewer.launch_passive`, what --render viewer
uses elsewhere in this repo) -- that needs `mjpython` on macOS
(RESEARCH_NOTES.md section 3a) and hands back no rgb array, but the policy
needs exactly that array as its vision input every step anyway. So the
window you watch is literally what the policy sees, fed through one path,
under plain `python` on every platform.

On-screen controls, always shown at the bottom of the window:
  type digits/'.'/'-' to build a number, Enter to confirm, Backspace to
  edit, Esc to accept the shown default, q to quit at any prompt.

How coordinate overriding works: FetchPickAndPlace-v4's reset() randomizes
the block and goal internally and exposes no public API to set them
directly. This pokes the same internals gymnasium_robotics itself uses --
confirmed by reading the installed gymnasium_robotics==1.4.2 source
(envs/fetch/fetch_env.py's _reset_sim/_sample_goal/_render_callback) and by
directly testing it: set a position, re-read the observation, confirmed it
landed exactly where requested.
  - block position: env.unwrapped._utils.set_joint_qpos(model, data,
    "object0:joint", qpos) with qpos[:2] overridden (z/orientation left as
    the model's own resting pose, exactly like _reset_sim does)
  - goal/target: env.unwrapped.goal = np.array([x, y, z]), then
    env.unwrapped._render_callback() to move the visualized green marker
Valid ranges are read from the env's own obj_range/target_range/
height_offset/initial_gripper_xpos attributes, not hard-coded.

Usage:
    python play.py

Requires a trained checkpoint already sitting under outputs/train/ (see
approaches/imitation_act/README.md) and `pip install "lerobot[training]"`.
"""
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # repo root

import cv2
import mujoco
import numpy as np
import torch
from PIL import Image

from approaches.imitation_act.scripted_expert import scripted_expert_action
from envs.arm_pick_place import make_env
from utils.repo_paths import OUTPUTS_DIR

POLICY_IMAGE_SIZE = 96  # must match format_dataset.py's IMAGE_SIZE for whatever checkpoint is loaded
MAX_STEPS_PER_ATTEMPT = 100
WINDOW_NAME = "manipulation_sim -- interactive pick-and-place"

# "Nudged ACT" -- test-time fine-tuning against a one-shot oracle demo of the
# exact scene being tested, chained across attempts (each attempt's nudge
# steps carry into the next). LR matches the real training run's own LR
# (RESEARCH_NOTES.md); step count and effect were verified directly before
# wiring this in: 10 steps on a real 16-frame oracle demo dropped ACT's own
# training loss from 0.400 to 0.146 (l1_loss 0.235 -> 0.064).
NUDGE_LR = 1e-5
NUDGE_STEPS_PER_ATTEMPT = 10
ORACLE_MAX_STEPS = 100
ORACLE_MAX_RETRIES = 2  # scripted_expert_action is verified 20/20 elsewhere; retry a couple times just in case


class QuitRequested(Exception):
    """Raised when the user presses 'q' at any on-screen prompt."""


# --------------------------------------------------------------------------
# Checkpoint discovery / policy loading
# --------------------------------------------------------------------------

def discover_checkpoints() -> list[tuple[str, pathlib.Path]]:
    """Finds one checkpoint per outputs/train/<run>/ directory: the most-
    trained one available (LeRobot's "last" if present, else the highest
    numbered step). Note this is "most trained", not "best measured" --
    this repo doesn't log a per-checkpoint success rate anywhere, so "best"
    is a proxy, not a verified ranking."""
    train_dir = OUTPUTS_DIR / "train"
    found = []
    if not train_dir.exists():
        return found
    for run_dir in sorted(p for p in train_dir.iterdir() if p.is_dir()):
        ckpt_root = run_dir / "checkpoints"
        if not ckpt_root.exists():
            continue
        candidates = [p for p in ckpt_root.iterdir()
                      if p.is_dir() and (p / "pretrained_model" / "config.json").exists()]
        if not candidates:
            continue
        by_name = {p.name: p for p in candidates}
        if "last" in by_name:
            best = by_name["last"]
        else:
            numeric = [p for p in candidates if p.name.isdigit()]
            best = max(numeric, key=lambda p: int(p.name)) if numeric else sorted(candidates)[-1]
        found.append((run_dir.name, best / "pretrained_model"))
    return found


def load_policy(checkpoint_dir: pathlib.Path):
    # Imported here (not at module top) so this script can at least start
    # and list checkpoints before lerobot/torch are confirmed installed --
    # matches format_dataset.py/train_act.py/evaluate.py's own pattern.
    from lerobot.policies.factory import get_policy_class
    from lerobot.policies import make_pre_post_processors
    from lerobot.configs.policies import PreTrainedConfig

    config = PreTrainedConfig.from_pretrained(str(checkpoint_dir))
    policy_cls = get_policy_class(config.type)
    policy = policy_cls.from_pretrained(str(checkpoint_dir))
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(config, pretrained_path=str(checkpoint_dir))
    return policy, preprocessor, postprocessor


# --------------------------------------------------------------------------
# Scene control (block position, goal position)
# --------------------------------------------------------------------------

def get_bounds(env) -> dict:
    """Reads the exact randomization parameters gymnasium_robotics itself
    uses for this task, so bounds shown to the user match the environment's
    real valid domain instead of a guessed approximation. No minimum-
    distance-from-center exclusion is enforced here: that rule exists in
    the env's own random sampler (to avoid spawning ON TOP of the gripper
    on a random reset), but since this tool places the block at an exact
    user-chosen spot rather than sampling, it isn't needed -- dropped after
    it caused confusing rejections of otherwise-valid coordinates."""
    u = env.unwrapped
    return {
        "center_xy": u.initial_gripper_xpos[:2].copy(),
        "obj_range": float(u.obj_range),
        "target_range": float(u.target_range),
        "target_offset": float(np.asarray(u.target_offset).flat[0]),
        "height_offset": float(u.height_offset),
        "distance_threshold": float(u.distance_threshold),
    }


def set_block_xy(env, object_xy: np.ndarray):
    """Moves just the block, for immediate visual feedback while typing."""
    u = env.unwrapped
    model, data = u.model, u.data
    object_qpos = u._utils.get_joint_qpos(model, data, "object0:joint")
    object_qpos[0:2] = object_xy
    u._utils.set_joint_qpos(model, data, "object0:joint", object_qpos)
    mujoco.mj_forward(model, data)


def set_goal_xyz(env, goal_xyz: np.ndarray):
    """Moves just the green target marker, for immediate visual feedback."""
    u = env.unwrapped
    u.goal = np.asarray(goal_xyz, dtype=np.float64)
    u._render_callback()


def set_scene(env, object_xy: np.ndarray, goal_xyz: np.ndarray):
    """Places both the block and the goal marker -- used at the start of
    each attempt, after env.reset() has re-randomized both away from the
    user's chosen spot."""
    set_block_xy(env, object_xy)
    set_goal_xyz(env, goal_xyz)


# --------------------------------------------------------------------------
# On-screen drawing + input (everything happens in the cv2 window)
# --------------------------------------------------------------------------

FONT_SCALE = 0.38
LINE_HEIGHT = 15


def draw_overlay(frame_rgb: np.ndarray, lines: list[str]) -> np.ndarray:
    img = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR).copy()
    y = 14
    for line in lines:
        cv2.putText(img, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (80, 255, 80), 1, cv2.LINE_AA)
        y += LINE_HEIGHT
    return img


def show(env, lines: list[str]):
    frame = env.render()
    cv2.imshow(WINDOW_NAME, draw_overlay(frame, lines))
    cv2.waitKey(1)


def prompt_number(env, label: str, lo: float, hi: float, default: float,
                   is_int: bool = False, extra: list[str] | None = None) -> float:
    """Blocks until the user types a valid number (in-window) and presses
    Enter, or presses Esc to accept `default`, or q to quit the program."""
    buf = ""
    error = ""
    rng = f"{int(lo)}-{int(hi)}" if is_int else f"{lo:.2f} to {hi:.2f}"
    while True:
        frame = env.render()
        lines = (extra or []) + [
            f"{label} ({rng}):",
            f"> {buf}_" + (f"   [{error}]" if error else ""),
            "Enter=ok  Esc=default  q=quit",
        ]
        cv2.imshow(WINDOW_NAME, draw_overlay(frame, lines))
        key = cv2.waitKey(30) & 0xFF
        if key in (255, 0):
            continue
        if key == ord("q"):
            raise QuitRequested()
        if key == 27:  # Esc
            return default
        if key in (13, 10):  # Enter
            raw = buf if buf else str(default)
            try:
                val = int(raw) if is_int else float(raw)
            except ValueError:
                error, buf = "not a number", ""
                continue
            if not (lo <= val <= hi):
                error, buf = "out of range", ""
                continue
            return val
        if key in (8, 127):  # Backspace
            buf, error = buf[:-1], ""
            continue
        ch = chr(key) if key < 128 else ""
        if ch in "0123456789.-":
            buf += ch
            error = ""


def prompt_choice(env, options: list[str], default: int) -> int:
    extra = ["Pick a checkpoint:"] + [f"  {i}) {name}" for i, name in enumerate(options)]
    return int(prompt_number(env, "choice", 0, len(options) - 1, default, is_int=True, extra=extra))


def wait_for_key(env, lines: list[str], valid_keys: str) -> str:
    """Blocks until one of `valid_keys` (a string, e.g. 'ry mq') is pressed;
    always honors 'q' as quit even if not listed."""
    while True:
        frame = env.render()
        cv2.imshow(WINDOW_NAME, draw_overlay(frame, lines))
        key = cv2.waitKey(30) & 0xFF
        if key in (255, 0):
            continue
        if key == ord("q"):
            raise QuitRequested()
        ch = chr(key) if key < 128 else ""
        if ch in valid_keys:
            return ch


def prompt_object_xy(env, bounds: dict, current_xy: np.ndarray) -> np.ndarray:
    cx, cy = bounds["center_xy"]
    r = bounds["obj_range"]
    x = prompt_number(env, "Block X", cx - r, cx + r, default=cx + 0.10)
    set_block_xy(env, np.array([x, current_xy[1]]))
    y = prompt_number(env, "Block Y", cy - r, cy + r, default=cy)
    xy = np.array([x, y])
    set_block_xy(env, xy)
    return xy


def prompt_goal_xyz(env, bounds: dict) -> np.ndarray:
    cx, cy = bounds["center_xy"]
    r, off = bounds["target_range"], bounds["target_offset"]
    z_table = bounds["height_offset"]
    x = prompt_number(env, "Target X", cx + off - r, cx + off + r, default=cx + off)
    y = prompt_number(env, "Target Y", cy + off - r, cy + off + r, default=cy + off)
    set_goal_xyz(env, np.array([x, y, z_table]))
    z = prompt_number(env, "Target Z", z_table, z_table + 0.45, default=z_table)
    xyz = np.array([x, y, z])
    set_goal_xyz(env, xyz)
    return xyz


# --------------------------------------------------------------------------
# Rollout
# --------------------------------------------------------------------------

def build_observation(obs: dict, frame: np.ndarray):
    img = np.array(Image.fromarray(frame).resize((POLICY_IMAGE_SIZE, POLICY_IMAGE_SIZE)))
    return {
        "observation.state": torch.from_numpy(obs["observation"].astype(np.float32)).unsqueeze(0),
        "observation.images.top": torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0,
    }


def run_attempt(env, policy, preprocessor, postprocessor, object_xy, goal_xyz,
                 seed: int, attempt: int, max_attempts: int, nudge_enabled: bool = False) -> bool:
    obs, info = env.reset(seed=seed)
    set_scene(env, object_xy, goal_xyz)
    obs = env.unwrapped._get_obs()  # re-read now that the scene has actually moved
    policy.reset()

    tag = "NUDGED ACT" if nudge_enabled else "ACT"
    success = False
    for step in range(MAX_STEPS_PER_ATTEMPT):
        frame = env.render()
        cv2.imshow(WINDOW_NAME, draw_overlay(frame, [
            f"[{tag}] attempt {attempt}/{max_attempts}  q=skip",
        ]))
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        batch = preprocessor(build_observation(obs, frame))
        with torch.no_grad():
            action = policy.select_action(batch)
        action = postprocessor(action).squeeze(0).numpy()

        obs, reward, terminated, truncated, info = env.step(action)
        time.sleep(0.03)  # roughly watchable real-time-ish pacing
        if info.get("is_success"):
            success = True
        if terminated or truncated:
            break
    return success


# --------------------------------------------------------------------------
# Nudged ACT: oracle-guided test-time fine-tuning, chained across attempts
# --------------------------------------------------------------------------

def record_oracle_demo(env, object_xy: np.ndarray, goal_xyz: np.ndarray):
    """Plays the scripted (non-learned) expert -- the same one that
    generated this repo's real training data, verified 20/20 in
    RESEARCH_NOTES.md -- live in the window, and records its
    (state, image, action) trajectory. Returns (states, images, actions, success)."""
    obs, info = env.reset(seed=0)
    set_scene(env, object_xy, goal_xyz)
    obs = env.unwrapped._get_obs()

    states, images, actions = [], [], []
    success = False
    for step in range(ORACLE_MAX_STEPS):
        frame = env.render()
        cv2.imshow(WINDOW_NAME, draw_overlay(frame, [f"Playing expert demo -- step {step}"]))
        cv2.waitKey(1)

        img = np.array(Image.fromarray(frame).resize((POLICY_IMAGE_SIZE, POLICY_IMAGE_SIZE)))
        action = scripted_expert_action(obs)
        states.append(obs["observation"].astype(np.float32))
        images.append(img)
        actions.append(action)

        obs, reward, terminated, truncated, info = env.step(action)
        time.sleep(0.03)
        if info.get("is_success"):
            success = True
        if terminated or truncated:
            break
    return states, images, actions, success


def build_nudge_batch(states: list, images: list, actions: list, chunk_size: int) -> dict:
    """Builds a raw (unnormalized) ACT training batch from one recorded
    trajectory: one sample per timestep t, action target = the chunk_size
    actions starting at t (padded with the last action, flagged in
    action_is_pad, past the end of the trajectory) -- the same chunking
    scheme LeRobot's own dataset uses for training. Verified directly
    against the real ACTPolicy.forward() before this was wired in."""
    T = len(actions)
    action_dim = actions[0].shape[0]
    action_chunks = np.zeros((T, chunk_size, action_dim), dtype=np.float32)
    is_pad = np.zeros((T, chunk_size), dtype=bool)
    for t in range(T):
        for i in range(chunk_size):
            idx = t + i
            if idx < T:
                action_chunks[t, i] = actions[idx]
            else:
                action_chunks[t, i] = actions[-1]
                is_pad[t, i] = True
    return {
        "observation.state": torch.from_numpy(np.stack(states)),
        "observation.images.top": torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float() / 255.0,
        "action": torch.from_numpy(action_chunks),
        "action_is_pad": torch.from_numpy(is_pad),
    }


def nudge_once(policy, optimizer, norm_batch: dict):
    """One link in the nudge chain: NUDGE_STEPS_PER_ATTEMPT gradient steps
    against the (already normalized, fixed) oracle demo batch. ACT's vision
    backbone uses FrozenBatchNorm2d (see modeling_act.py's own import), so
    switching to train() for this doesn't destabilize any running BatchNorm
    stats on such a small batch."""
    policy.train()
    for _ in range(NUDGE_STEPS_PER_ATTEMPT):
        optimizer.zero_grad()
        loss, _ = policy.forward(norm_batch)
        loss.backward()
        optimizer.step()
    policy.eval()


# --------------------------------------------------------------------------
# Main state machine -- everything below drives the same open window
# --------------------------------------------------------------------------

def main():
    env = make_env(render_mode="rgb_array")
    env.reset(seed=0)
    bounds = get_bounds(env)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    try:
        checkpoints = discover_checkpoints()
        if not checkpoints:
            print(f"No checkpoints found under {OUTPUTS_DIR / 'train'}/*/checkpoints/*/pretrained_model.")
            print("Train one first, e.g.: python approaches/imitation_act/train_act.py ...")
            return

        policy = preprocessor = postprocessor = checkpoint_path = None
        nudge_enabled = False
        attempt_seed = 0
        object_xy = env.unwrapped._utils.get_joint_qpos(
            env.unwrapped.model, env.unwrapped.data, "object0:joint")[:2].copy()

        while True:
            if policy is None:
                idx = prompt_choice(env, [name for name, _ in checkpoints], default=len(checkpoints) - 1)
                checkpoint_path = checkpoints[idx][1]
                show(env, [f"Loading {checkpoints[idx][0]} ..."])
                policy, preprocessor, postprocessor = load_policy(checkpoint_path)
                nudge_enabled = wait_for_key(env, [
                    "Use nudged ACT? (oracle-guided live fine-tuning, chained across attempts)",
                    "y=yes  n=no",
                ], "yn") == "y"

            # Reset arm to its home pose (and block to the env's default spot,
            # about to be overridden below anyway) -- otherwise the previous
            # attempt's end-of-episode pose (arm holding the block up near the
            # goal) just sits there while you type the next coordinates.
            env.reset(seed=attempt_seed)

            object_xy = prompt_object_xy(env, bounds, object_xy)
            goal_xyz = prompt_goal_xyz(env, bounds)

            optimizer = norm_batch = None
            if nudge_enabled:
                # Fresh weights for this new (block, goal): a nudge chain is
                # scoped to one fixed scenario, not carried over from a
                # different spot tested earlier -- reload from the checkpoint
                # on disk (untouched by any previous nudging) to guarantee that.
                show(env, ["Loading fresh weights for nudging..."])
                policy, preprocessor, postprocessor = load_policy(checkpoint_path)

                demo_ok = False
                for _ in range(1 + ORACLE_MAX_RETRIES):
                    states, images, actions, demo_ok = record_oracle_demo(env, object_xy, goal_xyz)
                    if demo_ok:
                        break
                wait_for_key(env, [
                    "Expert demo: SUCCESS" if demo_ok else "Expert demo FAILED after retries -- continuing anyway",
                    "Enter=continue",
                ], "\r\n")

                raw_batch = build_nudge_batch(states, images, actions, policy.config.chunk_size)
                norm_batch = preprocessor(raw_batch)
                optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=NUDGE_LR)

            max_attempts = int(prompt_number(env, "Max attempts", 1, 50, default=5, is_int=True))

            wait_for_key(env, ["Ready -- Enter=play  q=quit"], "\r\n")

            succeeded = False
            for attempt in range(1, max_attempts + 1):
                if nudge_enabled:
                    show(env, [f"[NUDGED ACT] nudge step before attempt {attempt}/{max_attempts} ..."])
                    nudge_once(policy, optimizer, norm_batch)
                succeeded = run_attempt(env, policy, preprocessor, postprocessor,
                                         object_xy, goal_xyz, seed=attempt_seed,
                                         attempt=attempt, max_attempts=max_attempts,
                                         nudge_enabled=nudge_enabled)
                attempt_seed += 1
                if succeeded:
                    break

            result = f"SUCCESS (attempt {attempt}/{max_attempts})" if succeeded else f"FAILED ({max_attempts}/{max_attempts})"
            choice = wait_for_key(env, [result, "r=retry  m=new model  q=quit"], "rm")
            if choice == "m":
                policy = preprocessor = postprocessor = checkpoint_path = None
    except QuitRequested:
        pass
    finally:
        env.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
