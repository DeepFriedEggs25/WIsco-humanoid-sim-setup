# Research notes: pinned versions, APIs, and gotchas

Everything in this file was checked against a real, installed environment
during development (Python 3.12.14, macOS/arm64) — not taken on faith from
docs or blog posts, several of which turned out to be stale (see the
Gymnasium-Robotics section below). Where something could not be verified in
this environment, that's stated explicitly.

---

## 1. Pinned, mutually-compatible versions

| Package | Version | Why this version |
|---|---|---|
| Python | 3.12.x | Required by `lerobot>=0.6` (`>=3.12`). `gymnasium-robotics` alone would tolerate 3.10+, but starting on 3.12 avoids needing a second venv later. |
| `mujoco` | 3.2.7 | Verified: loads scenes, steps physics, offscreen renders, `mj_jacSite`/`mj_comPos` all confirmed working. |
| `gymnasium` | 1.3.0 | Satisfies both `gymnasium-robotics>=1.2.0` and `lerobot`'s `>=1.1.1,<2.0.0`. |
| `gymnasium-robotics` | 1.4.2 | Latest at time of writing; requires `gymnasium>=1.2.0`, `mujoco>=2.2.0`. |
| `lerobot` | 0.6.1 | Latest at time of writing; requires Python `>=3.12`, `torch>=2.7,<2.12`, `gymnasium>=1.1.1,<2.0.0`, `numpy>=2.0.0,<2.3.0`. |
| `torch` / `torchvision` | 2.11.0 / 0.26.0 | Pulled automatically by `lerobot[training]`; CPU wheels work fine for this repo's scale. |
| `stable-baselines3` | 2.9.0 | Latest at time of writing; works with `gymnasium<2.0`. |
| `numpy` | `>=2.0.0,<2.3.0` | **Upper bound is load-bearing**: `lerobot==0.6.1` pins `numpy<2.3.0`. Installing the base group alone would happily pull numpy 2.5.x; adding `lerobot` afterward silently downgrades it. Pin from the start. |

Real command used to discover these (not guessed): `pip index versions <pkg>`
against each package, then actually `pip install`ing the combination and
running real code against it (see "Verification log" below).

---

## 2. Gymnasium-Robotics: a stale-docs gotcha, and real environment details

**`FetchReach-v3` / `FetchPickAndPlace-v3` (and the other Fetch tasks'
`-v3`) are DEPRECATED as of `gymnasium-robotics>=1.3`.** Calling
`gym.make("FetchPickAndPlace-v3")` raises
`gymnasium.error.DeprecatedEnv: ... use FetchPickAndPlace-v4 instead`. This
was found the hard way: initial research (web search + a Farama docs page
snapshot) reported `-v3` as current, and the first attempt to actually run
it failed immediately. **Every environment ID used in this repo is `-v4`.**
If you find an outdated tutorial online using `-v3`, that's why it breaks.

Confirmed by directly inspecting a live `FetchPickAndPlace-v4` instance
(`env.unwrapped.model`, not docs):
- Observation: dict with `observation` (25,), `achieved_goal` (3,),
  `desired_goal` (3,). The 25-dim vector is `grip_pos(3), object_pos(3),
  object_rel_pos(3), gripper_state(2), object_rot(3), object_velp(3),
  object_velr(3), grip_velp(3), gripper_vel(2)`.
- Action: `Box(-1, 1, (4,))` = `[dx, dy, dz, gripper]`.
- **The arm is NOT joint-torque controlled.** `env.unwrapped.model.nu == 2`
  — only the gripper fingers are real actuators. `dx/dy/dz` move a mocap
  body (`robot0:mocap`) the gripper is welded to via an equality constraint;
  MuJoCo's solver resolves the arm's joint angles to follow it. This is why
  `approaches/scripted_ik` uses its own from-scratch actuated arm instead of
  this environment — there's no joint-level control surface here for a
  Jacobian-IK exercise to act on.
- A separate `AdroitHandRelocateDense-v1` (etc.) import-time warning was
  observed: reward functions for the Adroit `-v1` dense tasks changed in
  `gymnasium-robotics==1.2.1` without a version bump. Use `-v2` (available
  from `1.4.3+`) for the corrected reward, or pin `1.2.0` if you need the
  original v1 behavior reproduced exactly. (This repo doesn't use the Adroit
  tasks, but it's a real gotcha worth knowing if you branch out to them —
  see `envs/README.md`.)

Sources: https://robotics.farama.org/ , https://pypi.org/project/gymnasium-robotics/ ,
and direct inspection as described above.

---

## 3. LeRobot: dataset format and training/eval API, verified directly

Doc pages found by web search described an older/different CLI surface than
what `lerobot==0.6.1` actually installs (e.g. a `lerobot-record --robot.type=...`
hardware-teleop-oriented flow, not a simple "record from any Gymnasium env"
entry point). Rather than guess at how to bend that CLI to a Gymnasium env,
this repo calls `LeRobotDataset`'s Python API directly — verified end-to-end
by actually running it:

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

features = {
    "observation.state": {"dtype": "float32", "shape": (N,), "names": [...]},
    "action": {"dtype": "float32", "shape": (M,), "names": [...]},
    "observation.images.top": {"dtype": "video", "shape": (H, W, 3), "names": ["height","width","channel"]},
}
ds = LeRobotDataset.create(repo_id="user/name", fps=10, features=features, root="local/path", use_videos=True)
ds.add_frame({"observation.state": ..., "action": ..., "observation.images.top": ..., "task": "description string"})
ds.save_episode()   # after each episode's frames are added
ds.finalize()        # once, after all episodes
```

`add_frame`'s dict **must** include a `"task"` key (a string) alongside the
feature keys declared in `features`. `LeRobotDataset.create(root=...)`
writes a fully local dataset — **no Hugging Face account or internet access
needed** for this step; `push_to_hub()` is a separate, optional call.

**Training**, via the real `lerobot-train` CLI (verified: launched real
training against a dataset built exactly as above):
```bash
lerobot-train \
  --dataset.repo_id=local/my_dataset --dataset.root=/path/to/dataset \
  --policy.type=act --policy.device=cpu --policy.push_to_hub=false \
  --output_dir=out --job_name=run1 --steps=5000 --batch_size=8 \
  --save_checkpoint=true --wandb.enable=false
```
`--policy.type` also accepts `diffusion` (see gotcha below) and many others
(`smolvla`, `pi0`, `vqbet`, ...) not used in this repo.

Two extras are required beyond a bare `pip install lerobot`, discovered by
running the command and reading the resulting `ImportError` (both name the
exact fix, so this wasn't a guess):
- `pip install "lerobot[training]"` — adds `accelerate`; without it,
  `lerobot-train` fails immediately with `'accelerate' is required...`.
- `pip install "lerobot[diffusion]"` — adds `diffusers`; without it,
  `--policy.type=diffusion` fails with `'diffusers' is required...`.

**Diffusion Policy needs longer episodes than ACT.** A 5-frame-per-episode
test dataset trained ACT fine but made `lerobot-train --policy.type=diffusion`
fail outright: `ValueError: No valid frames remain after applying
drop_n_first_frames and drop_n_last_frames.` — Diffusion's default
observation/action horizon config drops several frames off each end of
every episode, and 5 frames wasn't enough to survive that. Rebuilding the
same dataset with 30-frame episodes fixed it immediately. This repo's
`collect_demos.py` produces 50-step episodes, comfortably clear of this.

**Inference on a trained checkpoint**, verified against a real (if
undertrained) checkpoint:
```python
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class
from lerobot.policies import make_pre_post_processors

config = PreTrainedConfig.from_pretrained(checkpoint_dir)   # checkpoint_dir = .../pretrained_model/
policy = get_policy_class(config.type).from_pretrained(checkpoint_dir)
policy.eval()
preprocessor, postprocessor = make_pre_post_processors(config, pretrained_path=checkpoint_dir)

policy.reset()
batch = preprocessor({"observation.state": ..., "observation.images.top": ...})  # torch tensors, batch dim 1
action = postprocessor(policy.select_action(batch))
```
This dispatches correctly regardless of whether the checkpoint is ACT or
Diffusion (`get_policy_class(config.type)` reads the policy type out of the
checkpoint's own saved config), which is why `approaches/imitation_act/evaluate.py`
and `approaches/imitation_diffusion/evaluate.py` are identical.

**Model sizes and CPU speed, measured directly** (Apple M-series CPU, this
repo's exact feature shapes — 25-dim state, 4-dim action, 96x96x3 image):
- ACT: 52M params, ~5-6 steps/sec at batch size 2-4 on CPU.
- Diffusion Policy: 264M params, ~1 step/sec at batch size 2 — noticeably
  heavier. First training run of either also downloads a pretrained
  ResNet-18 backbone (~45MB, from `download.pytorch.org`) — needs internet
  access once; cached afterward under `~/.cache/torch/hub/checkpoints/`.

**ACT on Apple Silicon GPU (`--policy.device=mps`), measured directly**: a
real 5000-step run (batch size 8, this repo's `train_act.py`) completed in
5:07 at ~17 steps/sec — roughly 3x faster than the CPU numbers above, with
no AMP-related errors or need for `--policy.use_amp=false`. Loss dropped
cleanly over the run: `loss:5.679` at step 200 (epoch 1.6) down to
`loss:0.223` at step 5000 (epoch 40) — real evidence the training loop is
learning, not just running. The resulting checkpoint scored 0/5 on
`evaluate.py`, exactly as expected at this scale (20 demo episodes, 40
epochs is nowhere near LeRobot's own ~100k-step guidance for a real task).

**Benign warning you will see and can ignore**: on machines without a
compatible system FFmpeg, `torchcodec` fails to load with a long traceback
ending in `'torchcodec' is installed but cannot be loaded ... Falling back
to 'pyav' as a default decoder.` This is not an error — `pyav` (already a
`lerobot` dependency) is used instead and everything still works. Installing
FFmpeg (`brew install ffmpeg` on macOS) silences the warning but is not
required.

**Also benign**: `AttributeError: 'NoneType' object has no attribute
'get_current_context'` printed from `OffScreenViewer.__del__`/`GLContext.__del__`
at interpreter shutdown, after a script using `mujoco.Renderer` finishes.
This is MuJoCo's offscreen GL context being garbage-collected in an
unhelpful order at process exit; it does not affect anything the script
already did.

**Also benign, macOS**: `objc[...]: Class AVFAudioReceiver is implemented in
both .../cv2/.dylibs/libavdevice....dylib and .../av/.dylibs/libavdevice....dylib.
This may cause spurious casting failures and mysterious crashes.` This
appears once `opencv-python` (pulled in transitively by some `lerobot`
dependency chains) and `av`/`pyav` are both installed, since each bundles
its own copy of `libavdevice`. Observed during real `train_act.py` and
`evaluate.py` runs with no actual crash or incorrect behavior resulting —
noisy but harmless in this repo's usage (no video capture device access,
which is where this class is actually used). If you want to silence it
rather than ignore it, it means one of `opencv-python`/`opencv-python-headless`
and `av` has a redundant bundled `ffmpeg`; not investigated further since it
doesn't affect correctness here.

**`hf auth login` vs the old `huggingface-cli login`**: current
`huggingface_hub` versions print `Warning: huggingface-cli is deprecated
and no longer works. Use hf instead.` if you try the old command — it's a
full CLI rename, not a deprecation warning you can ignore. Use
`hf auth login` (same token, same effect: writes to
`~/.cache/huggingface/token`, used automatically by `push_to_hub()`).

---

## 3a. macOS-specific: the interactive MuJoCo viewer needs `mjpython`

`mujoco.viewer.launch_passive(model, data)` (used by
`utils/viz.py`'s `run_viewer_loop`, i.e. any `--render viewer` flag in this
repo) raises immediately on macOS under plain `python`:
```
RuntimeError: `launch_passive` requires that the Python script be run under
`mjpython` on macOS
```
This is a real, documented MuJoCo/Cocoa requirement (the viewer needs to
own the main UI thread on macOS), not a bug or a missing dependency.
`mjpython` is installed automatically into your venv alongside the
`mujoco` pip package (`venv/bin/mjpython`) — run the exact same script with
`mjpython` in place of `python`:
```bash
mjpython approaches/scripted_ik/pick_place_scripted.py --render viewer
```
Confirmed working end-to-end this way. `--render video` and `--render none`
are unaffected and run with plain `python` on every platform.

---

## 4. MuJoCo Jacobian IK: three real bugs, in the order they were found

Building `approaches/scripted_ik/ik_solver.py` did **not** work on the first
try. In order:

1. **`mj_jac` vs `mj_jacSite` mixup.** `mj_jac(model, data, jacp, jacr,
   point, body)`'s last argument is a **body id**, and `point` is an
   explicit world-frame coordinate. Passing a **site id** in the body-id slot
   compiles and runs without error, but silently computes the Jacobian of
   the wrong point — IK just never converges, with no exception to point at
   the cause. Fix: use `mj_jacSite(model, data, jacp, jacr, site)`, which
   takes a site id directly and removes this whole failure mode.

2. **Missing `mj_comPos`.** Even with `mj_jacSite`, calling only
   `mj_kinematics(model, data)` before it leaves `data.subtree_com` stale
   (zero, on a freshly constructed `MjData`) — `mj_jacSite` uses it
   internally and produces a garbage/zero Jacobian with, again, no error.
   Fix: call `mj_comPos(model, data)` right after `mj_kinematics` (or just
   call `mj_forward`, which includes both plus more than needed). Confirmed
   by direct experiment: identical code converged perfectly when called on
   an `MjData` that had been through `mj_forward` once, and failed
   identically (0 iterations of progress) on a fresh `MjData` that had only
   seen `mj_kinematics`.

3. **A genuine kinematic singularity at the naive "zero" home pose.** This
   arm's joints 2-4 (shoulder/elbow/wrist_pitch) all rotate about the same
   axis, and at all-zero qpos the arm points straight up along that same
   axis — at that exact configuration, the position Jacobian's y and z rows
   are analytically exactly zero (verified: `np.linalg.matrix_rank(jacp) == 1`
   at qpos=0, vs. 3 at a bent "ready" pose). IK from that starting point
   can't make progress in two of three Cartesian directions no matter how
   many iterations run. Fix: `reset_to_ready_pose()` seeds a bent starting
   configuration instead of the model's raw zero default — the same reason
   real robot arms are never "homed" fully extended.

After fixing all three, `ik_solver.py`'s self-test converges to sub-5mm
error in 2-3 iterations from a realistic starting pose (see that file's
`__main__` block for the exact numbers this produced).

Two further, non-IK bugs surfaced getting the full pick-and-place demo
working, both in `approaches/scripted_ik/`:
- **Gravity droop from underpowered position-actuator gains.** At the
  scene's initial `kp=60, kv=6`, the arm's actual settled pose under gravity
  differed from the IK-commanded joint targets by several centimeters
  (measured directly: commanded end-effector target vs. `site_xpos` after
  400 settling steps). Raised to `kp=300, kv=20`.
- **An end-effector site placed at the gripper's base, not between the
  finger pads** — an ~8cm error between where IK aimed and where the
  fingers could actually close around something, discovered by directly
  computing the finger geoms' local z-span (`[0.05, 0.14]`) versus the
  site's original local z (`0.06`, just barely inside the span's low end,
  nowhere near the pads' effective grasp region). Moved to `0.10`.
- **A zero-margin gripper-close target.** Commanding the finger joints to
  exactly `0.0` (the box's exact surface, by construction) generated
  ~zero squeeze force — nothing to push against until asked to go slightly
  past the surface. Widened the finger joint range to allow `-0.015` and
  commanded `-0.01` when closing, so the position servo genuinely
  compresses against the box (confirmed via `data.ncon`/`data.contact`:
  real box-finger contacts with nonzero penetration appeared only after
  this fix).

**End result, actually measured**: 10/10 successful grasps across
randomized box start positions (seeds 0-9), lifting the box 55-80mm off the
table each time.

Sources for the method itself:
https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html
(`mj_jacSite`, `mj_comPos`); Buss, *"Introduction to Inverse Kinematics with
Jacobian Transpose, Pseudoinverse and Damped Least Squares methods"*,
https://www.math.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf ;
https://pab47.github.io/mujoco/notes/Lec6_double_pendulum_ik.pdf

---

## 5. Video export gotcha

`imageio.mimsave(path, frames, fps=fps)` alone, writing a `.mp4`, fails
inside `av.codec.codec.Codec.__cinit__` with `TypeError: expected bytes,
NoneType found` on `imageio==2.37.4`'s `pyav` backend — the codec is not
reliably inferred from the `.mp4` extension alone for `mimsave`. Fix: pass
`codec="libx264"` explicitly. Confirmed working afterward: a 387-frame,
480x480 video was written and read back successfully.

**A bare `pip install -r requirements.txt` (no `pyav`/`lerobot` yet
installed) has no mp4 backend at all**, and fails earlier and differently:
`ValueError: Could not find a backend to open '....mp4' with iomode 'wI'`.
This didn't surface during initial development because `pyav` had already
been pulled in transitively by that point — but a clean venv with only the
base group hits it immediately on the very first `--render video` run. Fix:
added `imageio-ffmpeg` to `requirements.txt`'s base group (bundles its own
ffmpeg binary, no `brew install ffmpeg` required). Confirmed working on a
fresh venv after adding it.

---

## 6. Verification log (what was actually run, not just written)

In the order performed, all in a Python 3.12.14 venv with the pinned
versions above:

1. `gym.make("FetchReach-v4")` / `FetchPickAndPlace-v4"`: reset, sample
   action, step, render — all confirmed (found the v3->v4 deprecation here).
2. `ik_solver.py`'s convergence self-test: failed twice (bugs #1 and #2
   above), then converged sub-5mm in 2-3 iterations after fixes.
3. `pick_place_scripted.py --render none`: ran end-to-end without crashing
   on the first try, but reported grasp failure (box height decreased);
   diagnosed via direct `data.ncon`/`data.contact` inspection and fixed the
   three physical bugs in section 4; then verified 10/10 success across
   seeds 0-9.
4. `pick_place_scripted.py --render video`: failed on `imageio.mimsave`
   (section 5), fixed, then produced a real, readable 387-frame mp4.
5. `LeRobotDataset.create/add_frame/save_episode/finalize`: verified with a
   synthetic random dataset (2 episodes x 5 frames), then again with
   `collect_demos.py`'s real scripted-expert output (5 episodes x 50 frames).
6. `scripted_expert.py` (the collect_demos.py expert policy): 20/20
   successful episodes on `FetchPickAndPlace-v4`, seeds 0-19.
7. `lerobot-train --policy.type=act`: failed once (`accelerate` missing,
   section 3), then ran real training steps (2, then 5) against both a
   synthetic dataset and the real scripted-expert dataset from step 5.
8. `lerobot-train --policy.type=diffusion`: failed once (`diffusers`
   missing), failed again (episode-length gotcha, section 3), then ran real
   training steps after both fixes.
9. Inference (`ACTPolicy`/`get_policy_class` + `make_pre_post_processors` +
   `select_action`) against the checkpoint from step 7: produced a
   well-formed 4-dim action.
10. `approaches/imitation_act/evaluate.py`, full script, against that same
    checkpoint: ran 2 real episodes in the live environment (0% success, as
    expected from a 5-step-trained checkpoint) and saved a video.
11. `stable_baselines3.PPO("MultiInputPolicy", ...)` on both
    `FetchReachDense-v4` and `FetchPickAndPlaceDense-v4`: trained for a
    small number of timesteps on each, confirmed `.predict()` works.
12. `approaches/rl_grasp/train_rl.py --task reach --timesteps 256`: full
    script run end-to-end, produced a saved model and a measured 67%
    (2/3) held-out success rate.
13. `pick_place_scripted.py --render viewer` on macOS: failed once under
    plain `python` (section 3a's `RuntimeError`), then confirmed working
    under `mjpython`.
14. Fresh venv, base `requirements.txt` only, `pick_place_scripted.py
    --render video`: failed on the missing-backend error (section 5),
    fixed by adding `imageio-ffmpeg`, then produced a working mp4.
15. `train_act.py --steps 5000 --batch_size 8 --device mps` on Apple
    Silicon: full real run, 5:07 wall time, ~17 steps/sec, loss
    5.679 → 0.223 (see section 3a's MPS entry for the full numbers), then
    `evaluate.py --episodes 5`: 0/5 success, exactly as expected at this
    scale — real end-to-end confirmation of the GPU training path, not just
    the CPU path from steps 7-10.

**Not run**: any training to actual convergence (tens of thousands+ steps
for ACT/Diffusion, hundreds of thousands+ for PPO on `pick_place`) — that's
real compute time, and is the natural next thing for whoever is running
this to do themselves. Every script that would be involved in that was
still run for real, just for a small number of steps, to confirm the wiring
is correct end-to-end rather than merely written to look correct.
