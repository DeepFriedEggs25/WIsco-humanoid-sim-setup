# manipulation_sim

A MuJoCo + Gymnasium manipulation learning sandbox — pick up a simulated
robot arm, watch it move, teach it a task, entirely on a laptop. No physical
hardware, no Isaac Sim, no Docker, no GPU required for the basics (a GPU
just makes the training steps faster, see below).

This is an **onboarding playground**, not a production codebase or a
work-in-progress project — it's meant to be cloned, run, and picked apart by
anyone new to robot manipulation or to this kind of sim-to-policy pipeline,
whether that's a new member of a robotics team or just someone curious. If
your team also has a real-hardware repo, this is deliberately kept separate
from it: everything here is 100% simulated, so there's no way to break
anything by experimenting.

## The full walkthrough

This section is the actual path from a fresh clone to a trained (if
undertrained) policy, in the order you'd really run it — including the
rough edges you'll actually hit, not just the happy path.

### 0. Set up your environment

Follow **[`setup_guide.md`](setup_guide.md)** for your OS (Windows/WSL2,
macOS, Linux) — Python 3.12, a venv, `pip install -r requirements.txt`.
Come back here once that's done.

### 1. Verify MuJoCo + Gymnasium work at all

```bash
python envs/arm_reach.py
```
Expect: prints observation/action shapes, ends with `OK`. This is the
simplest possible check that your install isn't broken — no rendering, no
training, just "can I load a MuJoCo model and step physics through
Gymnasium's API."

### 2. Actually watch something move

Everything above ran headless (text only). To see the arm in MuJoCo, run
the reference demo (see step 3) with one of these two render modes:

```bash
# Option A: live interactive window
mjpython approaches/scripted_ik/pick_place_scripted.py --render viewer
```
**On macOS you must use `mjpython`, not `python`, for the interactive
viewer** — MuJoCo's viewer needs it for a Cocoa/UI-threading reason, and
`launch_passive` will refuse to run under plain `python` with a clear error
telling you this. `mjpython` is installed automatically alongside the
`mujoco` pip package (`venv/bin/mjpython`) — nothing extra to install.
Close the window when you're done watching.

```bash
# Option B: save a video and open it
python approaches/scripted_ik/pick_place_scripted.py --render video --out outputs/pick_place.mp4
open outputs/pick_place.mp4   # macOS; use your OS's equivalent elsewhere
```
This needs `imageio-ffmpeg` (already in `requirements.txt`) to actually
encode the mp4 — without it you'll get
`ValueError: Could not find a backend to open '....mp4'`. If you hit that
anyway (e.g. an older checkout), `pip install imageio-ffmpeg` fixes it; see
`RESEARCH_NOTES.md` section 5 for the full story.

### 3. Run the guaranteed, no-training demo

```bash
python approaches/scripted_ik/pick_place_scripted.py
```
This solves real inverse kinematics with `mujoco.mj_jacSite` and a damped
least-squares Jacobian pseudoinverse to make a simulated arm pick up a box —
verified working 10/10 times across randomized start positions during
development (see `RESEARCH_NOTES.md`). No learning involved; this is the
one guaranteed to work and is worth understanding before touching any of
the learning-based approaches.

### 4. Pick a learning approach

**[`approaches/README.md`](approaches/README.md)** compares all four:
classical IK (what you just ran), ACT, Diffusion Policy, and RL (PPO). Pick
whichever matches what you're trying to learn — you don't need to do all
four.

### 5. Walk the recommended path end-to-end: `imitation_act`

This is the fastest way to go from nothing to a real (if undertrained)
learned policy, and covers the full modern imitation-learning workflow
(collect → format → train → evaluate) that LeRobot/Hugging Face's tooling
is built around.

```bash
pip install "lerobot[training]"
```

**Record episodes.** A scripted expert drives the arm through
`FetchPickAndPlace-v4` — no teleop, no human operator needed. Each episode
resets to a *randomized* box position and goal position (that's built into
the environment itself, not something you configure), so even a small batch
of episodes already covers a range of starting conditions, not one fixed
spot:
```bash
python approaches/imitation_act/collect_demos.py --num_episodes 20
```
You should see 20 lines like `seed N: SUCCESS, 50 steps -> outputs/demos/raw/episode_00NN.npz`.

**Build the dataset, and optionally push it to Hugging Face.** This step
converts the raw `.npz` episodes into a real `LeRobotDataset` (parquet +
mp4, the format LeRobot's training code actually reads):
```bash
python approaches/imitation_act/format_dataset.py \
    --demos_dir outputs/demos/raw \
    --repo_id local/fetch_pick_place \
    --root outputs/datasets/fetch_pick_place
```
This alone needs no Hugging Face account — `--root` writes a fully local
dataset. If you want it on the Hub too (to share it, or to train from
elsewhere later), do this first, once:
1. Create a Hugging Face account if you don't have one.
2. On huggingface.co: **Settings → Access Tokens → New token**, role
   **Write**. Copy it.
3. Locally: `hf auth login`, paste the token. (The older
   `huggingface-cli login` command is deprecated in current
   `huggingface_hub` versions and will tell you to use `hf` instead.)

Then re-run with `--push_to_hub` (pointing `--root` at a fresh folder, or
deleting the old one first — `LeRobotDataset.create()` refuses to overwrite
an existing `root`):
```bash
python approaches/imitation_act/format_dataset.py \
    --demos_dir outputs/demos/raw \
    --repo_id <your-hf-username>/fetch_pick_place \
    --root outputs/datasets/fetch_pick_place \
    --push_to_hub
```
Check `https://huggingface.co/datasets/<your-hf-username>/fetch_pick_place`
afterward — the repo is created automatically on first push, no need to
create it manually on the site first.

**Train.** A real training run to convergence takes hours; a few thousand
steps is enough to confirm the whole pipeline works and see the loss
actually move:
```bash
python approaches/imitation_act/train_act.py \
    --dataset_root outputs/datasets/fetch_pick_place \
    --steps 5000 --batch_size 8 --device mps
```
`--device mps` uses the GPU on Apple Silicon. This has been directly
confirmed to work with no extra flags needed: a real 5000-step run on an
M-series Mac completed in about 5 minutes at ~17 steps/sec (roughly 3x
faster than the ~5-6 steps/sec measured on CPU for the same model). Use
`--device cpu` on Intel Macs or if you hit GPU-specific errors.

If you already pushed your dataset to the Hub and want to train from that
copy instead of the local folder, use `--from_hub` and drop
`--dataset_root`:
```bash
python approaches/imitation_act/train_act.py \
    --repo_id <your-hf-username>/fetch_pick_place --from_hub \
    --steps 5000 --device mps
```

**Evaluate.**
```bash
python approaches/imitation_act/evaluate.py \
    --checkpoint outputs/train/act/checkpoints/last/pretrained_model \
    --episodes 5 --video outputs/eval.mp4
```

### 6. What to actually expect at this scale

**A 0% (or near-0%) success rate here is the expected, correct result** —
not a bug, and not something to debug. 5000 training steps against 20
episodes (~1000 frames total, so ~40 epochs) is enough to see the loss
curve drop cleanly (a real run went from `loss:5.679` at step 200 down to
`loss:0.223` at step 5000) — that's the pipeline demonstrably learning
*something* — but nowhere near enough data or steps to solve pick-and-place
reliably. LeRobot's own guidance is closer to ~100k steps for a real task,
and this repo's own experiments used on the order of tens of demo episodes,
not the ~100-300 you'd realistically want for a policy that generalizes
across the environment's full randomized start/goal range.

What you should have at the end of this walkthrough: a real recorded
dataset (locally and optionally on the Hub), a real trained checkpoint, a
real evaluation video, and a working, debugged pipeline end to end. Getting
an actually competent policy from here is a matter of scaling both axes —
more episodes (better coverage of start/goal positions) *and* more training
steps (more epochs over that data) — and is real compute time, not further
debugging.

### 7. Going beyond the quickstart: a real training run

Section 6 above is a pipeline sanity check (20 episodes, 5000 steps) —
useful to confirm everything's wired correctly, but not enough data or
steps to actually solve the task reliably. Here's what scaling that up
for real looks like, with real numbers from an actual run:

```bash
# 1. Collect real coverage of the environment's randomized start/goal
#    range, not just a handful of episodes (~10-15 min for 250 episodes)
python approaches/imitation_act/collect_demos.py --num_episodes 250

# 2. Build the dataset and push it to the Hub (see step 5's HF setup above)
python approaches/imitation_act/format_dataset.py \
    --demos_dir outputs/demos/raw \
    --repo_id <your-hf-username>/fetch_pick_place_v2 \
    --root outputs/datasets/fetch_pick_place_v2 \
    --push_to_hub

# 3. Train for real -- 100k steps (LeRobot's own guidance for a real task),
#    sourcing the dataset straight from the Hub instead of the local copy,
#    checkpointing every 10k steps so you can inspect progress along the way
caffeinate -d -i python approaches/imitation_act/train_act.py \
    --repo_id <your-hf-username>/fetch_pick_place_v2 --from_hub \
    --steps 100000 --batch_size 8 --device mps --save_freq 10000
```
`caffeinate -d -i` (macOS) stops the machine from sleeping mid-run — a
screensaver or locked screen is harmless (background compute keeps running
regardless), but actual system sleep would pause it, so this matters for a
run this long. At the same ~17 steps/sec measured for `--device mps` in
section 5, 100,000 steps is on the order of 1.5-1.7 hours — genuinely real
compute time, not something to wait on at your desk.

```bash
# 4. Evaluate with enough episodes for a meaningful number (5 episodes only
#    gives 20% resolution; 30 gives ~3%) -- point this at the NEW run's
#    checkpoint, not the smoke-test one from section 6
python approaches/imitation_act/evaluate.py \
    --checkpoint outputs/train/act_v2/checkpoints/last/pretrained_model \
    --episodes 30 --video outputs/eval_v2.mp4
```

Two things worth knowing before you run this yourself:
- **More episodes needs more steps, not the same step count.** 250
  episodes at ~50 frames each is ~12,500 frames — the same 5000 steps from
  section 6 would now cover proportionally fewer epochs over that bigger
  pool. Scale both together.
- **A trained checkpoint is easiest to actually play with, not just
  benchmark, via `python play.py`** — an interactive MuJoCo window built
  alongside this training run. Pick a checkpoint, type exact block/target
  coordinates on-screen, and watch the policy attempt it live, retrying up
  to X times with the first success stopping the loop early. It also has an
  experimental "nudged ACT" mode that fine-tunes live against a one-shot
  demo from this repo's own scripted expert for the exact spot you're
  testing — worth trying to see test-time fine-tuning happen in real time,
  but it's only possible because simulation hands you a perfect oracle with
  privileged access to true object position; a real robot has no such
  oracle, which is the actual reason this technique doesn't carry over to
  real hardware as-is.

## What's in here

```
manipulation_sim/
  README.md               you are here
  RESEARCH_NOTES.md        pinned versions, verified APIs, every bug found & fixed
  setup_guide.md           Windows(WSL)/Mac/Linux setup
  requirements.txt         pinned, laptop-friendly base deps
  envs/                    Gymnasium-Robotics task wrappers (Fetch reach, pick-and-place)
  approaches/
    scripted_ik/            classical Jacobian IK -- fully verified, no training needed
    imitation_act/          ACT via LeRobot -- collect, format, train, evaluate
    imitation_diffusion/    Diffusion Policy via LeRobot -- same dataset as ACT
    rl_grasp/                SB3 PPO on a Fetch task
  utils/                   shared rendering/recording/path helpers
  examples/
    quickstart.md           a policy training in 10 minutes
  play.py                  interactive MuJoCo launcher for a trained checkpoint
                            (pick a model, set exact coordinates, watch it play)
```

## Why "learning sandbox" and not "production"

Every script here favors clarity and a working end-to-end path over
generality or performance. Config files, abstraction layers, and
configurability were deliberately kept minimal — if you need to change
something, edit the script directly; it's short enough to read in full.
This is also why there are only two Gymnasium tasks wrapped (reach,
pick-and-place) rather than the full Fetch/Adroit suite — see
`envs/README.md` for how to add more if you want them.

## What's actually been verified vs. what's scaffolded

Every script in this repo was run for real during development — not just
written to look plausible. What differs by approach is *how far*:

- **`scripted_ik`**: verified completely. 10/10 successful grasps across
  randomized seeds, and both the interactive viewer and video render paths
  confirmed working (including the macOS `mjpython` requirement above).
  This is the one guaranteed to work on your machine too.
- **`imitation_act` / `imitation_diffusion`**: the full pipeline (collect →
  format → train → evaluate) was run end-to-end for real, including a real
  ~5000-step ACT training run on Apple Silicon GPU (`--device mps`,
  ~17 steps/sec) producing the expected 0% success rate at that scale (see
  section 6 above and `RESEARCH_NOTES.md`'s verification log for the exact
  numbers). Training to actual convergence (tens of thousands+ steps) is
  real compute time and hasn't been run here — that's the natural next
  thing to try once the pipeline itself is confirmed working, which it is.
- **`rl_grasp`**: verified the same way — a short real training run
  (256 timesteps) produced a real saved model with a measured 67% success
  rate on the easy `reach` task. A serious `pick_place` run needs much more
  compute and patience (see that approach's README for why it's a genuinely
  hard RL problem).

`RESEARCH_NOTES.md`'s "Verification log" lists the exact commands run and
what they produced, including every real bug found and fixed along the way
— in the IK solver, in LeRobot version compatibility, in video export, and
in macOS-specific rendering. If you hit something not listed there, you've
found a new one — add it.

## Going further

Each approach's own README ends with a "Learning more" section — papers,
docs, and talks for understanding *why* the method works, not just how to
run it. Once you've got a pipeline working, the natural next steps are the
same ones called out throughout this repo: scale up episode count and
training steps for the imitation approaches, try Diffusion Policy on the
same dataset you already built for ACT, or push further into RL with
HER+SAC where `rl_grasp` leaves off.
