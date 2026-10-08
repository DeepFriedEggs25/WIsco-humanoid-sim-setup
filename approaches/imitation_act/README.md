# Imitation learning: ACT (Action Chunking with Transformers)

**Status: the full pipeline (collect -> format -> train -> evaluate) is
wired correctly and was verified end-to-end during development** on a tiny
smoke-test scale (5-30 demo episodes, 5-30 training steps), and separately
confirmed with a real ~5000-step run on Apple Silicon GPU (`--device mps`,
~17 steps/sec, loss dropping cleanly from ~5.7 to ~0.22 — see
`RESEARCH_NOTES.md`). A full, converged training run (tens of thousands+
steps) has not been run here — that's real compute time, and is the natural
next thing to try. See "What was verified vs. what's left to run yourself"
below.

## What this is

[ACT](https://arxiv.org/abs/2304.13705) (Zhao et al., 2023) is a transformer
that takes in camera images + robot proprioception and outputs a *chunk* of
future actions (not just one), which is what gives it its name — it's the
policy the [LeRobot](https://huggingface.co/docs/lerobot) team themselves
recommend starting with: lightweight (~52M params in this setup), trains in
hours not days, and works with as few as ~50 demonstrations.

## Pipeline

```
collect_demos.py  -->  format_dataset.py  -->  train_act.py  -->  evaluate.py
(scripted expert       (raw .npz -> real       (lerobot-train      (load checkpoint,
 generates episodes,    LeRobotDataset,          --policy.type=act)  measure success
 no teleop needed)       v3 format)                                  rate, save video)
```

## Run it

```bash
# 1. Generate demonstrations (no lerobot/torch needed for this step)
python approaches/imitation_act/collect_demos.py --num_episodes 20

# 2. Convert to a LeRobotDataset (needs `pip install lerobot`)
# add --push_to_hub (and use a real "<your-hf-username>/..." repo_id) to also
# publish it -- requires `hf auth login` first. See ../../README.md's walkthrough.
python approaches/imitation_act/format_dataset.py \
    --demos_dir outputs/demos/raw \
    --repo_id local/fetch_pick_place \
    --root outputs/datasets/fetch_pick_place

# 3. Train (needs `pip install "lerobot[training]"`)
# --device mps uses Apple Silicon GPU; add --from_hub (drop --dataset_root)
# to train from a dataset you already pushed instead of the local copy.
python approaches/imitation_act/train_act.py \
    --dataset_root outputs/datasets/fetch_pick_place \
    --steps 5000 --device mps

# 4. Evaluate
python approaches/imitation_act/evaluate.py \
    --checkpoint outputs/train/act/checkpoints/last/pretrained_model \
    --episodes 10 --video outputs/eval_act.mp4
```

## What was verified vs. what's left to run yourself

**Verified during development** (see `RESEARCH_NOTES.md` "Verification
log" for exact commands/output):
- `collect_demos.py`: the scripted expert (`scripted_expert.py`) succeeded
  20/20 test episodes on `FetchPickAndPlace-v4`.
- `format_dataset.py`: produced a real, loadable `LeRobotDataset` (checked
  via `lerobot`'s own `LeRobotDataset.create`/`add_frame`/`save_episode`/
  `finalize` API against `lerobot==0.6.1`).
- `train_act.py`: successfully launched `lerobot-train --policy.type=act`
  against that dataset and ran real training steps on CPU (52M params,
  ~5-6 steps/sec on an Apple M-series CPU).
- `evaluate.py`: successfully loaded a checkpoint and ran real inference
  (`policy.select_action`) in the live environment, producing well-formed
  4-dim actions.

**Not verified — this is where the real learning happens:**
- Training to convergence (tens of thousands of steps) and getting a real
  success rate above 0%. Both the 5-30-step smoke test and a real
  5000-step run (see `RESEARCH_NOTES.md`) produced an undertrained policy
  with 0% success, as expected at that scale — that's not a bug.
- Whether 20 demonstrations is enough for this task. ACT's own guidance is
  "often ~50" for a real robot with human teleop; this task's randomized
  start/goal positions likely want more like 100-300 for good coverage —
  try more if your success rate stays low after longer training.

## Learning more

- ACT paper: *Learning Fine-Grained Bimanual Manipulation with Low-Cost
  Hardware*, Zhao et al. 2023 — https://arxiv.org/abs/2304.13705
- LeRobot's own ACT guide (what `train_act.py` wraps):
  https://huggingface.co/docs/lerobot/act
- LeRobot's video tutorial: https://www.youtube.com/watch?v=ft73x0LfGpM
