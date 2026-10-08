# Quickstart: a policy training in 10 minutes

This walks the fastest path from a fresh clone to a real (if undertrained)
learned policy — using `imitation_act`, the recommended first approach.
For the guaranteed-no-training-needed demo, see
`approaches/scripted_ik/README.md` instead (it takes about 30 seconds, not
10 minutes, but there's nothing to train).

Assumes you've already done `setup_guide.md`'s setup and verification
steps.

## 1. Install LeRobot's training extra (~2 min)

```bash
pip install "lerobot[training]"
```

## 2. Collect demonstrations (~1 min)

```bash
python approaches/imitation_act/collect_demos.py --num_episodes 20
```

You should see 20 lines like `seed N: SUCCESS, 50 steps -> outputs/demos/raw/episode_00NN.npz`.
This uses a scripted expert (no teleop, no human) — see
`approaches/imitation_act/scripted_expert.py`.

## 3. Build the dataset (~30 sec)

```bash
python approaches/imitation_act/format_dataset.py \
    --demos_dir outputs/demos/raw \
    --repo_id local/fetch_pick_place \
    --root outputs/datasets/fetch_pick_place
```

You should see `Dataset created at outputs/datasets/fetch_pick_place`,
`episodes: 20, frames: 1000` (or close to it).

## 4. Train for a few minutes, not a few hours (~5 min on CPU)

A real training run to convergence takes hours (see
`approaches/imitation_act/README.md`) — for a 10-minute quickstart, run just
enough steps to see the loss move and confirm the whole pipeline works:

```bash
python approaches/imitation_act/train_act.py \
    --dataset_root outputs/datasets/fetch_pick_place \
    --steps 500 --batch_size 8
```

## 5. Evaluate (~1 min)

```bash
python approaches/imitation_act/evaluate.py \
    --checkpoint outputs/train/act/checkpoints/last/pretrained_model \
    --episodes 5 --video outputs/eval_quickstart.mp4
```

**Expect a low or 0% success rate** — 500 steps is a pipeline sanity check,
not a trained policy (LeRobot's own guidance is more like 100k steps for a
real task). What you should have at the end of this is: a real dataset, a
real checkpoint, a real evaluation video, and confidence that the full
pipeline works on your machine — so a longer real training run (change
`--steps 500` to `--steps 50000` or more, and go do something else while it
runs) is just a matter of time, not debugging.

## What's next

- Let `train_act.py` run much longer (thousands of steps) and re-run
  `evaluate.py` to see the success rate actually improve.
- Try `approaches/imitation_diffusion` on the exact same dataset you already
  built here — just swap which training script you call.
- Read `approaches/imitation_act/README.md`'s "Learning more" links to
  understand what ACT is actually doing, not just how to run it.
