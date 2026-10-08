# Imitation learning: Diffusion Policy

**Status: verified working the same way and to the same extent as
`imitation_act`** — real end-to-end wiring confirmed at smoke-test scale,
full training-to-convergence not yet run. See that approach's README's
"What was verified vs. what's left to run yourself" section; everything
there applies here too.

## What this is

[Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) (Chi et al.,
2023) frames action prediction as a denoising diffusion process — instead of
directly regressing an action, the model learns to iteratively refine a
noisy action sample into a good one, conditioned on the observation. It
tends to model multi-modal action distributions (multiple valid ways to do
something) better than ACT, at the cost of being heavier and slower to run.

## This approach reuses `imitation_act`'s dataset — no new data collection

Diffusion Policy and ACT can train on the exact same `LeRobotDataset`; only
`--policy.type` changes. If you've already run `imitation_act`'s steps 1-2
(`collect_demos.py` + `format_dataset.py`), skip straight to training here:

```bash
# (only if you haven't already) generate + format demos, see ../imitation_act/README.md
python approaches/imitation_act/collect_demos.py --num_episodes 20
python approaches/imitation_act/format_dataset.py \
    --demos_dir outputs/demos/raw --repo_id local/fetch_pick_place \
    --root outputs/datasets/fetch_pick_place

# train Diffusion Policy (needs `pip install "lerobot[diffusion]"`)
python approaches/imitation_diffusion/train_diffusion.py \
    --dataset_root outputs/datasets/fetch_pick_place \
    --steps 5000

# evaluate (identical script to imitation_act's, see its docstring)
python approaches/imitation_diffusion/evaluate.py \
    --checkpoint outputs/train/diffusion/checkpoints/last/pretrained_model \
    --episodes 10 --video outputs/eval_diffusion.mp4
```

## Gotchas found during development (see `RESEARCH_NOTES.md` for full detail)

1. **Separate install extra**: `pip install "lerobot[diffusion]"` (pulls in
   `diffusers`). Without it, `lerobot-train --policy.type=diffusion` fails
   with a clear `ImportError` naming the fix.
2. **Needs longer episodes than ACT.** Diffusion Policy's default config
   drops the first/last several frames of every episode (for its
   observation/action horizon) — on a 5-frame test episode this dropped
   *every* frame and training refused to start
   ("`No valid frames remain...`"). `collect_demos.py`'s 50-step episodes
   are comfortably long enough; if you ever shorten episodes, check this.
3. **Much heavier than ACT**: 264M params vs. ACT's 52M in this setup, and
   roughly 5x slower per training step on CPU (~1 step/sec vs. ~5-6/sec,
   measured on the same machine). Budget accordingly, or train on a smaller
   `--steps` count first to confirm the pipeline before committing to a long
   run.

## Learning more

- Diffusion Policy paper: *Diffusion Policy: Visuomotor Policy Learning via
  Action Diffusion*, Chi et al. 2023 — https://arxiv.org/abs/2303.04137
- Project page (videos, explainer): https://diffusion-policy.cs.columbia.edu/
- LeRobot's Diffusion Policy docs: https://huggingface.co/docs/lerobot/diffusion_policy
