# RL grasp (Stable-Baselines3 PPO)

**Status: verified working end-to-end.** A short smoke-test run
(`--task reach --timesteps 256`) was executed during development and
produced a real trained policy with a 67% success rate on held-out episodes
— see `RESEARCH_NOTES.md` "Verification log". Longer runs (the kind that
actually produce a good policy) are, as always, on you.

## What this is

Plain single-agent RL: [PPO](https://arxiv.org/abs/1707.06347) via
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/), trained
directly against a Gymnasium-Robotics Fetch task with a dense reward. No
demonstrations, no imitation — the policy learns purely from trial and
error and the reward signal.

## Run it

```bash
# easy warm-up: reach is close to trivial for PPO with a dense reward
python approaches/rl_grasp/train_rl.py --task reach --timesteps 50000

# the actual grasp task -- much harder, expect to need far more than this
# to see a good success rate (see "Why this is hard" below)
python approaches/rl_grasp/train_rl.py --task pick_place --timesteps 500000
```

Both print a held-out success rate at the end and save the trained model to
`outputs/rl/<task>_ppo.zip` (loadable with `PPO.load(path)`).

## Why this is harder than it looks

Fetch tasks default to a **sparse** reward (-1 every step, 0 on success).
The original [Fetch/HER paper](https://arxiv.org/abs/1802.09464)'s entire
point was that you need **Hindsight Experience Replay** (or a dense reward)
to learn anything in a reasonable number of samples with sparse rewards —
and SB3's HER implementation only pairs with off-policy algorithms (SAC,
TD3, DQN), not PPO. So `train_rl.py` uses the **dense** reward variant to
give plain on-policy PPO a fighting chance. Even then, `FetchPickAndPlace`
is a genuinely hard exploration problem — don't be surprised if a laptop-
scale run gets a low success rate. If you want to actually push on this
seriously, the natural next step (not implemented here) is SAC+HER, which
is what the original paper and most follow-up work actually uses for this
task.

## Learning more

- PPO paper: https://arxiv.org/abs/1707.06347
- Stable-Baselines3 docs: https://stable-baselines3.readthedocs.io/
- Hindsight Experience Replay (why sparse-reward Fetch tasks are hard, and
  the standard fix): https://arxiv.org/abs/1707.01495
- SB3's own HER+SAC example (a good next step beyond this script):
  https://stable-baselines3.readthedocs.io/en/master/modules/her.html
