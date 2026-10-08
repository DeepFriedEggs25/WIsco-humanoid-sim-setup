# Approaches

Four ways to make a simulated arm do useful manipulation. **Pick one to
learn deeply — you are not expected to do all four.** Each has its own
README with exact commands, expected output, and links to learn the
underlying method properly.

| Approach | Method | Needs training? | Needs a dataset? | Best for |
|---|---|---|---|---|
| [`scripted_ik/`](scripted_ik/) | Classical: Jacobian pseudoinverse IK | No | No | Understanding robot kinematics/control; a guaranteed-working demo |
| [`imitation_act/`](imitation_act/) | Imitation learning: ACT (transformer) | Yes (a few hours) | Yes (collected by a scripted expert, no teleop needed) | Learning modern imitation learning / LeRobot |
| [`imitation_diffusion/`](imitation_diffusion/) | Imitation learning: Diffusion Policy | Yes (longer than ACT) | Yes (same dataset as ACT) | Comparing diffusion-based policies to ACT |
| [`rl_grasp/`](rl_grasp/) | Reinforcement learning: PPO | Yes (many timesteps) | No | Learning RL fundamentals |

## How to choose

- Want to **guarantee something works** for a demo/presentation with zero
  training risk? `scripted_ik`.
- Want to learn the **current standard imitation-learning workflow** used
  across real robotics labs and companies right now (LeRobot is what
  HuggingFace, and a large chunk of the field, actually uses)? `imitation_act`.
- Already comfortable with ACT and curious about a different policy class
  (diffusion models)? `imitation_diffusion` — it's a small delta on top of
  `imitation_act`, same dataset.
- Want to learn **RL** specifically, not imitation learning? `rl_grasp`.

## What "done" looks like

Once you've picked an approach and gotten it working, you should have: a
real measured success rate (or, for `scripted_ik`, just confirmation it
works — there's no rate to measure), a saved checkpoint or model where
applicable, and a video or screenshot of it running. The point is to have
gone through a real pipeline once, end to end — not to produce a
publishable result. If you want to share what you built with your team,
that's entirely up to you; nothing here requires it.
