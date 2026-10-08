# envs

Thin, documented wrappers around [Gymnasium-Robotics](https://robotics.farama.org/)
tasks, used by the `imitation_act`, `imitation_diffusion`, and `rl_grasp`
approaches. `approaches/scripted_ik` does **not** use these — it has its own
standalone actuated arm (see that approach's README for why: Fetch's arm is
mocap-driven, not joint-torque driven, which doesn't suit a from-scratch
Jacobian-IK exercise).

## Available tasks

| File | Env ID | Task | Obs shape | Action shape |
|---|---|---|---|---|
| `arm_reach.py` | `FetchReach-v4` | move the gripper to a 3D target point | dict: (10,)+(3,)+(3,) | `Box(-1,1,(4,))` |
| `arm_pick_place.py` | `FetchPickAndPlace-v4` | pick up a block, move it to a target | dict: (25,)+(3,)+(3,) | `Box(-1,1,(4,))` |

Both wrappers also expose a `reward_type="dense"` option
(`FetchReachDense-v4` / `FetchPickAndPlaceDense-v4`) for plain RL without
Hindsight Experience Replay — see `approaches/rl_grasp`.

**Start with `arm_reach.py`** if this is your first time running anything in
this repo — it's the simplest possible check that your MuJoCo/Gymnasium
install works:

```bash
python envs/arm_reach.py
```

## Important version gotcha

`FetchReach-v3` / `FetchPickAndPlace-v3` (and the other Fetch tasks' `-v3`)
are **deprecated** as of `gymnasium-robotics>=1.3` — you'll get a
`gymnasium.error.DeprecatedEnv` if you (or an old tutorial) uses `-v3`. Use
`-v4`. See `RESEARCH_NOTES.md` for how this was discovered.

## Action space, in plain terms

Both tasks share a 4-dim action: `[dx, dy, dz, gripper]`. **This is not
joint control** — `dx/dy/dz` move a mocap target the gripper is welded to
(MuJoCo resolves the arm's joint angles to follow it), and `gripper` opens/
closes the fingers. If you inspect the underlying MuJoCo model
(`env.unwrapped.model`) you'll find only 2 actuators (the gripper fingers) —
the arm itself has none. This is a deliberate simplification Gymnasium-
Robotics makes for RL benchmarking; it's also exactly why `scripted_ik`
needed its own model with real joint actuators to teach Jacobian IK
meaningfully.

## Other tasks that exist but aren't wrapped here

Gymnasium-Robotics also ships `FetchPush`, `FetchSlide`, and the
Adroit/Shadow Hand tasks (`AdroitHandRelocate`, `AdroitHandHammer`,
`AdroitHandDoor`, etc.). We didn't wrap these to keep the sandbox focused,
but they register the same way (`gym.make("FetchPush-v4")` etc.) if a member
wants to experiment. One note found during research: the Adroit `-v1` dense
reward functions changed in `gymnasium-robotics==1.2.1` without a version
bump — gymnasium-robotics itself prints a warning about this on import; use
`-v2` (available from `gymnasium-robotics>=1.4.3`) for the corrected reward,
or pin `gymnasium-robotics==1.2.0` if you specifically need v1's original
behavior reproduced.
