"""Thin, documented wrapper around Gymnasium-Robotics' Fetch pick-and-place
task. Used by approaches/imitation_act, approaches/imitation_diffusion, and
approaches/rl_grasp.

VERIFIED: `make_env()` was run during development against
gymnasium==1.3.0 + gymnasium-robotics==1.4.2 + mujoco==3.2.7 -- reset/step/
render all confirmed working (see RESEARCH_NOTES.md).

IMPORTANT VERSION GOTCHA (see RESEARCH_NOTES.md for the full story):
FetchPickAndPlace-v3 is DEPRECATED as of gymnasium-robotics>=1.3 -- you must
use v4. Several tutorials/blog posts online still reference v3; if you copy
code from one of those and get a `gymnasium.error.DeprecatedEnv`, this is
why.

ENVIRONMENT DETAILS (from actually inspecting a live instance, not docs):
  observation: dict with keys 'observation' (25,), 'achieved_goal' (3,),
    'desired_goal' (3,). The 25-dim 'observation' vector is, in order:
    grip_pos (3), object_pos (3), object_rel_pos (3), gripper_state (2),
    object_rot (3), object_velp (3), object_velr (3), grip_velp (3),
    gripper_vel (2). This ordering is documented at
    https://robotics.farama.org/envs/fetch/pick_and_place/ and matches what
    was observed here.
  action: Box(-1, 1, (4,)) = [dx, dy, dz, gripper]. THE ARM IS NOT JOINT-
    TORQUE CONTROLLED: dx/dy/dz move a mocap body the gripper is welded to
    (confirmed by inspecting the model: only 2 actuators exist -- the
    gripper fingers -- everything else moves via the mocap weld constraint).
    This is why approaches/scripted_ik uses its OWN arm model instead of
    this one for a from-scratch Jacobian-IK demo: there's no joint-level
    actuation here for that exercise to act on.
  reward: sparse by default (-1 per step until success, 0 on success) unless
    you request the dense variant (see make_env(reward_type=...)).
"""
import gymnasium as gym
import gymnasium_robotics

_REGISTERED = False


def _ensure_registered():
    global _REGISTERED
    if not _REGISTERED:
        gym.register_envs(gymnasium_robotics)
        _REGISTERED = True


def make_env(render_mode: str | None = None, reward_type: str = "sparse"):
    """render_mode: None (fastest, no rendering), 'rgb_array' (for saving
    frames/video), or 'human' (opens a window -- not verified in this
    development sandbox, see utils/viz.py's note on displays).
    reward_type: 'sparse' (default, matches the original Fetch/HER paper) or
    'dense' (shaped, easier for vanilla RL without HER -- used by
    approaches/rl_grasp)."""
    _ensure_registered()
    return gym.make(
        "FetchPickAndPlace-v4" if reward_type == "sparse" else "FetchPickAndPlaceDense-v4",
        render_mode=render_mode,
    )


if __name__ == "__main__":
    # Quick smoke test: python envs/arm_pick_place.py
    env = make_env(render_mode="rgb_array")
    obs, info = env.reset(seed=0)
    print("observation keys:", list(obs.keys()))
    for k, v in obs.items():
        print(f"  {k}: shape={v.shape}")
    print("action_space:", env.action_space)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"step ok, reward={reward}, is_success={info.get('is_success')}")
    frame = env.render()
    print("rendered frame shape:", frame.shape)
    env.close()
    print("OK")
