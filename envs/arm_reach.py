"""Thin wrapper around Gymnasium-Robotics' FetchReach task -- the simplest
task in this repo, good for a first-timer's "does my setup even work" check
and for a quick RL warm-up before attempting FetchPickAndPlace.

VERIFIED: run during development against the same pinned versions as
arm_pick_place.py (see RESEARCH_NOTES.md). Same v3->v4 deprecation gotcha
applies here too.

Observation: dict with 'observation' (10,) = grip_pos(3), gripper_state(2),
grip_velp(3), gripper_vel(2); 'achieved_goal' (3,) = grip_pos;
'desired_goal' (3,) = target position. Action: Box(-1,1,(4,)) = [dx,dy,dz,
gripper] (gripper dimension is present but has no visible effect -- there is
nothing to grasp in this task).
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
    _ensure_registered()
    return gym.make(
        "FetchReach-v4" if reward_type == "sparse" else "FetchReachDense-v4",
        render_mode=render_mode,
    )


if __name__ == "__main__":
    # Quick smoke test: python envs/arm_reach.py
    env = make_env()
    obs, info = env.reset(seed=0)
    print("observation keys:", list(obs.keys()))
    for k, v in obs.items():
        print(f"  {k}: shape={v.shape}")
    print("action_space:", env.action_space)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print(f"step ok, reward={reward}")
    env.close()
    print("OK")
