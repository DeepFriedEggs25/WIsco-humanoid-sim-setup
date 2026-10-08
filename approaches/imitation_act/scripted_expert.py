"""A scripted (non-learned) expert policy for FetchPickAndPlace-v4, used by
collect_demos.py to generate demonstrations without needing teleop hardware
or a human operator.

VERIFIED: 20/20 episodes succeeded in a 20-episode test run during
development (seeds 0-19) -- see RESEARCH_NOTES.md "Verification log".

Reads the raw 25-dim observation vector directly (see envs/arm_pick_place.py
for the confirmed field layout) and runs a simple 4-phase state machine:
  1. move above the object (xy-aligned, some clearance in z)
  2. descend onto the object
  3. close the gripper
  4. move to the goal, gripper closed

This is deliberately simple (proportional control, hard-coded thresholds,
no path planning) -- it only needs to be good enough to generate clean
demonstrations, not to be a general-purpose controller.
"""
import numpy as np

XY_ALIGN_THRESHOLD = 0.02      # meters; how close in x,y before descending
GRIPPER_OPEN_THRESHOLD = 0.055  # sum of both finger positions; above this = "open enough to release/not yet grasped"
ABOVE_HEIGHT = 0.08
PROPORTIONAL_GAIN = 10.0


def scripted_expert_action(obs: dict) -> np.ndarray:
    o = obs["observation"]
    grip_pos = o[0:3]
    object_pos = o[3:6]
    object_rel_pos = o[6:9]     # grip_pos - object_pos (see envs/arm_pick_place.py)
    gripper_state = o[9:11]     # finger positions; larger = more open
    goal = obs["desired_goal"]

    xy_err = np.linalg.norm(object_rel_pos[:2])
    z_err = object_rel_pos[2]

    if xy_err > XY_ALIGN_THRESHOLD:
        target = object_pos + np.array([0, 0, ABOVE_HEIGHT])
        delta = target - grip_pos
        gripper = 1.0  # open
    elif abs(z_err) > 0.01:
        target = object_pos
        delta = target - grip_pos
        gripper = 1.0
    elif gripper_state.sum() > GRIPPER_OPEN_THRESHOLD:
        delta = np.zeros(3)
        gripper = -1.0  # close
    else:
        delta = goal - grip_pos
        gripper = -1.0  # stay closed while carrying

    action = np.concatenate([np.clip(delta * PROPORTIONAL_GAIN, -1, 1), [gripper]])
    return action.astype(np.float32)


if __name__ == "__main__":
    # Success-rate check: python approaches/imitation_act/scripted_expert.py [n_episodes]
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from envs.arm_pick_place import make_env

    n_episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    env = make_env()
    successes = 0
    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        ep_success = False
        for _ in range(100):
            obs, reward, terminated, truncated, info = env.step(scripted_expert_action(obs))
            if info.get("is_success"):
                ep_success = True
        successes += int(ep_success)
    env.close()
    print(f"{successes}/{n_episodes} episodes succeeded ({100 * successes / n_episodes:.0f}%)")
