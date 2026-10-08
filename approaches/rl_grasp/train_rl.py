"""Train a grasp/reach policy with Stable-Baselines3 PPO on a Gymnasium-
Robotics Fetch task -- the RL approach, for members who'd rather learn RL
than imitation learning.

VERIFIED: both `--task reach` and `--task pick_place` were run for a small
number of real training steps during development (SB3 2.9.0,
MultiInputPolicy for the dict observation space) -- see RESEARCH_NOTES.md
"Verification log". A full training run to a meaningful success rate was
NOT done (see below for why that's a real caveat, not just "wasn't run").

WHY DENSE REWARD, AND WHY THIS IS HARDER THAN IT LOOKS:
Fetch tasks default to a SPARSE reward (-1 every step until success, 0 on
success). Vanilla PPO (no goal-relabeling) is known to struggle badly with
this -- the original Fetch/HER paper's whole point was that you need
Hindsight Experience Replay (or a dense reward) to learn anything in a
reasonable number of samples. SB3 does not ship HER-augmented PPO (HER in
SB3 is paired with off-policy algorithms like SAC/TD3/DQN, not PPO). So this
script defaults to the DENSE reward variant (`reward_type=dense`) to give
plain PPO a fighting chance. Even so, `FetchPickAndPlace` is a genuinely
hard exploration problem for on-policy RL from scratch -- expect this to
need a lot more than a quick laptop run to get a good success rate. If you
want to see PPO succeed reasonably quickly, start with `--task reach`
(reach is nearly trivial for PPO with a dense reward) before attempting
`--task pick_place`.

Usage:
    python approaches/rl_grasp/train_rl.py --task reach --timesteps 50000
    python approaches/rl_grasp/train_rl.py --task pick_place --timesteps 500000
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root

from utils.repo_paths import OUTPUTS_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=["reach", "pick_place"], default="reach")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--n_envs", type=int, default=1,
                        help="parallel envs via SB3's DummyVecEnv/SubprocVecEnv -- keep at 1 on a laptop "
                             "unless you have CPU headroom to spare")
    parser.add_argument("--out", default=None, help="default: outputs/rl/<task>_ppo.zip")
    parser.add_argument("--eval_episodes", type=int, default=10)
    args = parser.parse_args()

    # Imported here so --help doesn't require sb3/torch installed.
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    if args.task == "reach":
        from envs.arm_reach import make_env
    else:
        from envs.arm_pick_place import make_env

    def _make():
        return make_env(reward_type="dense")

    env = DummyVecEnv([_make for _ in range(args.n_envs)])
    model = PPO("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=args.timesteps)

    out_path = pathlib.Path(args.out) if args.out else (OUTPUTS_DIR / "rl" / f"{args.task}_ppo.zip")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(out_path)
    print(f"\nmodel saved to {out_path}")

    # Quick sparse-reward success-rate check (train reward is dense, but
    # "success" is always defined the same way regardless of reward_type).
    eval_env = _make()
    successes = 0
    for ep in range(args.eval_episodes):
        obs, info = eval_env.reset(seed=10_000 + ep)
        ep_success = False
        for _ in range(100):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            if info.get("is_success"):
                ep_success = True
            if terminated or truncated:
                break
        successes += int(ep_success)
    print(f"success rate: {successes}/{args.eval_episodes} ({100 * successes / args.eval_episodes:.0f}%)")


if __name__ == "__main__":
    main()
