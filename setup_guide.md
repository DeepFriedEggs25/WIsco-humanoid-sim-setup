# Setup guide

Goal: clone this repo and see the arm move in simulation within an hour, on
whatever laptop you have. This guide covers Windows (via WSL2), macOS, and
Linux.

## 0. Python version

**Use Python 3.12.** `gymnasium-robotics` would tolerate 3.10+, but
`lerobot` (needed for the `imitation_act`/`imitation_diffusion` approaches)
requires `>=3.12` — starting on 3.12 from the beginning avoids needing a
second virtual environment later if you decide to try those approaches
after starting with `scripted_ik`.

Check what you have:
```bash
python3 --version
```

If you don't have 3.12:
- **macOS**: `brew install python@3.12`
- **Ubuntu/Debian (incl. WSL2)**: `sudo apt install python3.12 python3.12-venv`
- **Windows**: install Python 3.12 from python.org **inside WSL2**, not
  native Windows Python — see the Windows section below for why.

## 1. Windows

**Use WSL2 (Windows Subsystem for Linux), not native Windows Python.**
MuJoCo's rendering (even offscreen) and some of the libraries here (video
encoding, OpenGL contexts) are far better supported and tested on Linux;
running natively on Windows is possible but you will hit more friction than
this guide accounts for.

1. Install WSL2 if you don't have it (PowerShell, as Administrator):
   ```powershell
   wsl --install -d Ubuntu
   ```
   Reboot if prompted, then open the "Ubuntu" app from the Start menu.
2. Inside the WSL2 Ubuntu terminal, follow the **Linux** instructions below
   exactly.
3. For the interactive MuJoCo viewer (`--render viewer` in
   `scripted_ik`) or `--render human`, you need an X server on the Windows
   side. Recent WSL2 (Windows 11, or updated Windows 10) has this built in
   via WSLg — a MuJoCo viewer window should just open. If it doesn't, this
   is the one thing in this guide that may need extra troubleshooting
   specific to your Windows version; the headless (`--render none`) and
   video (`--render video`) paths work regardless and are what's actually
   verified in this repo (see `RESEARCH_NOTES.md`).

## 2. macOS

```bash
brew install python@3.12
# optional but recommended: silences a harmless torchcodec warning later
brew install ffmpeg

git clone <this-repo-url> manipulation_sim
cd manipulation_sim
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Then jump to **Verify your setup** below.

Apple Silicon (M1/M2/M3/...) note: everything in `requirements.txt` has
prebuilt `arm64` wheels — no compilation needed, verified during
development on Apple Silicon.

## 3. Linux (native, or inside WSL2)

```bash
sudo apt update
sudo apt install python3.12 python3.12-venv git ffmpeg

git clone <this-repo-url> manipulation_sim
cd manipulation_sim
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you're on a headless server/container with no GPU and no display at
all, offscreen rendering (`--render video`, and all training scripts) still
works — MuJoCo's software renderer doesn't need a GPU or an X server for
this. Only `--render viewer` / `--render human` need a display.

## Verify your setup

Run these in order — each should take well under a minute:

```bash
# 1. Does MuJoCo + Gymnasium work at all?
python envs/arm_reach.py
# expect: prints observation/action shapes, "OK"

# 2. Does the fully-verified reference approach work?
python approaches/scripted_ik/ik_solver.py
# expect: a small table, "ALL TARGETS CONVERGED"

python approaches/scripted_ik/pick_place_scripted.py
# expect: "SUCCESS: object grasped and lifted"
```

If both of those pass, your base environment is good — move on to
`approaches/README.md` to pick which approach to actually learn.

## Installing an approach's extra dependencies

Only install these once you know which approach you're doing —
they're not needed for `scripted_ik`:

```bash
# imitation_act / imitation_diffusion (heavy: pulls PyTorch)
pip install "lerobot[training]"     # for imitation_act
pip install "lerobot[diffusion]"    # additionally, for imitation_diffusion

# rl_grasp
pip install stable-baselines3
```

## Troubleshooting

**`gymnasium.error.DeprecatedEnv: Environment version v3 for 'FetchReach' is
deprecated`** — you're using an old tutorial's environment ID. Every env ID
used in this repo (and that you should use) ends in `-v4`. See
`RESEARCH_NOTES.md` section 2.

**A long traceback ending in `'torchcodec' is installed but cannot be
loaded ... Falling back to 'pyav' as a default decoder.`** — harmless, see
`RESEARCH_NOTES.md` section 3. `brew install ffmpeg` (or the apt equivalent)
silences it but isn't required.

**`ImportError: 'accelerate' is required...` or `'diffusers' is
required...`** when running `lerobot-train` — install the matching extra,
see "Installing an approach's extra dependencies" above. The error message
itself tells you the exact `pip install` command.

**MuJoCo viewer window doesn't open (`--render viewer`)** — this needs an
actual display/window server. Works out of the box on macOS and Linux
desktops and on Windows 11 WSL2 (WSLg); on older WSL2/Windows 10 you may
need to set up an X server (e.g. VcXsrv) manually. Use `--render video` or
`--render none` in the meantime; they don't need a display.

**On macOS specifically**, `--render viewer` fails immediately with
`RuntimeError: launch_passive requires that the Python script be run under
mjpython on macOS` — this is a real, confirmed requirement (Cocoa's UI
threading model), not optional. Run the script with `mjpython` instead of
`python`:
```bash
mjpython approaches/scripted_ik/pick_place_scripted.py --render viewer
```
`mjpython` is installed automatically into your venv by the `mujoco` pip
package (`venv/bin/mjpython`) — no separate install needed. This only
applies to the interactive viewer; `--render video` and `--render none`
run with plain `python` on every platform.

**Everything is slow / using 100% CPU during training** — expected on a
laptop with no GPU. `imitation_act`/`imitation_diffusion` training and
`rl_grasp` are all real machine learning; a few thousand steps is enough to
confirm the pipeline works, real results take real compute time regardless
of hardware. See each approach's README for what was and wasn't verified at
what scale.
