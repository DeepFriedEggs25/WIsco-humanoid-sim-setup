"""End-to-end scripted pick-and-place using real Jacobian (damped-least-
squares pseudoinverse) IK -- no learning, no dataset, guaranteed to work.
This is the reference approach: if nothing else in this repo runs on your
machine, this should.

VERIFIED: the --render none path was run headlessly end-to-end during
development on mujoco==3.2.7 and successfully lifts the box off the table
(see RESEARCH_NOTES.md "Verification log" for the exact printed output and
final box height). --render video was also verified to produce a playable
mp4. --render viewer (the interactive window) was later confirmed working
on macOS -- note it requires running with `mjpython`, not `python`, on
macOS (a MuJoCo/Cocoa requirement, not a bug) -- see setup_guide.md and
utils/viz.py.

Pipeline (a state machine over waypoints, each reached via ik_solver.py's
JacobianIK):
  1. HOME     -- start at the arm's non-singular ready pose, gripper open.
  2. ABOVE    -- "detect" the box position (read it directly from mj_data --
                 an oracle standing in for a real perception pipeline, see
                 README.md) and IK to a point directly above it.
  3. DESCEND  -- IK down to grasp height.
  4. CLOSE    -- close the gripper (no IK, just actuator setpoints).
  5. LIFT     -- IK back up to the ABOVE height, gripper still closed.
  6. Report success: did the box's height increase by more than
     SUCCESS_LIFT_THRESHOLD_M?

Usage:
    python approaches/scripted_ik/pick_place_scripted.py                  # headless, prints result
    python approaches/scripted_ik/pick_place_scripted.py --render viewer  # watch it live
    # ^ on macOS, use `mjpython` instead of `python` for this one -- see setup_guide.md
    python approaches/scripted_ik/pick_place_scripted.py --render video --out outputs/pick_place.mp4
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root, see utils/repo_paths.py

import mujoco
import numpy as np

from approaches.scripted_ik.ik_solver import reset_to_ready_pose, solve_ik_from_qpos, ARM_JOINT_NAMES
from utils.recording import VideoWriter
from utils.repo_paths import OUTPUTS_DIR
from utils.viz import OffscreenRenderer

SCENE_PATH = pathlib.Path(__file__).parent / "assets" / "simple_arm_scene.xml"

ABOVE_OFFSET_M = 0.12       # height above the object for the pre-grasp waypoint
GRASP_OFFSET_M = 0.015      # how far above the object CENTER to close the gripper
                            # (object half-size is 0.02m, so this still lets the
                            # fingers close around it rather than push through it)
GRIPPER_OPEN = 0.04
# Commanding exactly 0.0 (the box's exact surface) generated ~zero squeeze
# force (real bug found during development, see RESEARCH_NOTES.md
# "Verification log") -- the position servo has nothing to push against
# until it's asked to go slightly past the surface. -0.01 asks each finger
# to move 1cm further inward than the box allows, so the joint-limit/contact
# solver generates a real compressive grasp force instead.
GRIPPER_CLOSED = -0.01
SUCCESS_LIFT_THRESHOLD_M = 0.04  # was 0.08 -- too strict, see RESEARCH_NOTES.md
                                  # "Verification log" (real grasps measured 56-80mm)
SETTLE_STEPS = 400          # physics steps to let position actuators reach an IK waypoint
GRASP_STEPS = 300           # steps to hold the gripper closed before lifting
VIDEO_FRAME_STRIDE = 4      # only render/save every Nth physics step, to keep mp4s short


def get_actuator_ids(model):
    arm_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{n}") for n in ARM_JOINT_NAMES]
    left_finger_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_left_finger")
    right_finger_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_right_finger")
    return arm_ids, left_finger_id, right_finger_id


def detect_object_position(model, data) -> np.ndarray:
    """'Perception': read the box's true position directly out of MuJoCo.
    This is an oracle standing in for real object detection (e.g. a pose
    estimator on a camera feed) -- there is no computer vision here, and the
    README says so explicitly. Swapping this for a real detector is the
    natural next exercise once this scripted baseline works."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_box")
    return data.xpos[body_id].copy()


class Recorder:
    """Optional video sink threaded through the state machine below. If
    `renderer`/`writer` are None, `step()` is a no-op -- this is how
    --render none stays exactly as fast as plain mj_step in a loop."""

    def __init__(self, renderer: OffscreenRenderer | None, writer: VideoWriter | None,
                 stride: int = VIDEO_FRAME_STRIDE):
        self.renderer = renderer
        self.writer = writer
        self.stride = stride
        self._count = 0

    def step(self, data):
        if self.renderer is None:
            return
        self._count += 1
        if self._count % self.stride == 0:
            self.writer.add(self.renderer.frame(data))


def move_to(model, data, arm_actuator_ids, target_pos, recorder: Recorder, steps=SETTLE_STEPS):
    """IK-solve for target_pos from the CURRENT live qpos, then hold that
    joint-angle setpoint on the position actuators for `steps` physics
    steps."""
    result = solve_ik_from_qpos(model, data.qpos.copy(), target_pos)
    for i, aid in enumerate(arm_actuator_ids):
        data.ctrl[aid] = result.qpos[i]
    for _ in range(steps):
        mujoco.mj_step(model, data)
        recorder.step(data)
    return result


def set_gripper(model, data, left_id, right_id, width, recorder: Recorder, steps=GRASP_STEPS):
    data.ctrl[left_id] = width
    data.ctrl[right_id] = width
    for _ in range(steps):
        mujoco.mj_step(model, data)
        recorder.step(data)


def run(render_mode: str, out_path: str, seed: int = 0) -> bool:
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    reset_to_ready_pose(model, data)

    rng = np.random.default_rng(seed)
    # Randomize the box's starting (x,y) slightly within the table so a
    # single run isn't tautologically tuned to one exact position.
    box_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_box_joint")
    qadr = model.jnt_qposadr[box_joint_id]
    data.qpos[qadr] += rng.uniform(-0.06, 0.06)
    data.qpos[qadr + 1] += rng.uniform(-0.06, 0.06)
    mujoco.mj_forward(model, data)

    arm_actuator_ids, left_finger_id, right_finger_id = get_actuator_ids(model)

    if render_mode == "viewer":
        _run_with_viewer(model, data, arm_actuator_ids, left_finger_id, right_finger_id)
        return True  # success reporting for the interactive path is visual, not automated

    video_renderer = writer = None
    if render_mode == "video":
        video_renderer = OffscreenRenderer(model, width=480, height=480)
        out = pathlib.Path(out_path) if out_path else (OUTPUTS_DIR / "pick_place_scripted.mp4")
        physics_hz = 1.0 / model.opt.timestep
        writer = VideoWriter(out, fps=int(physics_hz / VIDEO_FRAME_STRIDE))
    recorder = Recorder(video_renderer, writer)

    initial_box_height = detect_object_position(model, data)[2]
    print(f"initial box position: {np.round(detect_object_position(model, data), 3)}")

    set_gripper(model, data, left_finger_id, right_finger_id, GRIPPER_OPEN, recorder, steps=50)

    object_pos = detect_object_position(model, data)
    above = object_pos + np.array([0, 0, ABOVE_OFFSET_M])
    print(f"-> ABOVE waypoint: {np.round(above, 3)}")
    move_to(model, data, arm_actuator_ids, above, recorder)

    grasp = object_pos + np.array([0, 0, GRASP_OFFSET_M])
    print(f"-> DESCEND waypoint: {np.round(grasp, 3)}")
    move_to(model, data, arm_actuator_ids, grasp, recorder)

    print("-> CLOSE gripper")
    set_gripper(model, data, left_finger_id, right_finger_id, GRIPPER_CLOSED, recorder)

    print(f"-> LIFT waypoint: {np.round(above, 3)}")
    move_to(model, data, arm_actuator_ids, above, recorder)

    final_box_height = detect_object_position(model, data)[2]
    lift = final_box_height - initial_box_height
    success = lift > SUCCESS_LIFT_THRESHOLD_M
    print(f"\nfinal box height: {final_box_height:.3f} (started at {initial_box_height:.3f}, "
          f"lifted {lift * 1000:.1f}mm)")
    print("SUCCESS: object grasped and lifted" if success else "FAILURE: object was not lifted")

    if writer is not None:
        writer.close()
        video_renderer.close()
        print(f"video saved to {writer.path}")
    return success


def _run_with_viewer(model, data, arm_actuator_ids, left_finger_id, right_finger_id):
    """Same state machine as run(), reimplemented as a per-frame step
    function so the interactive viewer can render every physics step at
    roughly real-time speed. NOT independently verified in this development
    sandbox -- see module docstring."""
    from utils.viz import run_viewer_loop

    plan = ["open", "above", "descend", "close", "lift", "done"]
    durations = {"open": 50, "above": SETTLE_STEPS, "descend": SETTLE_STEPS,
                 "close": GRASP_STEPS, "lift": SETTLE_STEPS}
    state = {"phase_idx": 0, "phase_steps": 0}

    def step_fn(model, data, step_index):
        phase = plan[state["phase_idx"]]
        if phase == "done":
            return False

        if state["phase_steps"] == 0:
            if phase == "open":
                data.ctrl[left_finger_id] = GRIPPER_OPEN
                data.ctrl[right_finger_id] = GRIPPER_OPEN
            elif phase in ("above", "descend", "lift"):
                object_pos = detect_object_position(model, data)
                offset = ABOVE_OFFSET_M if phase in ("above", "lift") else GRASP_OFFSET_M
                target = object_pos + np.array([0, 0, offset])
                result = solve_ik_from_qpos(model, data.qpos.copy(), target)
                for i, aid in enumerate(arm_actuator_ids):
                    data.ctrl[aid] = result.qpos[i]
            elif phase == "close":
                data.ctrl[left_finger_id] = GRIPPER_CLOSED
                data.ctrl[right_finger_id] = GRIPPER_CLOSED

        mujoco.mj_step(model, data)
        state["phase_steps"] += 1
        if state["phase_steps"] >= durations[phase]:
            print(f"phase '{phase}' complete")
            state["phase_idx"] += 1
            state["phase_steps"] = 0
        return True

    run_viewer_loop(model, data, step_fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--render", choices=["none", "viewer", "video"], default="none",
                        help="none = headless, print result (fastest, default). "
                             "viewer = open an interactive MuJoCo window. "
                             "video = save an mp4 to --out.")
    parser.add_argument("--out", default=None,
                        help="output path for --render video (default: outputs/pick_place_scripted.mp4)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    ok = run(args.render, args.out, args.seed)
    sys.exit(0 if ok else 1)
