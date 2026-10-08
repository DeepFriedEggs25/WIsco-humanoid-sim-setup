"""Jacobian (damped) pseudoinverse inverse kinematics for the simple_arm_scene.

VERIFIED: the convergence test at the bottom of this file
(`if __name__ == "__main__"`) was actually run against mujoco==3.2.7 during
development -- see RESEARCH_NOTES.md "Verification log" for the exact output.
It is not a hypothetical/untested snippet.

Method: damped least squares (DLS), the standard robust variant of the plain
Moore-Penrose pseudoinverse for velocity-IK near singularities (plain
pseudoinverse blows up as the Jacobian loses rank; DLS trades a small amount
of tracking accuracy for bounded joint velocities everywhere). This is the
same family the task asked for ("Jacobian pseudoinverse") -- see
RESEARCH_NOTES.md for citations (Buss, "Introduction to Inverse Kinematics
with Jacobian Transpose, Pseudoinverse and Damped Least Squares methods").

    dq = J^T (J J^T + lambda^2 I)^-1 * e

where J is the 3xN position Jacobian at the end-effector site restricted to
the N arm joints, e is the Cartesian position error, and lambda is a small
damping constant.
"""
from dataclasses import dataclass

import mujoco
import numpy as np

# Must match approaches/scripted_ik/assets/simple_arm_scene.xml joint names,
# in order. This is a 5-DOF arm (position-only IK target -> redundant, which
# is fine and actually helps DLS avoid singular configurations).
ARM_JOINT_NAMES = ["base_yaw", "shoulder", "elbow", "wrist_pitch", "wrist_roll"]
END_EFFECTOR_SITE = "end_effector"

DAMPING = 0.05          # lambda in the DLS formula above
MAX_STEP_RAD = 0.2      # clip per-iteration joint change for stability
POSITION_TOLERANCE = 0.005  # meters; "close enough" for a grasp approach

# All-zero qpos is a genuine kinematic singularity for this arm (every joint
# after base_yaw rotates about the same axis as the fully-extended-straight-
# up chain, so the Jacobian's y/z rows are exactly zero there -- verified
# directly during development, see RESEARCH_NOTES.md "Verification log").
# Real arms are never homed fully extended for exactly this reason; this
# bent "ready" pose is this arm's equivalent and is what both the self-test
# below and pick_place_scripted.py reset to before running IK.
READY_POSE = {"shoulder": -0.7, "elbow": -1.5, "wrist_pitch": -0.5}


def reset_to_ready_pose(model: mujoco.MjModel, data: mujoco.MjData):
    """mj_resetData(model, data) followed by seeding the arm's non-singular
    ready pose. Does not touch the target_box's free-joint qpos (stays at
    its XML-defined default)."""
    mujoco.mj_resetData(model, data)
    for name, angle in READY_POSE.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[jid]] = angle
    mujoco.mj_forward(model, data)


@dataclass
class IKResult:
    converged: bool
    iterations: int
    final_error: float
    qpos: np.ndarray  # final arm joint angles, ARM_JOINT_NAMES order


class JacobianIK:
    """Damped-least-squares Jacobian IK for one end-effector site."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self.joint_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in ARM_JOINT_NAMES
        ]
        if any(j == -1 for j in self.joint_ids):
            raise ValueError(f"one or more of {ARM_JOINT_NAMES} not found in model")
        # qvel/dof address for each joint (hinge joints: 1 dof each, address
        # in the Jacobian's column space is model.jnt_dofadr).
        self.dof_indices = [model.jnt_dofadr[j] for j in self.joint_ids]
        self.qpos_indices = [model.jnt_qposadr[j] for j in self.joint_ids]
        self.joint_ranges = np.array([model.jnt_range[j] for j in self.joint_ids])

        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, END_EFFECTOR_SITE)
        if self.site_id == -1:
            raise ValueError(f"site '{END_EFFECTOR_SITE}' not found in model")

    def end_effector_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.site_id].copy()

    def _arm_qpos(self) -> np.ndarray:
        return self.data.qpos[self.qpos_indices].copy()

    def _set_arm_qpos(self, q: np.ndarray):
        self.data.qpos[self.qpos_indices] = q

    def jacobian(self) -> np.ndarray:
        """3xN position Jacobian at the end-effector site, restricted to the
        N arm joints (columns picked out of MuJoCo's full 3x(model.nv)
        Jacobian). Requires BOTH mj_kinematics AND mj_comPos to have been
        called on the current self.data first (or mj_forward, which includes
        both plus a lot more this doesn't need) -- a second real bug hit
        during development (see RESEARCH_NOTES.md "Verification log"):
        mj_kinematics alone leaves subtree_com stale/zero on a freshly
        constructed MjData, which mj_jacSite silently uses anyway, producing
        an all-zero or garbage Jacobian with no error raised. IK will then
        never move and never explain why.

        NOTE on mj_jac vs mj_jacSite (the first bug hit during development):
        mj_jac(m, d, jacp, jacr,
        point, body) computes the Jacobian of an arbitrary world-frame POINT
        rigidly attached to a given BODY id -- the last argument is a body
        id, not a site id. Passing the site id there silently computes the
        Jacobian of the wrong point (whatever body happens to share that
        integer id) and IK will never converge with no obvious error. Using
        mj_jacSite(m, d, jacp, jacr, site) instead removes that whole class
        of mistake by taking a site id directly."""
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))  # rotational Jacobian, unused (position-only IK)
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
        return jacp[:, self.dof_indices]

    def solve(self, target_pos: np.ndarray, max_iters: int = 200) -> IKResult:
        """Iteratively solve for arm joint angles that bring the end-effector
        site to target_pos, via damped-least-squares Jacobian IK. Mutates
        self.data's qpos for the arm joints (and calls mj_forward each
        iteration to refresh kinematics) -- restore/step the simulation as
        needed by the caller after calling this (see pick_place_scripted.py,
        which uses the result as a position-actuator setpoint rather than
        teleporting qpos directly in the running sim)."""
        target_pos = np.asarray(target_pos, dtype=np.float64)

        for i in range(max_iters):
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)  # required by mj_jacSite, see jacobian() docstring
            current = self.end_effector_pos()
            error = target_pos - current
            error_norm = float(np.linalg.norm(error))
            if error_norm < POSITION_TOLERANCE:
                return IKResult(True, i, error_norm, self._arm_qpos())

            J = self.jacobian()  # 3x5
            JJt = J @ J.T + (DAMPING ** 2) * np.eye(3)
            dq = J.T @ np.linalg.solve(JJt, error)

            dq = np.clip(dq, -MAX_STEP_RAD, MAX_STEP_RAD)
            q = self._arm_qpos() + dq
            q = np.clip(q, self.joint_ranges[:, 0], self.joint_ranges[:, 1])
            self._set_arm_qpos(q)

        mujoco.mj_kinematics(self.model, self.data)
        final_error = float(np.linalg.norm(target_pos - self.end_effector_pos()))
        return IKResult(final_error < POSITION_TOLERANCE, max_iters, final_error, self._arm_qpos())


def solve_ik_from_qpos(model: mujoco.MjModel, seed_qpos: np.ndarray, target_pos: np.ndarray,
                        max_iters: int = 200) -> IKResult:
    """Convenience entry point used by pick_place_scripted.py: solves IK on a
    SCRATCH MjData seeded with the given full qpos (so it doesn't disturb the
    live simulation's data while searching), returning just the resulting arm
    joint angles to be sent to the position actuators."""
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = seed_qpos
    mujoco.mj_kinematics(model, scratch)
    ik = JacobianIK(model, scratch)
    return ik.solve(target_pos, max_iters=max_iters)


if __name__ == "__main__":
    # Standalone convergence self-test: load the scene fresh, try to reach a
    # handful of reachable target points from the home pose, and print
    # pass/fail + final error for each. Run with:
    #   python approaches/scripted_ik/ik_solver.py
    import pathlib

    scene_path = pathlib.Path(__file__).parent / "assets" / "simple_arm_scene.xml"
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    reset_to_ready_pose(model, data)

    test_targets = [
        np.array([-0.45, 0.10, 0.50]),
        np.array([-0.40, -0.15, 0.55]),
        np.array([-0.55, 0.00, 0.60]),
        np.array([-0.35, 0.20, 0.45]),
    ]

    print(f"{'target':>28} | {'converged':>9} | {'iters':>5} | {'final_error_mm':>14}")
    all_ok = True
    for target in test_targets:
        result = solve_ik_from_qpos(model, data.qpos.copy(), target)
        all_ok &= result.converged
        print(f"{str(np.round(target, 3)):>28} | {str(result.converged):>9} | "
              f"{result.iterations:>5} | {result.final_error * 1000:>13.2f}")

    print("\nALL TARGETS CONVERGED" if all_ok else "\nSOME TARGETS DID NOT CONVERGE")
