# Scripted IK pick-and-place

**Status: fully verified working.** This is the reference approach — if you
only try one thing in this repo, try this one. It needs no dataset, no
training, no GPU, and takes a few seconds to run.

## What this is

A classical (non-learning) robotics approach: a small state machine drives a
5-DOF simulated arm through pick-and-place waypoints (approach, descend,
grasp, lift), where each waypoint is reached by solving **inverse
kinematics** — given a target 3D position for the gripper, compute the joint
angles that get it there. `ik_solver.py` implements this with the **damped
least-squares Jacobian pseudoinverse** method, using `mujoco.mj_jacSite` to
get the Jacobian directly from the physics engine at each iteration.

"Perception" here is an oracle: `detect_object_position()` reads the box's
true position directly out of MuJoCo, standing in for a real object
detector. This keeps the demo self-contained and focused on the IK/control
problem — swapping in a real detector (e.g. a pose estimator over a
rendered/real camera image) is the natural next exercise once you understand
this baseline.

## Run it

```bash
# headless, prints the result (fastest, no rendering overhead)
python approaches/scripted_ik/pick_place_scripted.py

# watch it live in an interactive MuJoCo window
python approaches/scripted_ik/pick_place_scripted.py --render viewer

# save a video instead
python approaches/scripted_ik/pick_place_scripted.py --render video --out outputs/pick_place.mp4

# just the IK solver's own convergence self-test (no arm/gripper control)
python approaches/scripted_ik/ik_solver.py
```

## Expected output (headless)

```
initial box position: [-0.5    0.1    0.43 ]
-> ABOVE waypoint: [-0.5    0.1    0.55 ]
-> DESCEND waypoint: [-0.5    0.1    0.445]
-> CLOSE gripper
-> LIFT waypoint: [-0.5    0.1    0.55 ]

final box height: 0.51 (started at 0.430, lifted 76.4mm)
SUCCESS: object grasped and lifted
```

**Verified during development:** 10/10 successful grasps across randomized
box start positions (seeds 0-9), typical lift height 55-80mm. See
`RESEARCH_NOTES.md`'s "Verification log" for the exact debugging history —
three real bugs were found and fixed while getting to 10/10 (a Jacobian API
mixup, a missing `mj_comPos` call, a kinematic singularity in the home pose,
insufficient actuator gains causing gravity droop, a mis-placed
end-effector site, and a zero-margin gripper-close target). None of that is
hidden — it's exactly the kind of debugging you'll do yourself extending
this, so it's documented rather than smoothed over.

## Files

- `assets/simple_arm_scene.xml` — a from-scratch 5-DOF arm + 2-finger
  gripper + table + box, all primitive geoms (no mesh downloads). See the
  file's own header comment for the joint layout.
- `ik_solver.py` — `JacobianIK`, the damped-least-squares solver, plus a
  standalone convergence self-test.
- `pick_place_scripted.py` — the state machine that calls the solver to
  execute a full pick-and-place.

## Learning more

- MuJoCo's own Jacobian/IK lecture notes and worked example (the same
  method used here): https://pab47.github.io/mujoco/notes/Lec6_double_pendulum_ik.pdf
- Samuel Buss, *"Introduction to Inverse Kinematics with Jacobian Transpose,
  Pseudoinverse and Damped Least Squares methods"* — the standard reference
  for why damped least squares beats plain pseudoinverse near singularities:
  https://www.math.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf
- MuJoCo's own Jacobian API docs: https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html
  (search `mj_jacSite`)
