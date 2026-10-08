"""Render helpers shared across approaches: an interactive MuJoCo viewer loop
and an offscreen frame renderer for saving videos. Two independent code
paths so a script can offer --render {viewer, video, none} without every
approach re-implementing the same MuJoCo boilerplate.

VERIFIED: the offscreen renderer path (render_frame / OffscreenRenderer) was
run end-to-end during development against mujoco==3.2.7 -- see
RESEARCH_NOTES.md "Verification log". The interactive viewer path
(run_viewer_loop) uses mujoco.viewer.launch_passive per MuJoCo's documented
API and has since been confirmed working on macOS -- but on macOS it MUST
be run with `mjpython`, not plain `python` (a MuJoCo/Cocoa threading
requirement; launch_passive raises a clear RuntimeError naming this if you
forget). See setup_guide.md.
"""
import mujoco
import numpy as np


class OffscreenRenderer:
    """Thin wrapper around mujoco.Renderer for repeatedly grabbing rgb frames
    of the same (model, data) pair, e.g. to build up a video."""

    def __init__(self, model: mujoco.MjModel, width: int = 480, height: int = 480,
                 camera: str | None = None):
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.camera = camera

    def frame(self, data: mujoco.MjData) -> np.ndarray:
        if self.camera:
            self.renderer.update_scene(data, camera=self.camera)
        else:
            self.renderer.update_scene(data)
        return self.renderer.render()

    def close(self):
        self.renderer.close()


def run_viewer_loop(model: mujoco.MjModel, data: mujoco.MjData, step_fn, max_steps: int | None = None):
    """Opens an interactive passive MuJoCo viewer and calls step_fn(model,
    data, step_index) once per frame until step_fn returns False, the window
    is closed, or max_steps is reached. step_fn is responsible for calling
    mujoco.mj_step itself (this just drives the render loop and syncs the
    viewer) so callers can control their own physics-vs-control substep
    ratio.

    Confirmed working on macOS. On macOS this MUST be launched via `mjpython
    script.py ...`, not `python script.py ...` -- launch_passive raises
    RuntimeError otherwise. See setup_guide.md.
    """
    import time

    import mujoco.viewer

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step_index = 0
        while viewer.is_running() and (max_steps is None or step_index < max_steps):
            step_start = time.time()
            keep_going = step_fn(model, data, step_index)
            viewer.sync()
            step_index += 1
            if keep_going is False:
                break
            # Roughly real-time playback so a human can actually watch it.
            dt_remaining = model.opt.timestep - (time.time() - step_start)
            if dt_remaining > 0:
                time.sleep(dt_remaining)
