"""Isolate stereo OpenGL/GPU stalls from the live control interpreter."""

import contextlib
import multiprocessing as mp
import queue
import threading
import time
from types import SimpleNamespace

import mujoco
import numpy as np

STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION


def _render_worker(model_path, states, stopped, ready, options):
    from robot.yor_mujoco import StereoPublisher

    video = None
    try:
        model = mujoco.MjModel.from_xml_path(model_path)
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, 1920)
        model.vis.global_.offheight = max(model.vis.global_.offheight, 1080)
        data = mujoco.MjData(model)
        proxy = SimpleNamespace(model=model, data=data, _lock=threading.RLock())
        state = states.get(timeout=15)
        mujoco.mj_setState(model, data, state, STATE_SPEC)
        mujoco.mj_forward(model, data)
        video = StereoPublisher(proxy, **options)
        video.start()
        if not video.ready.wait(15):
            raise TimeoutError("Stereo renderer did not publish its first frame")
        ready.send({"ready": True})
        while not stopped.is_set():
            try:
                state = states.get(timeout=0.1)
            except queue.Empty:
                continue
            with proxy._lock:
                mujoco.mj_setState(model, data, state, STATE_SPEC)
                mujoco.mj_forward(model, data)
    except Exception as exc:
        ready.send({"error": str(exc)})
        raise
    finally:
        if video:
            video.stop()
        ready.close()


class ProcessStereoPublisher:
    """Latest-state queue only; the robot/control process never renders a frame."""

    def __init__(self, sim, model_path, **options):
        self.sim = sim
        self.period = 1 / options.get("fps", 30)
        ctx = mp.get_context("spawn")
        self.states = ctx.Queue(maxsize=1)
        self.stopped = ctx.Event()
        self.ready, child_ready = ctx.Pipe(duplex=False)
        self.child_ready = child_ready
        self.process = ctx.Process(
            target=_render_worker,
            args=(str(model_path), self.states, self.stopped, child_ready, options),
            daemon=True,
        )
        self.feeder = threading.Thread(target=self._feed, daemon=True)

    def _feed(self):
        model = self.sim.model
        while not self.stopped.is_set():
            started = time.monotonic()
            state = np.empty(mujoco.mj_stateSize(model, STATE_SPEC))
            with self.sim._lock:
                mujoco.mj_getState(model, self.sim.data, state, STATE_SPEC)
            if self.stopped.is_set():
                return
            try:
                self.states.put_nowait(state)
            except queue.Full:
                with contextlib.suppress(queue.Empty):
                    self.states.get_nowait()
                with contextlib.suppress(queue.Full):  # never wait on a slow renderer
                    self.states.put_nowait(state)
            self.stopped.wait(max(0, self.period - (time.monotonic() - started)))

    def start(self):
        self.process.start()
        self.child_ready.close()
        self.feeder.start()
        if not self.ready.poll(35):
            self.stop()
            raise TimeoutError("Stereo render process startup timed out")
        status = self.ready.recv()
        if not status.get("ready"):
            self.stop()
            raise RuntimeError("Stereo renderer: " + str(status.get("error")))

    def stop(self):
        self.stopped.set()
        if self.feeder.is_alive():
            self.feeder.join(timeout=1)
        if self.process.pid is not None:
            self.process.join(timeout=2)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=1)
                if self.process.is_alive():
                    self.process.kill()
                    self.process.join(timeout=1)
        self.states.cancel_join_thread()
        self.states.close()
        self.ready.close()
