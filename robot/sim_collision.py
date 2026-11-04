"""Sampled self-collision checks for opt-in simulation models.

Uses a separate MjModel/MjData so proximity margins never alter the live physics.
Checks robot self-contact only; world objects and intended finger-to-finger
closure are outside this guard. This is not a hardware safety controller or a
continuous collision proof. Call under the simulator's state lock.
"""

import copy

import mujoco
import numpy as np


class SelfCollisionGuard:
    def __init__(self, model, margin):
        if not np.isfinite(margin) or margin <= 0:
            raise ValueError("self_collision_margin_m must be positive and finite")
        self.margin = float(margin)
        self.model = copy.copy(model)
        self.model.geom_margin[:] = np.maximum(self.model.geom_margin, margin)
        # The mesh BVH midphase can cull positively separated pairs inside an
        # enlarged query margin (reproduced at the gripper/rail 2.9 mm gap).
        # Query all body-broadphase candidates; leave live physics unchanged.
        self.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE)
        self.data = mujoco.MjData(self.model)
        self.addresses = np.array(
            [
                model.jnt_qposadr[j]
                for j in range(model.njnt)
                if model.joint(j).name.startswith(
                    (
                        "left_arm_joint",
                        "right_arm_joint",
                        "left_gripper_finger",
                        "right_gripper_finger",
                        "head_",
                    )
                )
                or model.joint(j).name == "lift_joint"
            ],
            dtype=int,
        )
        self.resolution = np.array(
            [
                0.002 if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE else 0.02
                for j in range(model.njnt)
                if model.jnt_qposadr[j] in self.addresses
            ]
        )
        self.lift = model.jnt_qposadr[model.joint("lift_joint").id]
        self.middle = model.jnt_qposadr[model.joint("sophia_middle_slide").id]
        # A joint-angle grid alone can skip a thin obstacle at the end of a long
        # arm. Bound surface travel using descendant link lengths and geom
        # bounding radii; refine until each geom moves <= half the clearance.
        joints = [j for j in range(model.njnt) if model.jnt_qposadr[j] in self.addresses]
        self.motion_bound = np.zeros((model.ngeom, len(joints)))
        for g in range(model.ngeom):
            if not (model.geom_contype[g] or model.geom_conaffinity[g]):
                continue
            body = int(model.geom_bodyid[g])
            radius = float(np.linalg.norm(model.geom_pos[g]) + model.geom_rbound[g])
            ancestors = {}
            while body:
                ancestors[body] = radius
                radius += float(np.linalg.norm(model.body_pos[body]))
                for j in range(
                    model.body_jntadr[body], model.body_jntadr[body] + model.body_jntnum[body]
                ):
                    radius += 2 * float(np.linalg.norm(model.jnt_pos[j]))
                    if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE:
                        radius += float(np.max(abs(model.jnt_range[j])))
                body = int(model.body_parentid[body])
            for k, j in enumerate(joints):
                body = int(model.jnt_bodyid[j])
                if body in ancestors:
                    self.motion_bound[g, k] = (
                        1.0
                        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE
                        else ancestors[body] + np.linalg.norm(model.jnt_pos[j])
                    )
            if model.body("sophia_middle_stage").id in ancestors:
                self.motion_bound[g, list(self.addresses).index(self.lift)] = 0.5
        self.ignored = set()
        for side in ("left", "right"):
            self.ignored.add(frozenset(model.body(f"{side}_gripper_finger_{f}").id for f in "ab"))

    def _contacts(self, qpos):
        self.data.qpos[:] = qpos
        self.data.qpos[self.middle] = self.data.qpos[self.lift] / 2
        # Collision checking needs positions, not dynamics or constraint solving.
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        mujoco.mj_collision(self.model, self.data)
        contacts = {}
        for c in self.data.contact:
            geoms = tuple(sorted((int(c.geom1), int(c.geom2))))
            bodies = [int(self.model.geom_bodyid[g]) for g in geoms]
            if 0 in bodies or frozenset(bodies) in self.ignored or c.dist >= self.margin:
                continue
            contacts[geoms] = min(contacts.get(geoms, self.margin), float(c.dist))
        return contacts

    def _reason(self, pair, distance, progress):
        return {
            "reason": "self_collision",
            "geoms": [self.model.geom(g).name or f"geom_{g}" for g in pair],
            "bodies": [self.model.body(self.model.geom_bodyid[g]).name for g in pair],
            "clearance_m": distance,
            "required_clearance_m": self.margin,
            "path_fraction": float(progress),
        }

    def check(self, start, goal, speeds=None):
        """Check a segment or independently rate-limited joint trajectories.

        Sampling bounds each joint step to 20 mrad or 2 mm, including endpoints
        and every joint's arrival time. A state already inside the margin may
        retreat, but no existing pair may approach further and no new pair may
        enter the margin. Physical interpenetration cannot be an accepted goal.
        """
        start, goal = np.asarray(start), np.asarray(goal)
        if not np.isfinite(start).all() or not np.isfinite(goal).all():
            return {"reason": "nonfinite_configuration"}
        ids = self.addresses
        delta = goal[ids] - start[ids]
        if speeds is None:
            steps = max(
                1,
                int(np.ceil(np.max(abs(delta) / self.resolution))),
                int(np.ceil(2 * np.max(self.motion_bound @ abs(delta)) / self.margin)),
            )
            times = np.linspace(0, 1, steps + 1)
            duration = np.ones(len(ids))
        else:
            speeds = np.asarray(speeds)[ids]
            duration = np.divide(abs(delta), speeds, out=np.zeros(len(ids)), where=speeds > 0)
            if np.any((abs(delta) > 1e-9) & (speeds <= 0)):
                return {"reason": "invalid_trajectory_speed"}
            knots = np.unique(np.r_[0.0, duration])
            times = [0.0]
            for lo, hi in zip(knots, knots[1:], strict=False):
                active = duration > lo
                dt = np.min(self.resolution[active] / speeds[active])
                surface_speed = float(np.max(self.motion_bound @ np.where(active, speeds, 0.0)))
                if surface_speed > 0:
                    dt = min(dt, self.margin / (2 * surface_speed))
                times.extend(np.linspace(lo, hi, max(1, int(np.ceil((hi - lo) / dt))) + 1)[1:])
        allowances = self._contacts(start)
        total = max(float(times[-1]), 1e-9)
        for t in times[1:]:
            q = start.copy()
            fraction = np.divide(t, duration, out=np.ones(len(ids)), where=duration > 0)
            q[ids] += delta * np.minimum(fraction, 1.0)
            contacts = self._contacts(q)
            for pair, distance in contacts.items():
                if pair not in allowances or distance < allowances[pair] - 2e-5:
                    return self._reason(pair, distance, t / total)
            # Allow only monotonically increasing separation when retreating.
            allowances = {
                pair: max(distance, allowances[pair]) for pair, distance in contacts.items()
            }
        final = self._contacts(goal)
        for pair, distance in final.items():
            if distance < -1e-4:
                return self._reason(pair, distance, 1.0)
        return None
