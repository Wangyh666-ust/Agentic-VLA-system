"""Collision-aware grasp controller with three frozen waypoints.

Reuses local_grasp base action generation and post-sample gates. Adds a
collision_geometry descriptor (authored separately) and a three-waypoint
ABOVE route: escape (radial out + high z), align_clear (above old XY),
approach_open (above target XY).
"""

import math

import numpy as np

import local_grasp as base
import preparation_diagnostics as pd
import side_grasp as side
import collision_geometry


WINE_OBJECT_ID = 'wine_bottle_1'
ABOVE = 'above'
DESCEND = 'descend'
PHASES = (base.IDLE, ABOVE, DESCEND, base.CLOSE, base.LIFT, base.CONFIRMED, base.BYPASS, base.FAILED)


def read_geometry(env):
    reading = side.read_geometry(env)
    reading['collision_geometry'] = collision_geometry.describe(env, reading['pose']['position'])
    return reading


def _finite_scalar(value):
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        value = float(value)
        if math.isfinite(value) and value >= 0.0:
            return value
    return None


class CollisionGraspController(side.SideGraspController):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.route_targets = []
        self.route_index = 0
        self.route_counts = [0, 0, 0]
        self.high_z = None
        self._above_target_position = None
        self._above_target_orientation = None

    def _validate_descriptor(self, reading):
        if not isinstance(reading, dict):
            return None
        descriptor = reading.get('collision_geometry')
        if not isinstance(descriptor, dict):
            return None
        top = _finite_scalar(descriptor.get('bottle_top_z_m'))
        sweep = _finite_scalar(descriptor.get('hand_sweep_radius_m'))
        contacts = descriptor.get('wine_robot_contacts')
        if top is None or sweep is None or not isinstance(contacts, list):
            return None
        pose = reading.get('pose')
        if not isinstance(pose, dict):
            return None
        current_position = base._finite_vec3(pose.get('position'))
        if current_position is None:
            return None
        current_position = np.asarray(current_position, dtype=np.float64)
        orientation = base._finite_matrix3(pose.get('orientation_matrix'))
        if orientation is None:
            return None
        wine_xyz = base._snapshot_object_position(reading.get('snapshot'), base.WINE_OBJECT_ID)
        if wine_xyz is None:
            return None
        wine_xyz = np.asarray(wine_xyz, dtype=np.float64)
        radial = current_position[:2] - wine_xyz[:2]
        norm = float(np.linalg.norm(radial))
        if not math.isfinite(norm) or norm <= 1e-6:
            return None
        radial_unit = radial / norm
        return top, sweep, current_position, orientation, wine_xyz, radial_unit

    def _freeze_route(self, reading):
        validated = self._validate_descriptor(reading)
        if validated is None:
            return None
        top, sweep, current_position, orientation, wine_xyz, radial_unit = validated
        old_above = self._above_target_position
        old_above_orientation = self._above_target_orientation
        if old_above is None or old_above_orientation is None:
            return None
        old_above = np.asarray(old_above, dtype=np.float64)
        old_above_orientation = np.asarray(old_above_orientation, dtype=np.float64).reshape(3, 3)
        current_z = float(current_position[2])
        old_above_z = float(old_above[2])
        self.high_z = max(current_z + 0.10, old_above_z, top + sweep + 0.015)
        high_z = self.high_z

        escape_xy = current_position[:2] + 0.05 * radial_unit
        escape = np.array([escape_xy[0], escape_xy[1], high_z], dtype=np.float64)
        escape_orientation = orientation.copy()

        align_clear = np.array([old_above[0], old_above[1], high_z], dtype=np.float64)
        align_orientation = self._target_orientation.copy()

        approach_open = old_above.copy()
        approach_orientation = self._target_orientation.copy()

        self.route_targets = [
            (escape, escape_orientation),
            (align_clear, align_orientation),
            (approach_open, approach_orientation),
        ]
        self.route_index = 0
        return True

    def _stage_action(self, reading):
        if self.phase != ABOVE:
            return super()._stage_action(reading)
        if not self.route_targets:
            if self._freeze_route(reading) is None:
                return self._fail('unknown_collision_geometry')
        route_target = self.route_targets[self.route_index]
        self._above_target_position = route_target[0].copy()
        self._above_target_orientation = route_target[1].copy()
        return super()._stage_action(reading)

    def _observe_alignment(self, reading, stage, next_phase):
        super()._observe_alignment(reading, stage, next_phase)
        if stage == ABOVE and self.phase == DESCEND and self.route_index < 2:
            self.route_index += 1
            self.phase = ABOVE
            self._stage_streak[ABOVE] = 0

    def observe_after(self, reading):
        if self._outstanding_phase == ABOVE and not self.route_targets:
            base_worked = super().observe_after(reading)
            return base_worked
        if self._outstanding_phase == ABOVE:
            index = self.route_index
            if 0 <= index < len(self.route_counts):
                self.route_counts[index] += 1
        result = super().observe_after(reading)
        return result

    def next_action(self, reading, proposed_action=None):
        if not self._armed or self.phase != ABOVE:
            return super().next_action(reading, proposed_action)
        failure = self._active_failure(reading)
        if failure is not None:
            return self._fail(failure)
        if self.total_actions >= 200:
            return self._fail('total_budget_exceeded')
        if self.route_counts[self.route_index] >= 50:
            return self._fail('route_timeout')
        if sum(self.route_counts) >= 130:
            return self._fail('approach_budget_exceeded')
        return self._stage_action(reading)

    def _active_failure(self, reading):
        failure = super()._active_failure(reading)
        if failure is not None:
            return failure
        if self.phase != ABOVE:
            return None
        validated = self._validate_descriptor(reading)
        if validated is None:
            return 'unknown_collision_geometry'
        top, sweep, current_position, orientation, wine_xyz, radial_unit = validated
        descriptor = reading.get('collision_geometry')
        contacts = descriptor.get('wine_robot_contacts')
        if self.route_index >= 1 and isinstance(contacts, list) and len(contacts) > 0:
            return 'approach_collision'
        if self.route_index == 1:
            z = float(current_position[2])
            if z - sweep < top + 0.005:
                return 'rotation_clearance_lost'
        return None

    def summary(self):
        data = super().summary()
        if not isinstance(data, dict):
            return data
        data['approach_mode'] = 'clear_then_side'
        data['route_index'] = self.route_index
        route_stage_names = ('escape', 'align_clear', 'approach_open')
        if 0 <= self.route_index < len(route_stage_names):
            data['route_stage'] = route_stage_names[self.route_index]
        else:
            data['route_stage'] = None
        data['route_counts'] = list(self.route_counts)
        data['route_targets'] = [
            {'position': np.asarray(position, dtype=np.float64).tolist(),
             'orientation': np.asarray(orientation, dtype=np.float64).reshape(3, 3).tolist()}
            for position, orientation in self.route_targets
        ]
        data['high_z'] = self.high_z
        return data


LocalGraspController = CollisionGraspController


def __getattr__(name):
    return getattr(base, name)
