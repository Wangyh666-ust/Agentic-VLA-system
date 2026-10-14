"""Same-state observation reuse for simulator checking only."""
import copy
from time import perf_counter
import numpy as np
from scene_demo.xvla_backend import NativeXvlaEnv


class CachedNativeXvlaEnv(NativeXvlaEnv):
    def __init__(self, *args, **kwargs):
        self._observation_cache = None
        self._observation_fingerprint = None
        self.observation_cache_hits = 0
        self.observation_cache_misses = 0
        self.native_step_seconds = 0.0
        self.native_step_count = 0
        super().__init__(*args, **kwargs)

    def _state_fingerprint(self):
        if self._env is None:
            return None
        sim = self._env.env.sim
        return (id(sim), float(sim.data.time),
                *(np.asarray(getattr(sim.data, name)).tobytes() for name in
                  ('qpos', 'qvel', 'act', 'ctrl', 'mocap_pos', 'mocap_quat')),
                *(np.asarray(getattr(sim.model, name)).tobytes() for name in
                  ('cam_pos', 'cam_quat', 'cam_fovy')))

    def reset(self, *args, **kwargs):
        self._observation_cache = self._observation_fingerprint = None
        return super().reset(*args, **kwargs)

    def step(self, *args, **kwargs):
        self._observation_cache = self._observation_fingerprint = None
        started = perf_counter()
        try:
            result = super().step(*args, **kwargs)
        finally:
            self.native_step_seconds += perf_counter() - started
            self.native_step_count += 1
        self._observation_cache = copy.deepcopy(result[0])
        self._observation_fingerprint = self._state_fingerprint()
        return result

    def get_current_observation(self) -> dict:
        fingerprint = self._state_fingerprint()
        if (fingerprint is not None and self._observation_cache is not None
                and fingerprint == self._observation_fingerprint):
            self.observation_cache_hits += 1
            return copy.deepcopy(self._observation_cache)
        self.observation_cache_misses += 1
        observation = super()._get_observation()
        self._observation_cache = copy.deepcopy(observation)
        self._observation_fingerprint = self._state_fingerprint()
        return copy.deepcopy(observation)

    def render(self) -> np.ndarray:
        return np.flip(self.get_current_observation()['agentview_image'], (0, 1))
