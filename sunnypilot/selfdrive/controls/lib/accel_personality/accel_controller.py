"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from collections.abc import Sequence

import numpy as np

from cereal import messaging
from opendbc.car import structs
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.sunnypilot import get_sanitize_int_param
from openpilot.sunnypilot.selfdrive.controls.lib.accel_personality.constants import \
  NORMAL, PERSONALITY_MIN, PERSONALITY_MAX, A_CRUISE_MAX_BP, A_CRUISE_MAX_V, RISE_RATE, SMOOTH_DECEL_BP, SMOOTH_DECEL_V, \
  BRAKE_DEEPENING_JERK, BRAKE_RELEASE_JERK, SMOOTH_DECEL_LOOKAHEAD_T, MIN_SMOOTH_BRAKE_NEED, HARD_BRAKE_TARGET_ACCEL, HARD_BRAKE_NEED

_ZERO_ACCEL_EPS = 1e-6


class AccelController:
  def __init__(self, CP: structs.CarParams, mpc, params=None):
    self._CP = CP
    self._mpc = mpc
    self._params = params or Params()
    self._frame = 0
    self._enabled: bool = self._params.get_bool("AccelPersonalityEnabled")
    self._personality = NORMAL  # cereal AccelerationPersonality ordinal
    self._v_ego = 0.0
    self._last_target_accel = 0.0
    self._brake_need = 0.0
    self._smooth_active = False
    self._bypassed = False
    self._read_params()

  def _read_params(self) -> None:
    self._enabled = self._params.get_bool("AccelPersonalityEnabled")
    if not self._enabled:
      self._personality = NORMAL
      return

    self._personality = get_sanitize_int_param("AccelPersonality", PERSONALITY_MIN, PERSONALITY_MAX, self._params)

  def update(self, sm: messaging.SubMaster) -> None:
    if self._frame % int(1. / DT_MDL) == 0:
      self._read_params()
    self._v_ego = sm['carState'].vEgo
    self._frame += 1

  def get_max_accel(self, v_ego: float) -> float:
    return float(np.interp(v_ego, A_CRUISE_MAX_BP, A_CRUISE_MAX_V[self._personality]))

  def get_rise_rate(self) -> float:
    return RISE_RATE[self._personality]

  def get_decel_target(self, brake_need: float) -> float:
    return float(np.interp(max(0.0, float(brake_need)), SMOOTH_DECEL_BP, SMOOTH_DECEL_V[self._personality]))

  def smooth_target_accel(self, raw_target_accel: float, accel_trajectory: Sequence[float], t_idxs: Sequence[float],
                          should_stop: bool, reset: bool = False) -> float:
    raw_target_accel = float(raw_target_accel)
    self._brake_need = self._compute_brake_need(raw_target_accel, accel_trajectory, t_idxs)

    if reset or not self._enabled:
      self._bypassed = False
      return self._passthrough(raw_target_accel)

    self._bypassed = self._emergency_bypass(raw_target_accel, should_stop)
    if self._bypassed:
      return self._passthrough(raw_target_accel)

    if self._brake_need < MIN_SMOOTH_BRAKE_NEED:
      self._smooth_active = False
      return self._slew(raw_target_accel)

    self._smooth_active = True
    return self._slew(self.get_decel_target(self._brake_need))

  def _compute_brake_need(self, raw_target_accel: float, accel_trajectory: Sequence[float], t_idxs: Sequence[float]) -> float:
    min_accel = float(raw_target_accel)
    for accel, t in zip(accel_trajectory, t_idxs, strict=False):
      if float(t) <= SMOOTH_DECEL_LOOKAHEAD_T:
        min_accel = min(min_accel, float(accel))
    return max(0.0, -min_accel)

  def _emergency_bypass(self, raw_target_accel: float, should_stop: bool) -> bool:
    return (self._mpc.crash_cnt > 0 or should_stop or
            raw_target_accel <= HARD_BRAKE_TARGET_ACCEL or self._brake_need >= HARD_BRAKE_NEED)

  def _passthrough(self, target_accel: float) -> float:
    target_accel = self._clean_accel(target_accel)
    self._last_target_accel = target_accel
    self._smooth_active = False
    return target_accel

  def _slew(self, target_accel: float) -> float:
    rate = BRAKE_DEEPENING_JERK[self._personality] if target_accel < self._last_target_accel else BRAKE_RELEASE_JERK
    step = rate * DT_MDL
    smoothed = self._clean_accel(float(np.clip(target_accel, self._last_target_accel - step, self._last_target_accel + step)))
    self._last_target_accel = smoothed
    return smoothed

  @staticmethod
  def _clean_accel(accel: float) -> float:
    accel = float(accel)
    return 0.0 if abs(accel) < _ZERO_ACCEL_EPS else accel

  def enabled(self) -> bool:
    return self._enabled

  def personality(self):
    return self._personality  # cereal AccelerationPersonality ordinal

  def max_accel(self) -> float:
    # Cached value for publishing; publish_longitudinal_plan_sp has no v_ego in scope.
    return self.get_max_accel(self._v_ego)

  def brake_need(self) -> float:
    return self._brake_need

  def smooth_active(self) -> bool:
    return self._smooth_active

  def bypassed(self) -> bool:
    return self._bypassed
