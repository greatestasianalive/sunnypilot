"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from cereal import custom

# Profile ids come from cereal: eco @0, normal @1, sport @2.
AccelerationPersonality = custom.LongitudinalPlanSP.AccelerationPersonality
ECO = AccelerationPersonality.eco
NORMAL = AccelerationPersonality.normal
SPORT = AccelerationPersonality.sport

PERSONALITY_MIN = min(AccelerationPersonality.schema.enumerants.values())
PERSONALITY_MAX = max(AccelerationPersonality.schema.enumerants.values())

A_CRUISE_MAX_BP = [0., 10., 25., 40.]

# Stock openpilot acceleration ceiling. Normal and disabled mode intentionally match this path.
STOCK_A_CRUISE_MAX_V = [1.6, 1.2, 0.8, 0.6]
STOCK_RISE_RATE = 0.05

# Eco keeps launch near stock and mainly softens cruise/highway roll-on.
A_CRUISE_MAX_V = {
  ECO:    [1.6, 1.10, 0.55, 0.40],
  NORMAL: STOCK_A_CRUISE_MAX_V,
  SPORT:  [1.8, 1.40, 1.00, 0.75],
}

RISE_RATE = {
  ECO:    0.025,
  NORMAL: STOCK_RISE_RATE,
  SPORT:  0.10,
}

# Positive predicted brake need -> comfort accel target.
SMOOTH_DECEL_BP = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4]
SMOOTH_DECEL_V = {
  ECO:    [0.00, -0.12, -0.28, -0.50, -0.78, -1.05, -1.30],
  NORMAL: [0.00, -0.15, -0.35, -0.65, -0.95, -1.25, -1.55],
  SPORT:  [0.00, -0.18, -0.45, -0.80, -1.15, -1.45, -1.75],
}

BRAKE_DEEPENING_JERK = {
  ECO:    0.7,
  NORMAL: 0.9,
  SPORT:  1.1,
}

BRAKE_RELEASE_JERK = 2.0
SMOOTH_DECEL_LOOKAHEAD_T = 2.5
MIN_SMOOTH_BRAKE_NEED = 0.05
HARD_BRAKE_TARGET_ACCEL = -2.0
HARD_BRAKE_NEED = 2.6
