import numpy as np
import pytest

from openpilot.sunnypilot.selfdrive.controls.lib.accel_personality.accel_controller import AccelController
from openpilot.sunnypilot.selfdrive.controls.lib.accel_personality.constants import \
  AccelerationPersonality, ECO, NORMAL, SPORT, STOCK_RISE_RATE, SMOOTH_DECEL_BP, SMOOTH_DECEL_V, BRAKE_DEEPENING_JERK, \
  BRAKE_RELEASE_JERK, ACCEL_RISE_JERK, HARD_BRAKE_TARGET_ACCEL, HARD_BRAKE_NEED
from openpilot.common.realtime import DT_MDL

# Stock openpilot accel ceiling, duplicated independently here so the test fails if the normal tier ever drifts.
STOCK_BP = [0., 10., 25., 40.]
STOCK_V = [1.6, 1.2, 0.8, 0.6]
V_EGO_GRID = [0.0, 5.0, 10.0, 17.0, 25.0, 33.0, 40.0, 50.0]


class MockParams:
  def __init__(self, enabled=False, personality=NORMAL):
    self._vals = {"AccelPersonalityEnabled": enabled, "AccelPersonality": int(personality)}

  def get_bool(self, key):
    return bool(self._vals.get(key, False))

  def get(self, key, return_default=False):
    return self._vals.get(key, 0)

  def put(self, key, val, block=False):
    self._vals[key] = int(val)


class MockCarState:
  def __init__(self, vEgo=0.0):
    self.vEgo = vEgo


def make_sm(v_ego=0.0):
  return {'carState': MockCarState(vEgo=v_ego)}


@pytest.fixture
def mock_cp():
  class CP:
    radarUnavailable = False
  return CP()


@pytest.fixture
def mock_mpc():
  class MPC:
    crash_cnt = 0
  return MPC()


def stock_max_accel(v_ego):
  return float(np.interp(v_ego, STOCK_BP, STOCK_V))


def test_disabled_matches_stock(mock_cp, mock_mpc):
  c = AccelController(mock_cp, mock_mpc, params=MockParams(enabled=False, personality=SPORT))
  for v in V_EGO_GRID:
    assert c.get_max_accel(v) == pytest.approx(stock_max_accel(v))
  assert c.get_rise_rate() == STOCK_RISE_RATE
  assert c.personality() == AccelerationPersonality.normal
  assert not c.enabled()
  assert c.smooth_target_accel(-0.5, [-2.0], [1.0], should_stop=False) == -0.5


def test_normal_matches_stock(mock_cp, mock_mpc):
  c = AccelController(mock_cp, mock_mpc, params=MockParams(enabled=True, personality=NORMAL))
  for v in V_EGO_GRID:
    assert c.get_max_accel(v) == pytest.approx(stock_max_accel(v))
  assert c.get_rise_rate() == STOCK_RISE_RATE
  assert c.personality() == AccelerationPersonality.normal


def test_eco_is_gentler(mock_cp, mock_mpc):
  c = AccelController(mock_cp, mock_mpc, params=MockParams(enabled=True, personality=ECO))
  for v in V_EGO_GRID:
    assert c.get_max_accel(v) <= stock_max_accel(v) + 1e-6
  assert c.get_rise_rate() < STOCK_RISE_RATE
  assert c.personality() == AccelerationPersonality.eco


def test_sport_is_brisker(mock_cp, mock_mpc):
  c = AccelController(mock_cp, mock_mpc, params=MockParams(enabled=True, personality=SPORT))
  for v in V_EGO_GRID:
    assert c.get_max_accel(v) >= stock_max_accel(v) - 1e-6
  assert c.get_rise_rate() > STOCK_RISE_RATE
  assert c.personality() == AccelerationPersonality.sport


def test_param_clamp(mock_cp, mock_mpc):
  # Out-of-range int must clamp to the max tier (sport), not raise.
  c = AccelController(mock_cp, mock_mpc, params=MockParams(enabled=True, personality=9))
  assert c.personality() == AccelerationPersonality.sport


@pytest.mark.parametrize("personality", [ECO, NORMAL, SPORT])
def test_profile_decel_tables(mock_cp, mock_mpc, personality):
  c = AccelController(mock_cp, mock_mpc, params=MockParams(enabled=True, personality=personality))
  for bp, expected in zip(SMOOTH_DECEL_BP, SMOOTH_DECEL_V[personality], strict=True):
    assert c.get_decel_target(bp) == pytest.approx(expected)


def test_profile_jerk_limits_match_plan():
  assert BRAKE_DEEPENING_JERK[ECO] == pytest.approx(0.7)
  assert BRAKE_DEEPENING_JERK[NORMAL] == pytest.approx(0.9)
  assert BRAKE_DEEPENING_JERK[SPORT] == pytest.approx(1.1)
  assert BRAKE_RELEASE_JERK == pytest.approx(2.0)
  assert ACCEL_RISE_JERK[ECO] < ACCEL_RISE_JERK[NORMAL]


def test_zero_accel_outputs_positive_zero(mock_cp, mock_mpc):
  c = AccelController(mock_cp, mock_mpc, params=MockParams(enabled=True, personality=NORMAL))

  out = c.smooth_target_accel(-0.0, [-0.0], [0.0], should_stop=False, reset=True)

  assert out == 0.0
  assert not np.signbit(out)


def test_future_brake_need_starts_decel_before_raw_target_brakes(mock_cp, mock_mpc):
  c = AccelController(mock_cp, mock_mpc, params=MockParams(enabled=True, personality=NORMAL))
  c.smooth_target_accel(0.2, [0.2], [0.0], should_stop=False, reset=True)

  out = c.smooth_target_accel(0.2, [0.2, -1.2], [0.0, 1.0], should_stop=False)

  assert c.brake_need() == pytest.approx(1.2)
  assert c.smooth_active()
  assert out < 0.2


def test_brake_deepening_is_jerk_limited(mock_cp, mock_mpc):
  c = AccelController(mock_cp, mock_mpc, params=MockParams(enabled=True, personality=NORMAL))
  c.smooth_target_accel(0.0, [0.0], [0.0], should_stop=False, reset=True)

  out = c.smooth_target_accel(0.0, [-1.2], [1.0], should_stop=False)

  assert out == pytest.approx(-BRAKE_DEEPENING_JERK[NORMAL] * DT_MDL)


def test_brake_release_is_faster_than_deepening(mock_cp, mock_mpc):
  c = AccelController(mock_cp, mock_mpc, params=MockParams(enabled=True, personality=NORMAL))
  c.smooth_target_accel(-1.0, [-1.0], [0.0], should_stop=False, reset=True)

  out = c.smooth_target_accel(0.5, [0.5], [0.0], should_stop=False)

  assert out == pytest.approx(-1.0 + BRAKE_RELEASE_JERK * DT_MDL)
  assert BRAKE_RELEASE_JERK > BRAKE_DEEPENING_JERK[NORMAL]


def test_eco_positive_accel_rise_is_smooth(mock_cp, mock_mpc):
  c = AccelController(mock_cp, mock_mpc, params=MockParams(enabled=True, personality=ECO))
  c.smooth_target_accel(0.0, [0.0], [0.0], should_stop=False, reset=True)

  out = c.smooth_target_accel(1.0, [1.0], [0.0], should_stop=False)

  assert out == pytest.approx(ACCEL_RISE_JERK[ECO] * DT_MDL)


def test_eco_brake_release_does_not_jump_to_gas(mock_cp, mock_mpc):
  c = AccelController(mock_cp, mock_mpc, params=MockParams(enabled=True, personality=ECO))
  c.smooth_target_accel(-0.01, [-0.01], [0.0], should_stop=False, reset=True)

  out = c.smooth_target_accel(1.0, [1.0], [0.0], should_stop=False)

  assert out == pytest.approx(ACCEL_RISE_JERK[ECO] * DT_MDL)


@pytest.mark.parametrize("raw_target, accel_trajectory, should_stop, crash_cnt", [
  (HARD_BRAKE_TARGET_ACCEL - 0.1, [-0.5], False, 0),
  (-0.5, [-HARD_BRAKE_NEED], False, 0),
  (-0.5, [-0.5], True, 0),
  (-0.5, [-0.5], False, 1),
])
def test_emergency_bypass_passthrough(mock_cp, mock_mpc, raw_target, accel_trajectory, should_stop, crash_cnt):
  mock_mpc.crash_cnt = crash_cnt
  c = AccelController(mock_cp, mock_mpc, params=MockParams(enabled=True, personality=NORMAL))
  c.smooth_target_accel(0.0, [0.0], [0.0], should_stop=False, reset=True)

  out = c.smooth_target_accel(raw_target, accel_trajectory, [1.0], should_stop=should_stop)

  assert out == raw_target
  assert c.bypassed()


def test_frame_gated_read(mock_cp, mock_mpc):
  params = MockParams(enabled=True, personality=NORMAL)
  c = AccelController(mock_cp, mock_mpc, params=params)
  c.update(make_sm())  # frame 0 -> reads, picks up normal
  params._vals["AccelPersonality"] = int(SPORT)
  for _ in range(19):  # frames 1..19, none on the 20-frame gate boundary
    c.update(make_sm())
  assert c.personality() == AccelerationPersonality.normal
  c.update(make_sm())  # frame 20 -> reads, picks up sport
  assert c.personality() == AccelerationPersonality.sport
