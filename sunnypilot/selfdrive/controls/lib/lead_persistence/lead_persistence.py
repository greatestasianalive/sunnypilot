"""
Copyright (c) 2021-, rav4kumar, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from collections import deque
from dataclasses import dataclass


_HOLD_FRAMES = 16
_STATUS_WINDOW = 20
_STABILITY_FLIPS_FULL = 6.0

# Fresh low-confidence close leads only get masked when they are not urgent.
_URGENT_TTC_MIN = 4.0             # s, below this = urgent, no masking
_URGENT_CLOSING_VREL = -8.0       # m/s, faster closing = no masking

# Phantom-lead mask: a fresh low-modelProb close lead is usually a radar ghost; mask it
# so the MPC ignores it. Established/held leads are never masked (see smooth()).
_PHANTOM_MODELPROB_MAX = 0.5
_PHANTOM_DREL_MAX = 5.0


@dataclass
class _LeadSnap:
  dRel: float = 0.0
  yRel: float = 0.0
  vRel: float = 0.0
  vLead: float = 0.0
  aLeadK: float = 0.0
  aLeadTau: float = 0.0
  modelProb: float = 0.0
  aRel: float = 0.0
  fcw: bool = False


class _LeadProxy:
  __slots__ = ('status', 'dRel', 'yRel', 'vRel', 'vLead', 'aLeadK', 'aLeadTau',
               'modelProb', 'aRel', 'fcw')

  def __init__(self, snap: _LeadSnap):
    self.status = True
    self.dRel = snap.dRel
    self.yRel = snap.yRel
    self.vRel = snap.vRel
    self.vLead = snap.vLead
    self.aLeadK = snap.aLeadK
    self.aLeadTau = snap.aLeadTau
    self.modelProb = snap.modelProb
    self.aRel = snap.aRel
    self.fcw = snap.fcw


class _LeadProxyMasked:
  __slots__ = ('status', 'dRel', 'yRel', 'vRel', 'vLead', 'aLeadK', 'aLeadTau',
               'modelProb', 'aRel', 'fcw')

  def __init__(self):
    self.status = False
    self.dRel = 0.0
    self.yRel = 0.0
    self.vRel = 0.0
    self.vLead = 0.0
    self.aLeadK = 0.0
    self.aLeadTau = 0.0
    self.modelProb = 0.0
    self.aRel = 0.0
    self.fcw = False


class _RadarStateProxy:
  __slots__ = ('_raw', '_lead_one', '_lead_two')

  def __init__(self, raw, lead_one, lead_two):
    self._raw = raw
    self._lead_one = lead_one
    self._lead_two = lead_two

  @property
  def leadOne(self):
    return self._lead_one if self._lead_one is not None else self._raw.leadOne

  @property
  def leadTwo(self):
    return self._lead_two if self._lead_two is not None else self._raw.leadTwo

  def __getattr__(self, name):
    return getattr(self._raw, name)


class LeadPersistence:
  """Internal helper. Hold last-known leadOne/leadTwo alive for HOLD_FRAMES
  after a status drop, so the MPC view of radarState ignores brief flicker.
  Also masks freshly-acquired close-range low-confidence phantom leads so the
  planner doesn't demand emergency brake on radar ghosts.
  No own param — owner (RadarDistanceController) gates via force_enabled."""

  def __init__(self):
    self._last_one: _LeadSnap | None = None
    self._last_two: _LeadSnap | None = None
    self._alive_one = 0
    self._alive_two = 0

    self._status_hist: deque[bool] = deque(maxlen=_STATUS_WINDOW)
    self._stability = 1.0

  @property
  def stability(self) -> float:
    return self._stability

  def reset(self) -> None:
    self._last_one = None
    self._last_two = None
    self._alive_one = 0
    self._alive_two = 0
    self._status_hist.clear()
    self._stability = 1.0

  def update(self, radarstate, force_enabled: bool = True) -> None:
    if radarstate is None:
      return
    if not force_enabled:
      self.reset()
      return

    one = radarstate.leadOne
    two = radarstate.leadTwo

    # phantom counts as not-valid so hold-alive doesn't re-inject the ghost
    one_valid = bool(one.status) and not self._is_phantom(one)
    two_valid = bool(two.status) and not self._is_phantom(two)

    if one_valid:
      self._last_one = self._snap(one)
      self._alive_one = _HOLD_FRAMES
    elif self._alive_one > 0:
      self._alive_one -= 1

    if two_valid:
      self._last_two = self._snap(two)
      self._alive_two = _HOLD_FRAMES
    elif self._alive_two > 0:
      self._alive_two -= 1

    self._status_hist.append(bool(one.status))
    if len(self._status_hist) >= 5:
      flips = sum(1 for i in range(1, len(self._status_hist))
                  if self._status_hist[i] != self._status_hist[i - 1])
      self._stability = max(0.0, 1.0 - min(1.0, flips / _STABILITY_FLIPS_FULL))
    else:
      self._stability = 1.0

  @staticmethod
  def _is_phantom(lead) -> bool:
    if not (bool(lead.status)
            and float(lead.modelProb) < _PHANTOM_MODELPROB_MAX
            and float(lead.dRel) < _PHANTOM_DREL_MAX):
      return False
    # don't mask an urgently-closing lead
    v_rel = float(lead.vRel)
    if v_rel <= _URGENT_CLOSING_VREL:
      return False
    ttc = float(lead.dRel) / max(0.1, -v_rel) if v_rel < 0.0 else float('inf')
    return ttc > _URGENT_TTC_MIN

  def smooth(self, radarstate, force_enabled: bool = True):
    if not force_enabled or radarstate is None:
      return radarstate

    l1 = None
    l2 = None

    # mask only a fresh ghost (alive==0); never drop an established/held lead
    if self._is_phantom(radarstate.leadOne) and self._alive_one == 0:
      l1 = _LeadProxyMasked()
    elif not radarstate.leadOne.status and self._alive_one > 0 and self._last_one is not None:
      l1 = _LeadProxy(self._last_one)

    if self._is_phantom(radarstate.leadTwo) and self._alive_two == 0:
      l2 = _LeadProxyMasked()
    elif not radarstate.leadTwo.status and self._alive_two > 0 and self._last_two is not None:
      l2 = _LeadProxy(self._last_two)

    if l1 is None and l2 is None:
      return radarstate

    return _RadarStateProxy(radarstate, l1, l2)

  @staticmethod
  def _snap(lead) -> _LeadSnap:
    return _LeadSnap(
      dRel=float(lead.dRel),
      yRel=float(lead.yRel),
      vRel=float(lead.vRel),
      vLead=float(lead.vLead),
      aLeadK=float(lead.aLeadK),
      aLeadTau=float(lead.aLeadTau),
      modelProb=float(lead.modelProb),
      aRel=float(lead.aRel),
      fcw=bool(lead.fcw),
    )
