"""Bounded speech gain and named music duck leases (issue #356).

The upstream `v1.1.15` media-player entity writes the same persisted volume to
both the music player and the announcement/TTS player, so a low music volume
also makes speech quiet. This module is the single place that turns the one
persisted media volume ``V`` (0.0–1.0) into the two *effective* gains the two
mpv players should use:

* speech/notification gain ``N = clamp(V, Nmin, Nmax)`` while ``V`` is audible;
* music gain ``V`` when idle, or ``duck_ratio × min(V, Nmax)`` — scaled by the
  lowest requested gain — while one or more named duck leases are active.

Two invariants make the behaviour safe:

* ``V == 0`` or a mute is **silence**. Volume bounds never resurrect audio the
  user explicitly turned off; they only lift speech above a low-but-audible
  ``V``.
* External duck leases are keyed by a caller-chosen ``source_id`` and expire on
  a monotonic clock, so a missed ``unduck`` cannot strand music at low volume.

The controller is deliberately dependency-free (stdlib only) so it can be unit
tested without mpv, aioesphomeapi or a running LVA instance.
"""

from __future__ import annotations

import logging
import math
import re
import threading
import time
from typing import Callable, Dict, List, Optional

_LOGGER = logging.getLogger(__name__)

#: Reason name used by the voice pipeline (wake word, TTS, timers,
#: announcements). Kept as a constant because the satellite, the peripheral
#: API and the reaper all need to agree on it.
SATELLITE_REASON = "satellite"

#: source_id / reason names are caller supplied and end up in logs and the
#: lease snapshot, so keep them short and boring.
_SOURCE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


class LeaseError(ValueError):
    """Raised for an invalid duck request (bad source_id, duration or factor)."""


def _clamp(value: float, low: float, high: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise LeaseError("gain values must be finite")
    return max(low, min(high, value))


class GainController:
    """Owns the persisted media volume and applies bounded/ducked gains.

    Parameters
    ----------
    music_player, speech_player:
        Objects exposing ``set_volume(percent: float)`` where ``percent`` is
        0–100. These are the LVA ``MpvMediaPlayer`` instances (music and
        TTS/announcement).
    duck_ratio:
        Default music reduction factor while ducked (upstream used 0.5).
    speech_min, speech_max:
        Notification bounds ``Nmin``/``Nmax`` as fractions (0.0–1.0).
    max_lease_seconds:
        Upper bound for an external lease duration.
    clock:
        Monotonic clock, injectable for tests.
    reap_interval:
        How often the background reaper looks for expired leases. Only used
        when ``autostart`` is true.
    autostart:
        Start the lease-reaper thread. Tests pass ``False`` and call
        :meth:`reap` explicitly.
    """

    def __init__(
        self,
        music_player,
        speech_player,
        *,
        duck_ratio: float = 0.5,
        speech_min: float = 0.30,
        speech_max: float = 0.70,
        max_lease_seconds: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
        reap_interval: float = 0.5,
        autostart: bool = True,
    ) -> None:
        duck_ratio = _clamp(duck_ratio, 0.0, 1.0)
        speech_min = _clamp(speech_min, 0.0, 1.0)
        speech_max = _clamp(speech_max, 0.0, 1.0)
        if speech_min > speech_max:
            # Be forgiving on the operator's input order; the contract is a
            # window, not an ordering requirement.
            speech_min, speech_max = speech_max, speech_min
        if not math.isfinite(max_lease_seconds) or max_lease_seconds <= 0:
            raise LeaseError("max_lease_seconds must be finite and > 0")

        self._music = music_player
        self._speech = speech_player
        self._duck_ratio = duck_ratio
        self._speech_min = speech_min
        self._speech_max = speech_max
        self._max_lease_seconds = float(max_lease_seconds)
        self._clock = clock

        self._media_volume = 0.0
        self._muted = False
        # reason -> (expiry_monotonic or None for indefinite, factor)
        self._leases: Dict[str, float] = {}
        self._lease_factor: Dict[str, float] = {}
        self._lease_expiry: Dict[str, Optional[float]] = {}

        self._lock = threading.RLock()
        self._reap_interval = max(0.05, float(reap_interval))
        self._stop_event = threading.Event()
        self._reaper: Optional[threading.Thread] = None

        if autostart:
            self._start_reaper()

        # Push the initial (zero) gains so the players agree with the model.
        self.apply()

    # ------------------------------------------------------------------
    # Properties / tuning
    # ------------------------------------------------------------------

    @property
    def duck_ratio(self) -> float:
        return self._duck_ratio

    @property
    def speech_bounds(self) -> tuple[float, float]:
        return (self._speech_min, self._speech_max)

    @property
    def max_lease_seconds(self) -> float:
        return self._max_lease_seconds

    @property
    def muted(self) -> bool:
        with self._lock:
            return self._muted

    def get_media_volume(self) -> float:
        with self._lock:
            return self._media_volume

    # ------------------------------------------------------------------
    # Media volume / mute
    # ------------------------------------------------------------------

    def set_media_volume(self, volume: float) -> float:
        """Set the persisted media volume ``V`` and re-apply both gains."""
        with self._lock:
            self._media_volume = _clamp(volume, 0.0, 1.0)
            self.apply_locked()
            return self._media_volume

    def set_muted(self, muted: bool) -> None:
        """Mute/unmute the media voice path. Mute means silence, not ``Nmin``."""
        with self._lock:
            self._muted = bool(muted)
            self.apply_locked()

    # ------------------------------------------------------------------
    # Named duck leases
    # ------------------------------------------------------------------

    def duck(
        self,
        source_id: str,
        duration_seconds: Optional[float] = None,
        *,
        factor: Optional[float] = None,
    ) -> Optional[float]:
        """Acquire or renew a named music duck lease.

        ``duration_seconds=None`` creates an indefinite lease (used by the
        voice pipeline, which releases it explicitly via :meth:`unduck`).
        External callers must pass a positive, bounded duration. Returns the
        lease expiry (monotonic) or ``None`` for an indefinite lease.
        """
        self._validate_source_id(source_id)
        if duration_seconds is not None:
            duration_seconds = float(duration_seconds)
            if not math.isfinite(duration_seconds) or duration_seconds <= 0:
                raise LeaseError("duration_seconds must be finite and > 0")
            if duration_seconds > self._max_lease_seconds:
                raise LeaseError(f"duration_seconds must be <= {self._max_lease_seconds:g}")
            expiry: Optional[float] = self._clock() + duration_seconds
        else:
            expiry = None

        if factor is None:
            lease_factor = self._duck_ratio
        else:
            lease_factor = float(factor)
            if not math.isfinite(lease_factor) or not 0.0 <= lease_factor <= 1.0:
                raise LeaseError("factor must be finite and between 0 and 1")

        with self._lock:
            self._leases[source_id] = self._clock()
            self._lease_expiry[source_id] = expiry
            self._lease_factor[source_id] = lease_factor
            self.apply_locked()

        _LOGGER.debug(
            "music duck lease held: source=%s factor=%.3f expiry=%s",
            source_id,
            lease_factor,
            "indefinite" if expiry is None else f"{duration_seconds:.1f}s",
        )
        return expiry

    def unduck(self, source_id: str) -> bool:
        """Release one named lease. Repeated removal is harmless."""
        self._validate_source_id(source_id)
        with self._lock:
            existed = source_id in self._leases
            self._leases.pop(source_id, None)
            self._lease_expiry.pop(source_id, None)
            self._lease_factor.pop(source_id, None)
            if existed:
                self.apply_locked()
        if existed:
            _LOGGER.debug("music duck lease released: source=%s", source_id)
        return existed

    def lease_snapshot(self) -> List[Dict[str, object]]:
        """Return the active leases for health/debug endpoints."""
        now = self._clock()
        with self._lock:
            if self._purge_locked(now):
                self.apply_locked()
            snapshot: List[Dict[str, object]] = []
            for source_id in sorted(self._leases):
                expiry = self._lease_expiry.get(source_id)
                snapshot.append(
                    {
                        "source_id": source_id,
                        "factor": self._lease_factor.get(source_id),
                        "expires_in": None if expiry is None else max(0.0, expiry - now),
                    }
                )
            return snapshot

    # ------------------------------------------------------------------
    # Gains
    # ------------------------------------------------------------------

    def speech_gain(self) -> float:
        with self._lock:
            return self._target_speech_gain_locked()

    def music_gain(self) -> float:
        with self._lock:
            if self._purge_locked(self._clock()):
                self.apply_locked()
            return self._target_music_gain_locked()

    def _target_speech_gain_locked(self) -> float:
        if self._muted or self._media_volume <= 0.0:
            return 0.0
        return _clamp(self._media_volume, self._speech_min, self._speech_max)

    def _target_music_gain_locked(self) -> float:
        if self._muted or self._media_volume <= 0.0:
            return 0.0
        if not self._lease_factor:
            # Idle: music always returns to the persisted volume, even above
            # Nmax. Bounds only constrain the speech/notification path.
            return self._media_volume
        lowest = min(self._lease_factor.values())
        # min(V, Nmax): below Nmin and in range the proportional duck applies;
        # above Nmax the music is pulled down to the capped speech level.
        return lowest * min(self._media_volume, self._speech_max)

    def apply(self) -> None:
        with self._lock:
            self.apply_locked()

    def apply_locked(self) -> None:
        music = self._target_music_gain_locked()
        speech = self._target_speech_gain_locked()
        self._set_player_volume(self._music, music)
        self._set_player_volume(self._speech, speech)

    @staticmethod
    def _set_player_volume(player, gain: float) -> None:
        percent = int(round(_clamp(gain, 0.0, 1.0) * 100))
        try:
            player.set_volume(percent)
        except Exception:  # pragma: no cover - a dead player must not kill LVA
            _LOGGER.exception("Failed to apply volume %d to %r", percent, player)

    # ------------------------------------------------------------------
    # Lease reaping
    # ------------------------------------------------------------------

    def reap(self) -> bool:
        """Purge expired leases and re-apply if anything changed."""
        with self._lock:
            changed = self._purge_locked(self._clock())
            if changed:
                self.apply_locked()
        return changed

    def _purge_locked(self, now: float) -> bool:
        expired = [
            source_id
            for source_id, expiry in self._lease_expiry.items()
            if expiry is not None and now >= expiry
        ]
        for source_id in expired:
            self._leases.pop(source_id, None)
            self._lease_expiry.pop(source_id, None)
            self._lease_factor.pop(source_id, None)
            _LOGGER.debug("music duck lease expired: source=%s", source_id)
        return bool(expired)

    def _start_reaper(self) -> None:
        self._reaper = threading.Thread(
            target=self._reap_loop,
            name="lva-gain-reaper",
            daemon=True,
        )
        self._reaper.start()

    def _reap_loop(self) -> None:
        while not self._stop_event.wait(self._reap_interval):
            try:
                self.reap()
            except Exception:  # pragma: no cover - keep the thread alive
                _LOGGER.exception("lease reaper iteration failed")

    def close(self) -> None:
        """Stop the reaper thread (used on shutdown and in tests)."""
        self._stop_event.set()
        reaper = self._reaper
        if reaper is not None and reaper.is_alive():
            reaper.join(timeout=2.0)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_source_id(source_id: str) -> None:
        if not isinstance(source_id, str) or not _SOURCE_ID_RE.match(source_id):
            raise LeaseError("source_id must match ^[A-Za-z0-9_.-]{1,64}$")
