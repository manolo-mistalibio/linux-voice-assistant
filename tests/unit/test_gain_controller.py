"""Tests for the bounded speech gain and named duck leases (#356)."""

import pytest

from linux_voice_assistant.gain_controller import GainController, LeaseError


class FakePlayer:
    """Records the volume percentages a real mpv player would receive."""

    def __init__(self):
        self.volumes = []

    def set_volume(self, volume):
        self.volumes.append(int(volume))

    @property
    def last(self):
        return self.volumes[-1]


def make_controller(**kwargs):
    music, speech = FakePlayer(), FakePlayer()
    controller = GainController(music, speech, autostart=False, **kwargs)
    return controller, music, speech


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------


def test_speech_bounds_lift_low_volume():
    controller, music, speech = make_controller()
    assert controller.speech_bounds == (0.30, 0.70)

    controller.set_media_volume(0.15)
    assert speech.last == 30
    assert music.last == 15


def test_in_range_is_unchanged():
    controller, music, speech = make_controller()
    controller.set_media_volume(0.50)
    assert speech.last == 50
    assert music.last == 50


def test_speech_capped_above_nmax_but_music_idle_is_v():
    controller, music, speech = make_controller()
    controller.set_media_volume(0.90)
    assert speech.last == 70
    assert music.last == 90


def test_speech_min_max_swapped_when_reversed():
    controller, _, _ = make_controller(speech_min=0.8, speech_max=0.2)
    assert controller.speech_bounds == (0.2, 0.8)


# ---------------------------------------------------------------------------
# Zero / mute are silence
# ---------------------------------------------------------------------------


def test_zero_volume_is_silence_even_when_ducked():
    controller, music, speech = make_controller()
    controller.duck("doorbell", 20)
    controller.set_media_volume(0.0)
    assert music.last == 0
    assert speech.last == 0
    assert controller.music_gain() == 0
    assert controller.speech_gain() == 0


def test_mute_and_unmute():
    controller, music, speech = make_controller()
    controller.set_media_volume(0.4)
    controller.duck("doorbell", 20)
    assert music.last == 20

    controller.set_muted(True)
    assert music.last == 0 and speech.last == 0

    # Volume changes while muted update the model but stay silent.
    controller.set_media_volume(0.8)
    assert music.last == 0 and speech.last == 0

    controller.set_muted(False)
    assert speech.last == 70  # clamp(0.8)
    assert music.last == 40  # 0.5 * min(0.8, 0.7)


# ---------------------------------------------------------------------------
# Leases
# ---------------------------------------------------------------------------


def test_single_lease_and_restore():
    controller, music, _ = make_controller()
    controller.set_media_volume(0.5)
    controller.duck("doorbell", 20)
    assert music.last == 25
    assert controller.unduck("doorbell") is True
    assert music.last == 50
    assert controller.unduck("doorbell") is False
    assert music.last == 50


def test_leases_compose_with_lowest_gain_winning():
    controller, music, _ = make_controller()
    controller.set_media_volume(0.5)

    controller.duck(GainController.SATELLITE_REASON)
    controller.duck("doorbell", 20)
    assert controller.music_gain() == pytest.approx(0.25)

    controller.unduck("doorbell")
    assert controller.music_gain() == pytest.approx(0.25)

    controller.unduck(GainController.SATELLITE_REASON)
    assert controller.music_gain() == pytest.approx(0.5)

    controller.duck(GainController.SATELLITE_REASON)
    controller.duck("doorbell", 20, factor=0.2)
    assert controller.music_gain() == pytest.approx(0.2 * 0.5)


def test_above_nmax_caps_music_while_ducked():
    controller, music, speech = make_controller()
    controller.set_media_volume(0.9)
    controller.duck("doorbell", 20)
    assert speech.last == 70
    assert music.last == 35  # 0.5 * min(0.9, 0.7)

    controller.set_media_volume(0.95)
    assert speech.last == 70 and music.last == 35


def test_lease_expiry_uses_monotonic_clock():
    clock = [1000.0]
    music, speech = FakePlayer(), FakePlayer()
    controller = GainController(music, speech, clock=lambda: clock[0], autostart=False)
    controller.set_media_volume(0.5)
    controller.duck("doorbell", 20)
    assert music.last == 25

    clock[0] += 19.0
    assert controller.reap() is False
    assert music.last == 25

    clock[0] += 2.0
    assert controller.reap() is True
    assert music.last == 50


def test_indefinite_lease_never_expires():
    clock = [1000.0]
    music, speech = FakePlayer(), FakePlayer()
    controller = GainController(music, speech, clock=lambda: clock[0], autostart=False)
    controller.set_media_volume(0.5)
    controller.duck(GainController.SATELLITE_REASON)
    clock[0] += 10_000
    assert controller.reap() is False
    assert controller.music_gain() == pytest.approx(0.25)


@pytest.mark.parametrize("source_id", ["", "has space", "bad/slash", "x" * 65])
def test_bad_source_id_rejected(source_id):
    controller, _, _ = make_controller()
    with pytest.raises(LeaseError):
        controller.duck(source_id, 5)


@pytest.mark.parametrize("duration", [0, -1, 301])
def test_bad_duration_rejected(duration):
    controller, _, _ = make_controller(max_lease_seconds=300)
    with pytest.raises(LeaseError):
        controller.duck("doorbell", duration)


def test_max_duration_boundary_allowed():
    controller, _, _ = make_controller(max_lease_seconds=300)
    controller.duck("doorbell", 300)
    controller.unduck("doorbell")


def test_lease_snapshot():
    controller, _, _ = make_controller()
    controller.duck("doorbell", 20)
    snapshot = controller.lease_snapshot()
    assert snapshot[0]["source_id"] == "doorbell"
    assert snapshot[0]["expires_in"] > 0
