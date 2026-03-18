"""Tests for the rule-based gesture classifier."""

import pytest
from ai_engine.gesture_classifier import classify_gesture, GESTURES


def _make_landmarks(tips_up=None, thumb_up=False, pinch=False):
    """Build a fake 21-point landmark array.

    MediaPipe hand layout (y-axis: 0=top, 1=bottom):
      0=wrist, 1-4=thumb, 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky
      PIP joints: 6(index), 10(middle), 14(ring), 18(pinky)
      Tips:       8(index), 12(middle), 16(ring),  20(pinky)
    """
    tips_up = tips_up or []
    lm = [[0.5, 0.5, 0.0]] * 21

    # finger PIP joints at y=0.6, tips default at y=0.7 (curled)
    for pip in (6, 10, 14, 18):
        lm[pip] = [0.5, 0.6, 0.0]
    for tip in (8, 12, 16, 20):
        lm[tip] = [0.5, 0.7, 0.0]

    # extend requested fingers (tip above PIP)
    finger_tips = {0: 8, 1: 12, 2: 16, 3: 20}
    for f in tips_up:
        lm[finger_tips[f]] = [0.5, 0.4, 0.0]

    # thumb
    lm[3] = [0.3, 0.55, 0.0]
    lm[4] = [0.3, 0.45 if thumb_up else 0.65, 0.0]

    if pinch:
        lm[4] = [0.5, 0.5, 0.0]
        lm[8] = [0.52, 0.5, 0.0]

    return lm


class TestRuleBasedClassifier:
    def test_open_hand(self):
        r = classify_gesture(_make_landmarks(tips_up=[0, 1, 2, 3]))
        assert r["gesture"] == "open_hand"
        assert r["confidence"] > 0.5

    def test_grab(self):
        r = classify_gesture(_make_landmarks(tips_up=[]))
        assert r["gesture"] == "grab"
        assert r["confidence"] > 0.5

    def test_pinch(self):
        r = classify_gesture(_make_landmarks(pinch=True))
        assert r["gesture"] == "pinch"

    def test_point(self):
        r = classify_gesture(_make_landmarks(tips_up=[0]))
        assert r["gesture"] == "point"

    def test_thumbs_up(self):
        r = classify_gesture(_make_landmarks(thumb_up=True))
        assert r["gesture"] == "thumbs_up"

    def test_peace(self):
        r = classify_gesture(_make_landmarks(tips_up=[0, 1]))
        assert r["gesture"] == "peace"

    def test_all_gestures_known(self):
        assert len(GESTURES) == 6

    def test_output_has_confidence(self):
        r = classify_gesture(_make_landmarks())
        assert "confidence" in r
        assert 0.0 <= r["confidence"] <= 1.0
