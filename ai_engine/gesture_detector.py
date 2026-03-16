"""
Hand landmark detection via MediaPipe. Wraps the multi-hand API
and exposes a simpler interface for the rest of the pipeline.
"""

import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass

mp_hands = mp.solutions.hands


@dataclass
class HandResult:
    landmarks: list[list[float]]  # 21 points, each [x, y, z]
    handedness: str               # "Left" or "Right"


def create_detector(max_hands=2, min_confidence=0.7):
    return mp_hands.Hands(
        max_num_hands=max_hands,
        min_detection_confidence=min_confidence,
        min_tracking_confidence=0.5,
    )


def detect_hands(frame: np.ndarray, detector=None) -> list[HandResult]:
    if detector is None:
        detector = _default_detector

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)

    if not results.multi_hand_landmarks:
        return []

    hands = []
    for i, hand_lm in enumerate(results.multi_hand_landmarks):
        # mediapipe sometimes returns handedness list shorter than landmarks
        # (seen this happen when hands overlap), so we guard the index
        handedness = "Right"
        if results.multi_handedness and i < len(results.multi_handedness):
            handedness = results.multi_handedness[i].classification[0].label
        hands.append(HandResult(
            landmarks=[[p.x, p.y, p.z] for p in hand_lm.landmark],
            handedness=handedness,
        ))
    return hands


# keep a single-hand detector around for the old API
_default_detector = create_detector(max_hands=1)


def detect_hand(frame: np.ndarray) -> dict:
    """Single-hand API, kept for backward compat with the REST endpoint."""
    hands = detect_hands(frame, _default_detector)
    if not hands:
        return {"hand_detected": False}
    return {"hand_detected": True, "landmarks": hands[0].landmarks}
