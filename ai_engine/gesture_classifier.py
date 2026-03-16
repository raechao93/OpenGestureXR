"""
Gesture classifier. Has two modes:
  1. Rule-based heuristics (default, no model needed)
  2. ONNX neural net (call load_onnx_model() first)
"""

import numpy as np

GESTURES = ["open_hand", "grab", "pinch", "point", "thumbs_up", "peace"]

_onnx_session = None


def load_onnx_model(model_path: str):
    """Switch to ONNX-based classification. Falls back to rules if this isn't called."""
    global _onnx_session
    import onnxruntime as ort
    _onnx_session = ort.InferenceSession(model_path)


def classify_gesture(landmarks: list[list[float]]) -> dict:
    if _onnx_session is not None:
        return _classify_onnx(landmarks)
    return _classify_rules(landmarks)


def _classify_onnx(landmarks):
    flat = np.array(landmarks, dtype=np.float32).flatten().reshape(1, -1)
    input_name = _onnx_session.get_inputs()[0].name
    output = _onnx_session.run(None, {input_name: flat})
    probs = output[0][0]
    idx = int(np.argmax(probs))
    return {"gesture": GESTURES[idx], "confidence": float(probs[idx])}


def _classify_rules(landmarks: list[list[float]]) -> dict:
    """
    Heuristic classifier based on finger extension + thumb-index distance.
    Not great with edge cases (e.g. partially curled fingers) but works
    well enough for the demo gestures.
    """
    lm = landmarks

    def is_extended(tip, pip):
        # finger is "up" when tip is above (lower y) its PIP joint
        return lm[tip][1] < lm[pip][1]

    # pinch = thumb tip very close to index tip
    thumb_idx_dist = float(np.linalg.norm(
        np.array(lm[4][:2]) - np.array(lm[8][:2])
    ))
    if thumb_idx_dist < 0.05:
        return {"gesture": "pinch", "confidence": 0.91}

    fingers = [
        is_extended(8, 6),    # index
        is_extended(12, 10),  # middle
        is_extended(16, 14),  # ring
        is_extended(20, 18),  # pinky
    ]
    thumb_up = lm[4][1] < lm[3][1]
    n_extended = sum(fingers)

    # thumbs up — only thumb, all fingers curled
    if thumb_up and n_extended == 0:
        return {"gesture": "thumbs_up", "confidence": 0.83}

    # peace sign — index + middle up, rest down
    if fingers[0] and fingers[1] and not fingers[2] and not fingers[3]:
        return {"gesture": "peace", "confidence": 0.84}

    if n_extended == 4:
        return {"gesture": "open_hand", "confidence": 0.93}
    if n_extended == 0:
        return {"gesture": "grab", "confidence": 0.87}
    if fingers[0] and not any(fingers[1:]):
        return {"gesture": "point", "confidence": 0.86}

    # fallback — if we can't tell, assume open hand with low confidence
    return {"gesture": "open_hand", "confidence": 0.55}
