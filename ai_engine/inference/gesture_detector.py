"""
Standalone detection loop — opens webcam, runs detection + classification,
and prints (or callbacks) results each frame. Useful for quick testing
without spinning up the full server.

    python -m ai_engine.inference.gesture_detector
"""

import cv2
import time
from ai_engine.gesture_detector import detect_hands, create_detector
from ai_engine.gesture_classifier import classify_gesture


def run_detector(on_gesture=None, max_hands=2):
    cap = cv2.VideoCapture(0)
    detector = create_detector(max_hands=max_hands)
    t_prev = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        hands = detect_hands(frame, detector)
        hand_results = []
        for h in hands:
            c = classify_gesture(h.landmarks)
            hand_results.append({
                "handedness": h.handedness,
                "gesture": c["gesture"],
                "confidence": c["confidence"],
            })

        now = time.time()
        dt = now - t_prev
        t_prev = now

        result = {
            "hands": hand_results,
            "gesture": hand_results[0]["gesture"] if hand_results else "none",
            "confidence": hand_results[0]["confidence"] if hand_results else 0.0,
            "fps": round(1.0 / max(dt, 0.001), 1),
        }

        if on_gesture:
            on_gesture(result)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_detector(on_gesture=print)
