"""
Quick benchmark for the detection + classification pipeline.
Runs N frames through the webcam and reports latency stats.

    python -m ai_engine.utils.benchmark --frames 200
"""

import argparse
import time
import cv2
import numpy as np
from ai_engine.gesture_detector import detect_hands, create_detector
from ai_engine.gesture_classifier import classify_gesture


def benchmark(n_frames=100):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("error: can't open webcam")
        return

    detector = create_detector(max_hands=1)
    times = []

    print(f"running {n_frames} frames...")

    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        hands = detect_hands(frame, detector)
        if hands:
            classify_gesture(hands[0].landmarks)
        ms = (time.perf_counter() - t0) * 1000
        times.append(ms)

    cap.release()

    t = np.array(times)
    print(f"\n  {len(t)} frames")
    print(f"  mean:   {t.mean():.1f} ms")
    print(f"  median: {np.median(t):.1f} ms")
    print(f"  p95:    {np.percentile(t, 95):.1f} ms")
    print(f"  p99:    {np.percentile(t, 99):.1f} ms")
    print(f"  range:  {t.min():.1f} – {t.max():.1f} ms")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=int, default=100)
    benchmark(p.parse_args().frames)
