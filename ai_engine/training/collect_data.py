"""
Collect training data for gesture classification.

Point your webcam at your hand, hold a gesture, and this records
the 21 landmark coordinates to a CSV. Press 'q' to stop early.

    python -m ai_engine.training.collect_data --gesture grab --output data/grab.csv
"""

import argparse
import csv
import cv2
from ai_engine.gesture_detector import detect_hands, create_detector


def collect(gesture_label, output_path, max_samples=500):
    cap = cv2.VideoCapture(0)
    detector = create_detector(max_hands=1)
    rows = []

    print(f"recording '{gesture_label}' — hold the gesture, press q to stop")

    while cap.isOpened() and len(rows) < max_samples:
        ret, frame = cap.read()
        if not ret:
            break

        hands = detect_hands(frame, detector)
        if hands:
            flat = [c for pt in hands[0].landmarks for c in pt]
            rows.append(flat + [gesture_label])

        # show progress on the frame so you know it's working
        cv2.putText(frame, f"{gesture_label}: {len(rows)}/{max_samples}",
                     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("collect", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    header = [f"{a}{i}" for i in range(21) for a in ("x", "y", "z")] + ["label"]
    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    print(f"saved {len(rows)} samples -> {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gesture", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-samples", type=int, default=500)
    args = p.parse_args()
    collect(args.gesture, args.output, args.max_samples)
