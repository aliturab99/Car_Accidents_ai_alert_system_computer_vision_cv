"""Standalone live accident monitor.

Usage:   python live_accident_monitor.py [path/to/video.mp4] [--every 3] [--threshold 0.6]
Needs:   incident_model.keras and incident_classes.json (both saved by the notebook), opencv-python, keras (torch backend).
Keys:    q or Esc closes the window.
"""
import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import argparse, json
from pathlib import Path

import cv2
import keras
import numpy as np

ALERT_CLASSES = ["accident", "cctv_accident", "fire"]


class TriggerLogic:
    # Smooths P(alert), requires several consecutive high frames, then enforces a cooldown
    def __init__(self, threshold, min_consecutive, cooldown_s, ema=0.5):
        self.threshold, self.min_consecutive, self.cooldown_s, self.ema = threshold, min_consecutive, cooldown_s, ema
        self.smooth, self.streak, self.last_alert = None, 0, -1e9

    def update(self, p, t):
        self.smooth = p if self.smooth is None else self.ema * self.smooth + (1 - self.ema) * p
        self.streak = self.streak + 1 if self.smooth >= self.threshold else 0
        fire = self.streak >= self.min_consecutive and t - self.last_alert >= self.cooldown_s
        if fire:
            self.last_alert, self.streak = t, 0
        return self.smooth, fire


def watch_video(video_path, model_path="incident_model.keras", classes_path="incident_classes.json", every_n=3, threshold=0.6, min_consecutive=3, cooldown_s=30):
    model = keras.saving.load_model(model_path, compile=False)
    class_names = json.loads(Path(classes_path).read_text())
    alert_idx = [class_names.index(c) for c in ALERT_CLASSES if c in class_names]
    size = tuple(model.input_shape[1:3])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    logic = TriggerLogic(threshold, min_consecutive, cooldown_s)
    Path("alerts").mkdir(exist_ok=True)
    n, label, smooth, last_fire_t, alerts = 0, "-", 0.0, -1e9, 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        t = n / fps
        if n % every_n == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            x = cv2.resize(rgb, size[::-1]).astype("float32")[None] # raw 0-255, the model normalises itself
            probs = keras.ops.convert_to_numpy(model(x, training=False))[0]
            smooth, fire = logic.update(float(probs[alert_idx].sum()), t)
            label = class_names[int(np.argmax(probs))]
            if fire:
                alerts += 1
                last_fire_t = t
                snap = f"alerts/alert_{alerts:03d}_{t:06.1f}s.jpg"
                cv2.imwrite(snap, frame)
                print(f"ACCIDENT ALERT at {t:.1f}s  confidence={smooth:.0%}  snapshot={snap}")

        alerting = t - last_fire_t < 3
        color = (0, 0, 255) if alerting else (0, 200, 0)
        cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), color, 6 if alerting else 2)
        cv2.putText(frame, f"{'ACCIDENT DETECTED' if alerting else label}  P(alert)={smooth:.2f}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("Accident monitor - press q to quit", frame)
        if cv2.waitKey(max(1, int(1000 / fps))) & 0xFF in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. {alerts} alert(s) raised.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", default=r"testing_footages\v_1.mp4",
                    help="video file (default: testing_footages\\v_1.mp4)")
    ap.add_argument("--every", type=int, default=3, help="analyse every Nth frame")
    ap.add_argument("--threshold", type=float, default=0.6)
    args = ap.parse_args()
    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = Path(__file__).resolve().parent / video_path
    watch_video(video_path, every_n=args.every, threshold=args.threshold)