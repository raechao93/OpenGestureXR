"""Tests for the training pipeline (data loading and model export)."""

import csv
import tempfile
import os
import pytest
import numpy as np

from ai_engine.gesture_classifier import GESTURES
from ai_engine.training.train import load_csvs
from ai_engine.training.export_onnx import GestureNet


class TestLoadCSVs:
    def test_loads_valid_data(self, tmp_path):
        header = [f"{a}{i}" for i in range(21) for a in ("x", "y", "z")] + ["label"]
        csv_path = tmp_path / "grab.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerow([0.1] * 63 + ["grab"])
            w.writerow([0.2] * 63 + ["open_hand"])
        X, y = load_csvs(str(tmp_path))
        assert X.shape == (2, 63)
        assert len(y) == 2

    def test_skips_unknown_labels(self, tmp_path):
        header = [f"{a}{i}" for i in range(21) for a in ("x", "y", "z")] + ["label"]
        csv_path = tmp_path / "bad.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerow([0.1] * 63 + ["unknown_gesture"])
        X, y = load_csvs(str(tmp_path))
        assert len(X) == 0

    def test_empty_dir(self, tmp_path):
        X, y = load_csvs(str(tmp_path))
        assert len(X) == 0


class TestGestureNet:
    def test_forward_shape(self):
        model = GestureNet(num_classes=6)
        x = __import__("torch").randn(4, 63)
        out = model(x)
        assert out.shape == (4, 6)

    def test_export_onnx(self, tmp_path):
        onnx = pytest.importorskip("onnx")
        from ai_engine.training.export_onnx import export_onnx
        model = GestureNet(num_classes=6)
        path = str(tmp_path / "test.onnx")
        export_onnx(model, path)
        assert os.path.exists(path)
