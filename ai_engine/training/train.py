"""
Train gesture classifier from collected CSV data and export to ONNX.

    python -m ai_engine.training.train --data-dir data/ --epochs 50
"""

import argparse
import glob
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ai_engine.training.export_onnx import GestureNet, export_onnx
from ai_engine.gesture_classifier import GESTURES


def load_csvs(data_dir):
    X, y = [], []
    label_to_idx = {g: i for i, g in enumerate(GESTURES)}

    for path in glob.glob(f"{data_dir}/*.csv"):
        with open(path) as f:
            reader = csv.reader(f)
            next(reader)  # header
            for row in reader:
                label = row[-1]
                if label not in label_to_idx:
                    continue
                X.append([float(v) for v in row[:-1]])
                y.append(label_to_idx[label])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def train(data_dir, epochs, output):
    X, y = load_csvs(data_dir)
    if len(X) == 0:
        print("no data found — run collect_data.py first")
        return

    print(f"loaded {len(X)} samples across {len(set(y))} classes")

    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    model = GestureNet(num_classes=len(GESTURES))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item()

        if (epoch + 1) % 10 == 0:
            avg = total / len(loader)
            print(f"  epoch {epoch+1}/{epochs}  loss={avg:.4f}")

    # save both pytorch checkpoint and onnx
    pt_path = output.replace(".onnx", ".pt")
    torch.save(model.state_dict(), pt_path)
    export_onnx(model, output)
    print("done")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--output", default="ai_engine/models/gesture_classifier.onnx")
    args = p.parse_args()
    train(args.data_dir, args.epochs, args.output)
