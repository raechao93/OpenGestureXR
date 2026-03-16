"""
Export gesture classification model to ONNX.

    python -m ai_engine.training.export_onnx --output ai_engine/models/gesture_classifier.onnx
"""

import argparse
import torch
import torch.nn as nn


class GestureNet(nn.Module):
    """Simple MLP: 63 inputs (21 landmarks * 3 coords) -> gesture class."""

    def __init__(self, num_classes=6):
        super().__init__()
        # tried a bigger hidden layer (256) but it didn't help much
        # and doubled the onnx file size, so keeping it small
        self.net = nn.Sequential(
            nn.Linear(63, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def export_onnx(model, output_path):
    model.eval()
    dummy = torch.randn(1, 63)
    torch.onnx.export(
        model, dummy, output_path,
        input_names=["landmarks"],
        output_names=["gesture_probs"],
        dynamic_axes={"landmarks": {0: "batch"}, "gesture_probs": {0: "batch"}},
        opset_version=17,
    )
    print(f"exported to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="ai_engine/models/gesture_classifier.onnx")
    args = parser.parse_args()

    model = GestureNet()
    # load weights if you have them:
    # model.load_state_dict(torch.load("checkpoint.pt"))
    export_onnx(model, args.output)
