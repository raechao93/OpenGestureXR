"""
ONNX Runtime inference wrapper.

Loads an exported .onnx gesture model and runs classification.
Optionally tries TensorRT/CUDA providers if available, otherwise CPU.
"""

import numpy as np
import onnxruntime as ort
from ai_engine.gesture_classifier import GESTURES


class ONNXGestureClassifier:
    def __init__(self, model_path: str, use_gpu=False):
        if use_gpu:
            # try tensorrt first, then cuda, then fall back to cpu
            providers = ["TensorrtExecutionProvider",
                         "CUDAExecutionProvider",
                         "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def classify(self, landmarks: list[list[float]]) -> dict:
        flat = np.array(landmarks, dtype=np.float32).flatten().reshape(1, -1)
        output = self.session.run(None, {self.input_name: flat})
        probs = output[0][0]
        idx = int(np.argmax(probs))
        return {"gesture": GESTURES[idx], "confidence": float(probs[idx])}
