"""Tests for the FastAPI gesture server endpoints.

Mocks the ai_engine imports so we don't need mediapipe/webcam at test time.
"""

import sys
import types
import pytest
from unittest.mock import MagicMock


@pytest.fixture
def client():
    # Save and remove any cached modules that would cause mediapipe import
    saved = {}
    for mod_name in list(sys.modules):
        if mod_name.startswith(("gesture_api", "ai_engine")):
            saved[mod_name] = sys.modules.pop(mod_name)

    # Stub ai_engine modules
    mock_detector = types.ModuleType("ai_engine.gesture_detector")
    mock_detector.detect_hands = MagicMock(return_value=[])
    mock_detector.create_detector = MagicMock()
    mock_classifier = types.ModuleType("ai_engine.gesture_classifier")
    mock_classifier.classify_gesture = MagicMock(return_value={"gesture": "none", "confidence": 0.0})
    mock_classifier.GESTURES = ["open_hand", "grab", "pinch", "point", "thumbs_up", "peace"]

    # Also need the parent package stubs
    if "ai_engine" not in sys.modules:
        sys.modules["ai_engine"] = types.ModuleType("ai_engine")
    sys.modules["ai_engine.gesture_detector"] = mock_detector
    sys.modules["ai_engine.gesture_classifier"] = mock_classifier

    from fastapi.testclient import TestClient
    from gesture_api.server.main import app, _state, _lock

    with _lock:
        _state["gesture"] = "grab"
        _state["confidence"] = 0.87
        _state["hands"] = [{"handedness": "Right", "gesture": "grab", "confidence": 0.87}]
        _state["timestamp"] = 1000.0

    yield TestClient(app)

    # Restore original modules
    for mod_name in list(sys.modules):
        if mod_name.startswith(("gesture_api", "ai_engine")):
            del sys.modules[mod_name]
    sys.modules.update(saved)


class TestRESTEndpoints:
    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

    def test_get_gesture(self, client):
        r = client.get("/gesture")
        assert r.status_code == 200
        data = r.json()
        assert data["gesture"] == "grab"
        assert data["confidence"] == 0.87

    def test_get_gesture_multi(self, client):
        r = client.get("/gesture/multi")
        assert r.status_code == 200
        data = r.json()
        assert data["gesture"] == "grab"
        assert len(data["hands"]) == 1
        assert data["hands"][0]["handedness"] == "Right"


class TestWebSocket:
    def test_ws_receives_data(self, client):
        with client.websocket_connect("/ws/gesture") as ws:
            data = ws.receive_json()
            assert data["gesture"] == "grab"
            assert "confidence" in data
