"""
Gesture API server.

Runs MediaPipe detection in a background thread and exposes results
via WebSocket (preferred) and REST (for simpler clients).

    uvicorn gesture_api.server.main:app --reload
"""

import asyncio
import threading
import time
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from ai_engine.gesture_detector import detect_hands, create_detector
from ai_engine.gesture_classifier import classify_gesture

app = FastAPI(title="OpenGestureXR API", version="0.2.0")

_state = {"gesture": "none", "confidence": 0.0, "hands": [], "timestamp": 0.0}
_lock = threading.Lock()
_ws_clients: set[WebSocket] = set()


def _detection_loop():
    cap = cv2.VideoCapture(0)
    detector = create_detector(max_hands=2)

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

        # first detected hand is "primary" for the simple /gesture endpoint
        if hand_results:
            primary = hand_results[0]
        else:
            primary = {"gesture": "none", "confidence": 0.0}

        with _lock:
            _state["gesture"] = primary["gesture"]
            _state["confidence"] = primary["confidence"]
            _state["hands"] = hand_results
            _state["timestamp"] = time.time()

    cap.release()


@app.on_event("startup")
def startup():
    t = threading.Thread(target=_detection_loop, daemon=True)
    t.start()


# -- websocket stream --

@app.websocket("/ws/gesture")
async def ws_gesture(ws: WebSocket):
    await ws.accept()
    _ws_clients.add(ws)
    try:
        while True:
            with _lock:
                data = dict(_state)
            await ws.send_json(data)
            await asyncio.sleep(0.033)  # roughly 30fps
    except WebSocketDisconnect:
        pass
    finally:
        _ws_clients.discard(ws)


# -- REST endpoints --

class GestureResponse(BaseModel):
    gesture: str
    confidence: float

class MultiHandResponse(BaseModel):
    gesture: str
    confidence: float
    hands: list[dict]
    timestamp: float


@app.get("/gesture", response_model=GestureResponse)
def get_gesture():
    with _lock:
        return GestureResponse(gesture=_state["gesture"],
                               confidence=_state["confidence"])

@app.get("/gesture/multi", response_model=MultiHandResponse)
def get_gesture_multi():
    with _lock:
        return MultiHandResponse(**_state)

@app.get("/health")
def health():
    return {"status": "ok"}
