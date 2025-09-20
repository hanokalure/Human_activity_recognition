from __future__ import annotations

import os
import sys
import json
import tempfile
from pathlib import Path
from collections import deque
from typing import Deque, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Ensure we can import project modules
# Resolve project root dynamically (backend/.. = repo root)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    # Import from your existing inference module
    from phase5_inference import Phase5Predictor, VideoProcessor
except Exception as e:
    raise RuntimeError(f"Failed to import Phase 5 inference modules: {e}")

# Configuration
DEFAULT_MODEL_PATH = str(PROJECT_ROOT / "models" / "phase5" / "phase5_best_model.pth")
TMP_DIR = PROJECT_ROOT / "outputs" / "api_tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Phase5 Activity Recognition API", version="1.0.0")

# CORS for local web and Expo development
origins = [
    "*",  # During development; tighten later if needed
    "http://localhost:3000",            # CRA web
    "http://127.0.0.1:3000",
    "http://localhost:5173",            # Vite
    "http://127.0.0.1:5173",
    "http://localhost:19006",           # Expo web
    "http://127.0.0.1:19006",
    "exp://127.0.0.1:19000",            # Expo Go (LAN)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
predictor: Optional[Phase5Predictor] = None
video_processor = VideoProcessor()

@app.on_event("startup")
async def startup_event():
    global predictor
    model_path = os.environ.get("PHASE5_MODEL_PATH", DEFAULT_MODEL_PATH)
    mp = Path(model_path)
    if not mp.exists():
        raise RuntimeError(f"Model file not found at {mp}")
    predictor = Phase5Predictor(model_path)


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": predictor is not None}


@app.post("/predict/video")
async def predict_video(file: UploadFile = File(...)):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Save to a temporary file to be read by OpenCV
    suffix = Path(file.filename).suffix or ".mp4"
    try:
        with tempfile.NamedTemporaryFile(delete=False, dir=str(TMP_DIR), suffix=suffix) as tmp:
            data = await file.read()
            tmp.write(data)
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to persist upload: {e}")

    try:
        result = predictor.predict_video(tmp_path, return_probabilities=True)
        if result is None:
            raise HTTPException(status_code=400, detail="Failed to process video")
        return JSONResponse(content=result)
    finally:
        # Clean up tmp file
        try:
            os.remove(tmp_path)
        except Exception:
            pass


class StreamSession:
    def __init__(self, frames_per_clip: int = 16):
        self.buffer: Deque[np.ndarray] = deque(maxlen=frames_per_clip)
        self.frames_per_clip = frames_per_clip
        self.last_result: Optional[dict] = None

    def add_frame_bytes(self, data: bytes) -> bool:
        # Decode JPEG/PNG bytes to BGR frame
        arr = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return False
        self.buffer.append(frame)
        return True

    def ready(self) -> bool:
        return len(self.buffer) == self.frames_per_clip


@app.websocket("/ws/frames")
async def ws_frames(ws: WebSocket):
    if predictor is None:
        await ws.close(code=1011)
        return

    await ws.accept()
    session = StreamSession(frames_per_clip=16)

    # Protocol:
    # - Text messages are JSON control packets: {"type":"start"} or {"type":"end"}
    # - Binary messages are single image frames (JPEG/PNG). When 16 frames are present, a prediction is emitted.
    try:
        while True:
            msg = await ws.receive()
            if "bytes" in msg and msg["bytes"] is not None:
                ok = session.add_frame_bytes(msg["bytes"])
                if not ok:
                    await ws.send_text(json.dumps({"type": "error", "message": "Bad frame encoding"}))
                    continue
                if session.ready():
                    res = predictor.predict_frames(list(session.buffer), return_probabilities=True)
                    session.last_result = res
                    await ws.send_text(json.dumps({"type": "prediction", "data": res}))
            elif "text" in msg and msg["text"] is not None:
                try:
                    payload = json.loads(msg["text"]) if msg["text"] else {}
                except json.JSONDecodeError:
                    await ws.send_text(json.dumps({"type": "error", "message": "Invalid JSON"}))
                    continue
                mtype = payload.get("type")
                if mtype == "start":
                    session = StreamSession(frames_per_clip=16)
                    await ws.send_text(json.dumps({"type": "ack", "message": "started"}))
                elif mtype == "end":
                    # On end, return the last result if available and close
                    await ws.send_text(json.dumps({
                        "type": "final",
                        "data": session.last_result
                    }))
                    await ws.close()
                    return
                else:
                    await ws.send_text(json.dumps({"type": "ack"}))
            else:
                # Unknown message
                await ws.send_text(json.dumps({"type": "error", "message": "Unknown message"}))
    except WebSocketDisconnect:
        return
    except Exception as e:
        try:
            await ws.send_text(json.dumps({"type": "error", "message": f"Server error: {e}"}))
        except Exception:
            pass
        finally:
            await ws.close(code=1011)


# For local running with: python backend/main.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
