#!/usr/bin/env python3
"""
Phase 5 FastAPI Server Launcher
===============================

Starts the FastAPI backend server for Phase 5 Activity Recognition.
Ensures the model path is set correctly and handles startup validation.
"""

print("🔧 [DEBUG] Script started, importing modules...")

import os
import sys
from pathlib import Path

print(f"🔧 [DEBUG] Basic imports successful. Python: {sys.version_info.major}.{sys.version_info.minor}")

def main():
    print("🔧 [DEBUG] Entering main() function...")
    
    # Resolve important paths
    backend_dir = Path(__file__).resolve().parent
    project_root = backend_dir.parent
    print(f"🔧 [DEBUG] Paths resolved. Backend: {backend_dir}, Project: {project_root}")

    # Ensure we run from the backend directory so imports like "main:app" work
    os.chdir(backend_dir)
    print(f"🔧 [DEBUG] Changed to backend directory: {os.getcwd()}")

    # Debug: Check Python and environment (kept minimal)
    print(f"🐍 Python version: {sys.version}")
    print(f"📁 Working directory: {os.getcwd()}")

    # Local-only: set default model path if not provided
    if "PHASE5_MODEL_PATH" not in os.environ:
        local_model = project_root / "models" / "phase5" / "phase5_best_model.pth"
        os.environ["PHASE5_MODEL_PATH"] = str(local_model)
        print(f"ℹ️  PHASE5_MODEL_PATH not set, defaulting to: {local_model}")

    # Ensure backend is on sys.path
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))
    
    # Import and run uvicorn (local dev)
    try:
        import uvicorn
    except ImportError as e:
        print(f"❌ uvicorn not installed: {e}")
        print("Install with: pip install -r backend/requirements.txt")
        return 1

    port = 8000
    print("🚀 Starting Phase 5 API Server (local)...")
    print(f"📍 http://127.0.0.1:{port}")
    print(f"📖 Docs: http://127.0.0.1:{port}/docs")

    try:
        uvicorn.run(
            "backend.main:app",
            host="127.0.0.1",
            port=port,
            reload=True,
            log_level="info",
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
        return 0

if __name__ == "__main__":
    print("🔧 [DEBUG] Script called directly, starting main()...")
    try:
        sys.exit(main())
    except Exception as e:
        print(f"🔧 [DEBUG] FATAL ERROR in main(): {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
