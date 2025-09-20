#!/usr/bin/env python3
"""
Phase 5 FastAPI Server Launcher
===============================

Starts the FastAPI backend server for Phase 5 Activity Recognition.
Ensures the model path is set correctly and handles startup validation.
"""

import os
import sys
from pathlib import Path

def main():
    # Resolve important paths
    backend_dir = Path(__file__).resolve().parent
    project_root = backend_dir.parent

    # Ensure we run from the backend directory so imports like "main:app" work
    os.chdir(backend_dir)

    # Debug: Check Python and environment
    print(f"🐍 Python version: {sys.version}")
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"📦 Python path (first entries): {sys.path[:5]}...")  # First entries
    
    # Set model path if not already set
    if "PHASE5_MODEL_PATH" not in os.environ:
        model_path = project_root / "models" / "phase5" / "phase5_best_model.pth"
        
        if model_path.exists():
            os.environ["PHASE5_MODEL_PATH"] = str(model_path)
            print(f"✅ Using model: {model_path}")
        else:
            print(f"⚠️  Model not found at {model_path}")
            print(f"📥 Attempting to download model...")
            
            # Try to download model
            try:
                # backend_dir already is CWD, still ensure it's on sys.path
                if str(backend_dir) not in sys.path:
                    sys.path.insert(0, str(backend_dir))

                from download_model import download_model
                downloaded_path = download_model()
                
                if downloaded_path:
                    os.environ["PHASE5_MODEL_PATH"] = downloaded_path
                    print(f"✅ Using downloaded model: {downloaded_path}")
                else:
                    print(f"❌ Model download failed")
                    print(f"💡 Set MODEL_DOWNLOAD_URL environment variable")
                    return 1
            except ImportError as e:
                print(f"❌ Could not import download_model: {e}")
                return 1
    
    # Import and run uvicorn
    try:
        import uvicorn
    except ImportError as e:
        print(f"❌ uvicorn not installed: {e}")
        print("💡 This usually means:")
        print("   1. requirements.txt wasn't installed properly")
        print("   2. Python is using wrong environment")
        print("   3. Build failed silently")
        print("🔧 Solution: Check build logs and ensure 'pip install -r requirements.txt' succeeded")
        return 1

    # Resolve the port robustly. Render sets $PORT; if it's missing or empty, default to 10000.
    port_str = os.environ.get("PORT") or "10000"
    try:
        port = int(port_str)
    except (TypeError, ValueError):
        port = 10000

    print("🚀 Starting Phase 5 API Server...")
    print(f"📍 Server will be available at: http://localhost:{port}")
    print(f"📖 API docs: http://localhost:{port}/docs")
    print("🔄 Press Ctrl+C to stop")

    try:
        # Since CWD is backend/, use module path without package prefix
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=port,
            reload=False,  # Disable reload in production
            log_level="info",
        )
    except ModuleNotFoundError as e:
        print(f"❌ Failed to import app module: {e}")
        print("🔍 Tips:")
        print("   - Ensure you're in the backend directory (we are: see Working directory log)")
        print("   - Ensure main.py exists and defines 'app' (FastAPI instance)")
        print("   - If using packages, ensure correct PYTHONPATH")
        return 1
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
        return 0

if __name__ == "__main__":
    sys.exit(main())