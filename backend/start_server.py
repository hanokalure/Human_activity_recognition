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

    # Debug: Check Python and environment
    print(f"🐍 Python version: {sys.version}")
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"📦 Python path (first entries): {sys.path[:5]}...")  # First entries
    
    # Set model path - handle both missing and relative paths
    print(f"🔧 [DEBUG] Current PHASE5_MODEL_PATH env var: {os.environ.get('PHASE5_MODEL_PATH', 'NOT SET')}")
    
    # Check if PHASE5_MODEL_PATH exists and is valid
    needs_download = True
    if "PHASE5_MODEL_PATH" in os.environ:
        env_path = Path(os.environ["PHASE5_MODEL_PATH"])
        if not env_path.is_absolute():
            # Convert relative path to absolute (relative to project root)
            env_path = project_root / env_path
        
        if env_path.exists():
            # Update env var with absolute path
            os.environ["PHASE5_MODEL_PATH"] = str(env_path)
            print(f"✅ Using existing model: {env_path}")
            needs_download = False
        else:
            print(f"⚠️  Model not found at {env_path} (from env var)")
    
    if needs_download:
        model_path = project_root / "models" / "phase5" / "phase5_best_model.pth"
        print(f"🔧 [DEBUG] Checking default model path: {model_path}")
        
        if model_path.exists():
            os.environ["PHASE5_MODEL_PATH"] = str(model_path)
            print(f"✅ Using default model: {model_path}")
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
                    # Ensure absolute path
                    abs_path = Path(downloaded_path)
                    if not abs_path.is_absolute():
                        abs_path = project_root / abs_path
                    os.environ["PHASE5_MODEL_PATH"] = str(abs_path)
                    print(f"✅ Using downloaded model: {abs_path}")
                else:
                    print(f"❌ Model download failed")
                    print(f"💡 Set MODEL_DOWNLOAD_URL environment variable")
                    return 1
            except ImportError as e:
                print(f"❌ Could not import download_model: {e}")
                return 1
            except Exception as e:
                print(f"❌ Unexpected error during model download: {e}")
                import traceback
                traceback.print_exc()
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
    print("🔧 [DEBUG] Script called directly, starting main()...")
    try:
        sys.exit(main())
    except Exception as e:
        print(f"🔧 [DEBUG] FATAL ERROR in main(): {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
