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
    # Ensure we're running from the project root context
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Debug: Check Python and environment
    print(f"🐍 Python version: {sys.version}")
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"📦 Python path: {sys.path[:3]}...")  # First 3 entries
    
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
                # Add backend directory to path for absolute import
                backend_dir = Path(__file__).parent
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
        
        uvicorn.run(
            "backend.main:app", 
            host="0.0.0.0", 
            port=port, 
            reload=False,  # Disable reload in production
            log_level="info"
        )
    except ImportError as e:
        print(f"❌ uvicorn not installed: {e}")
        print("💡 This usually means:")
        print("   1. requirements.txt wasn't installed properly")
        print("   2. Python is using wrong environment")
        print("   3. Build failed silently")
        print("🔧 Solution: Check build logs and ensure 'pip install -r requirements.txt' succeeded")
        return 1
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
        return 0

if __name__ == "__main__":
    sys.exit(main())