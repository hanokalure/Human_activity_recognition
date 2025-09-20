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
                from .download_model import download_model
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
        print("🚀 Starting Phase 5 API Server...")
        print("📍 Server will be available at: http://localhost:8000")
        print("📖 API docs: http://localhost:8000/docs")
        print("🔄 Press Ctrl+C to stop")
        
        uvicorn.run(
            "backend.main:app", 
            host="0.0.0.0", 
            port=8000, 
            reload=True,
            log_level="info"
        )
    except ImportError:
        print("❌ uvicorn not installed. Install with: pip install uvicorn[standard]")
        return 1
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
        return 0

if __name__ == "__main__":
    sys.exit(main())