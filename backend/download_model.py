#!/usr/bin/env python3
"""
Model Download Script for Render Deployment
==========================================

Downloads the Phase 5 model from external storage if not present.
This avoids committing large model files to GitHub.
"""

import os
import sys
import requests
from pathlib import Path
import tempfile

def download_model():
    """Optional utility: Download Phase 5 model if not present"""
    
    # Define paths
    project_root = Path(__file__).parent.parent
    model_dir = project_root / "models" / "phase5"
    model_path = model_dir / "phase5_best_model.pth"
    
    # Check if model already exists
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024*1024)
        print(f"✅ Model already exists at: {model_path}")
        print(f"📊 Model size: {size_mb:.1f} MB")
        return str(model_path)
    
    # Require explicit URL (no defaults)
    model_url = os.environ.get("MODEL_DOWNLOAD_URL")
    if not model_url:
        print("❌ MODEL_DOWNLOAD_URL not set")
        print("Set MODEL_DOWNLOAD_URL environment variable to download model")
        return None
    
    print(f"📥 Downloading model from: {model_url}")
    
    try:
        # Create model directory
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Simple download with requests
        response = requests.get(model_url, stream=True, timeout=600)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r📥 Downloading: {percent:.1f}% ({downloaded/(1024*1024):.1f}/{total_size/(1024*1024):.1f} MB)", end="")
        
        size_mb = model_path.stat().st_size / (1024*1024)
        print(f"\n📊 Final size: {size_mb:.1f} MB")
        print(f"✅ Model downloaded successfully!")
        return str(model_path)
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        if model_path.exists():
            model_path.unlink()
        return None

def main():
    """Main function"""
    print("🚀 Phase 5 Model Download Script")
    print("=" * 40)
    
    model_path = download_model()
    if model_path:
        print(f"✅ Model ready at: {model_path}")
        return 0
    else:
        print("❌ Model download failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())