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
    """Download Phase 5 model if not present"""
    
    # Define paths
    project_root = Path(__file__).parent.parent
    model_dir = project_root / "models" / "phase5"
    model_path = model_dir / "phase5_best_model.pth"
    
    # Check if model already exists
    if model_path.exists():
        print(f"✅ Model already exists at: {model_path}")
        print(f"📊 Model size: {model_path.stat().st_size / (1024*1024):.1f} MB")
        return str(model_path)
    
    # Get model URL from environment, fallback to default if provided
    model_url = os.environ.get("MODEL_DOWNLOAD_URL")
    if not model_url:
        # Fallback to known public Google Drive link (provided by user)
        DEFAULT_FILE_ID = "1a6JVuSIGxlKTX1ejyDXinsr8AScODPf5"
        model_url = f"https://drive.google.com/uc?export=download&id={DEFAULT_FILE_ID}"
        print("⚠️  MODEL_DOWNLOAD_URL not set; using default Google Drive link")
        print(f"🔗 {model_url}")
    
    print(f"📥 Downloading model from: {model_url}")
    
    try:
        # Create model directory
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Handle Google Drive downloads (they can redirect)
        session = requests.Session()
        
        # First request to get any redirect
        print("📡 Making initial request...")
        response = session.get(model_url, stream=True, timeout=600)
        
        # Google Drive sometimes serves HTML warning page for large files
        if 'text/html' in response.headers.get('content-type', ''):
            print("⚠️  Detected HTML response, looking for direct download link...")
            # Try to find the confirmation link
            content = response.text
            if 'confirm=' in content:
                import re
                # Extract the confirmation token
                confirm_match = re.search(r'confirm=([^&]+)', content)
                if confirm_match:
                    confirm_token = confirm_match.group(1)
                    # Build new URL with confirmation
                    if 'id=' in model_url:
                        file_id = model_url.split('id=')[1]
                        confirm_url = f"https://drive.google.com/uc?export=download&confirm={confirm_token}&id={file_id}"
                        print(f"🔄 Retrying with confirmation: {confirm_url[:80]}...")
                        response = session.get(confirm_url, stream=True, timeout=600)
        
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
        
        print(f"\n✅ Model downloaded successfully!")
        print(f"📊 Final size: {model_path.stat().st_size / (1024*1024):.1f} MB")
        return str(model_path)
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Download failed: {e}")
        # Clean up partial download
        if model_path.exists():
            model_path.unlink()
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
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