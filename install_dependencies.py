#!/usr/bin/env python3
"""
Daily Activities Video Recognition System - Setup Script
========================================================

This script automates the installation of all required dependencies.
It works on both Windows and Linux systems.

Usage:
    python install_dependencies.py [--dev] [--skip-torch]

Options:
    --dev: Install development dependencies (pytest, black, etc.)
    --skip-torch: Skip PyTorch installation (if you have CUDA version or custom install)
    --help: Show this help message
"""

import os
import sys
import subprocess
import platform
import argparse
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8+ is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    else:
        print(f"✅ Python version: {sys.version.split()[0]}")


def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {description}:")
        print(f"Command: {command}")
        print(f"Error: {e.stderr}")
        return False


def check_gpu_support():
    """Check if CUDA is available for PyTorch"""
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected - PyTorch with CUDA support recommended")
            return True
        else:
            print("ℹ️  No NVIDIA GPU detected - CPU-only PyTorch will be installed")
            return False
    except FileNotFoundError:
        print("ℹ️  No NVIDIA drivers found - CPU-only PyTorch will be installed")
        return False


def install_pytorch(skip_torch=False, has_gpu=False):
    """Install PyTorch with appropriate version"""
    if skip_torch:
        print("⏭️  Skipping PyTorch installation")
        return True
    
    if has_gpu:
        # Install CUDA version
        torch_command = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
        return run_command(torch_command, "Installing PyTorch with CUDA support")
    else:
        # Install CPU version
        torch_command = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
        return run_command(torch_command, "Installing PyTorch (CPU-only)")


def install_requirements(include_dev=False):
    """Install requirements from requirements.txt"""
    req_file = Path("requirements.txt")
    if not req_file.exists():
        print("❌ Error: requirements.txt not found")
        return False
    
    # First install without PyTorch/ML packages
    basic_packages = [
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "requests>=2.31.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0"
    ]
    
    basic_install = f"pip install {' '.join(basic_packages)}"
    if not run_command(basic_install, "Installing basic dependencies"):
        return False
    
    # Install remaining packages
    other_packages = [
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "h5py>=3.8.0",
        "Pillow>=10.0.0"
    ]
    
    other_install = f"pip install {' '.join(other_packages)}"
    if not run_command(other_install, "Installing data processing libraries"):
        return False
    
    if include_dev:
        dev_packages = [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "jupyter>=1.0.0"
        ]
        dev_install = f"pip install {' '.join(dev_packages)}"
        run_command(dev_install, "Installing development tools")
    
    return True


def create_directories():
    """Create necessary directories for the project"""
    directories = [
        "outputs",
        "outputs/checkpoints",
        "outputs/metrics", 
        "outputs/inference_results",
        "data",
        "models"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        # Create .gitkeep file to preserve directory structure
        gitkeep_file = Path(dir_path) / ".gitkeep"
        if not gitkeep_file.exists():
            gitkeep_file.touch()
    
    print("✅ Created project directories")


def create_env_file():
    """Create .env file from .env.example if it doesn't exist"""
    env_example = Path(".env.example")
    env_file = Path(".env")
    
    if env_example.exists() and not env_file.exists():
        import shutil
        shutil.copy(env_example, env_file)
        print("✅ Created .env file from .env.example")
        print("ℹ️  Please edit .env file with your specific configuration")
    else:
        print("ℹ️  .env file already exists or .env.example not found")


def main():
    parser = argparse.ArgumentParser(description="Install dependencies for Daily Activities Recognition System")
    parser.add_argument("--dev", action="store_true", help="Install development dependencies")
    parser.add_argument("--skip-torch", action="store_true", help="Skip PyTorch installation")
    args = parser.parse_args()

    print("🚀 Daily Activities Video Recognition System - Setup")
    print("=" * 55)
    
    # Check system requirements
    check_python_version()
    print(f"✅ Operating System: {platform.system()} {platform.release()}")
    
    # Upgrade pip first
    if not run_command("python -m pip install --upgrade pip", "Upgrading pip"):
        print("⚠️  Warning: Could not upgrade pip, continuing anyway...")
    
    # Check GPU support
    has_gpu = check_gpu_support()
    
    # Install PyTorch first (if not skipped)
    if not install_pytorch(skip_torch=args.skip_torch, has_gpu=has_gpu):
        print("❌ PyTorch installation failed")
        return False
    
    # Install other requirements
    if not install_requirements(include_dev=args.dev):
        print("❌ Requirements installation failed")
        return False
    
    # Create project directories
    create_directories()
    
    # Create .env file
    create_env_file()
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next Steps:")
    print("1. Edit .env file with your configuration")
    print("2. Test the installation:")
    print("   python -c \"import torch, cv2, fastapi; print('All imports successful!')\"")
    print("3. Run the backend: python backend/main.py")
    print("4. Run inference: python src/phase5_inference.py --help")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)