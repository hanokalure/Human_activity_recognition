"""
Quick Setup Test for Basketball Binary Classifier
🧪 This tests if everything is ready before full training
"""

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        import torch
        import torchvision
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ TorchVision: {torchvision.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA: {torch.version.cuda}")
            print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️ CUDA: Not available (will use CPU - slower training)")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    try:
        from binary_dataset import get_binary_dataloaders
        from binary_model import get_binary_model
        print("✅ Custom modules imported successfully")
    except ImportError as e:
        print(f"❌ Custom module error: {e}")
        return False
    
    return True

def test_data_paths():
    """Test if data paths exist"""
    print("\n📂 Testing data paths...")
    
    import os
    
    data_root = r"C:\ASH_PROJECT\data\UCF101"
    annotation_path = r"C:\ASH_PROJECT\data\ucfTrainTestlist"
    
    if os.path.exists(data_root):
        print("✅ UCF101 data directory found")
        # Count some directories
        dirs = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
        print(f"   📁 Found {len(dirs)} action classes")
        
        # Check if Basketball exists
        if "Basketball" in dirs:
            print("✅ Basketball class found")
            basket_path = os.path.join(data_root, "Basketball")
            videos = [f for f in os.listdir(basket_path) if f.endswith(('.avi', '.mp4'))]
            print(f"   🏀 Basketball videos: {len(videos)}")
        else:
            print("⚠️ Basketball class not found")
            
    else:
        print(f"❌ UCF101 data not found at: {data_root}")
        return False
    
    if os.path.exists(annotation_path):
        print("✅ Annotation path found")
    else:
        print(f"❌ Annotations not found at: {annotation_path}")
        return False
    
    return True

def test_model_creation():
    """Test if we can create a binary model"""
    print("\n🔧 Testing model creation...")
    
    try:
        from binary_model import get_binary_model
        
        model = get_binary_model("simple")
        print("✅ Binary model created successfully")
        
        # Test with dummy input
        import torch
        dummy_input = torch.randn(1, 3, 16, 112, 112)
        
        with torch.no_grad():
            output = model(dummy_input)
            print(f"✅ Model forward pass: {output.shape}")
            
            if output.shape == (1, 2):
                print("✅ Correct output shape for binary classification")
            else:
                print(f"⚠️ Expected (1, 2), got {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def test_dataset_loading():
    """Test if we can load a small batch of data"""
    print("\n📊 Testing dataset loading...")
    
    try:
        from binary_dataset import get_binary_dataloaders
        
        # Test with very small batch
        train_loader, test_loader, class_weights = get_binary_dataloaders(
            r"C:\ASH_PROJECT\data\UCF101",
            r"C:\ASH_PROJECT\data\ucfTrainTestlist", 
            target_action="Basketball",
            batch_size=2,  # Very small for testing
            num_workers=0
        )
        
        print("✅ DataLoaders created successfully")
        print(f"   📊 Training batches: {len(train_loader)}")
        print(f"   📊 Test batches: {len(test_loader)}")
        print(f"   ⚖️ Class weights: {class_weights}")
        
        # Try to load one batch
        print("\n🔍 Testing batch loading...")
        for clips, labels in train_loader:
            print(f"✅ First batch: clips={clips.shape}, labels={labels}")
            break
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return False

def main():
    """Run all tests"""
    
    print("🚀" * 30)
    print("🚀 BASKETBALL BINARY CLASSIFIER SETUP TEST")
    print("🚀" * 30)
    
    all_passed = True
    
    all_passed &= test_imports()
    all_passed &= test_data_paths() 
    all_passed &= test_model_creation()
    all_passed &= test_dataset_loading()
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Ready for basketball binary classifier training!")
        print("\n🚀 Next step: Run 'python train_basketball.py'")
    else:
        print("❌ SOME TESTS FAILED!")
        print("💡 Fix the issues above before training")
    print("=" * 50)

if __name__ == "__main__":
    main()