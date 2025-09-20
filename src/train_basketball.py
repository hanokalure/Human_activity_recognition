"""
Basketball Binary Classifier Training Script
🏀 PROOF OF CONCEPT: Individual Action Mastery 

This is Step 1 of the Evolution Strategy:
1. ✅ Master Basketball (95%+ accuracy) ← YOU ARE HERE
2. ⏳ Scale to other sports (Tennis, Golf, etc.)
3. ⏳ Combine into Sports category
4. ⏳ Build confidence-based frontend

Expected Results:
- Training Time: 1-2 hours
- Target Accuracy: 95%+ 
- Much easier than 101-class classification!
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import time
from datetime import datetime

from binary_dataset import get_binary_dataloaders
from binary_model import get_binary_model, BinaryModelTrainer, save_binary_model

def train_basketball_binary_classifier():
    """
    Train ONLY basketball vs non-basketball classifier
    This proves the Evolution Strategy works!
    """
    
    print("🏀" * 20)
    print("🏀 BASKETBALL BINARY CLASSIFIER TRAINING")  
    print("🏀 Individual Action Mastery - Phase 1")
    print("🏀" * 20)
    
    # Configuration
    config = {
        'data_root': r"C:\ASH_PROJECT\data\UCF101",
        'annotation_path': r"C:\ASH_PROJECT\data\ucfTrainTestlist",
        'target_action': 'Basketball',
        'model_type': 'simple',  # Start with simple model
        'batch_size': 8,         # Smaller batch for stability
        'num_epochs': 10,        # Should be enough for 95%+
        'save_dir': r"C:\ASH_PROJECT\outputs\binary_models"
    }
    
    print(f"🎯 Target Action: {config['target_action']}")
    print(f"📊 Batch Size: {config['batch_size']}")
    print(f"📈 Epochs: {config['num_epochs']}")
    print(f"🏗️ Model Type: {config['model_type']}")
    
    # Check GPU availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for training. GPU not available.")
    
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"🚀 Using GPU: {gpu_name}")
    
    # Create binary datasets
    print("\n📂 Loading Binary Classification Data...")
    train_loader, test_loader, class_weights = get_binary_dataloaders(
        config['data_root'], 
        config['annotation_path'],
        target_action=config['target_action'],
        batch_size=config['batch_size'],
        num_workers=0  # Windows compatibility
    )
    
    print(f"✅ Data loaded successfully!")
    print(f"   📊 Training batches: {len(train_loader)}")
    print(f"   📊 Test batches: {len(test_loader)}")
    
    # Create binary model
    print("\n🔧 Creating Binary Classification Model...")
    model = get_binary_model(config['model_type']).to(device)
    print(f"✅ Model created and moved to GPU")
    
    # Create trainer
    trainer = BinaryModelTrainer(model, device, class_weights)
    
    # Training tracking
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    best_test_acc = 0
    
    print(f"\n🚀 Starting Training...")
    print("=" * 60)
    
    start_time = time.time()
    
    for epoch in range(1, config['num_epochs'] + 1):
        epoch_start = time.time()
        
        print(f"\n📈 Epoch {epoch}/{config['num_epochs']}")
        print("-" * 40)
        
        # Training
        print("🏋️ Training...")
        train_loss, train_acc = trainer.train_epoch(train_loader)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Validation
        print("🧪 Validating...")
        test_loss, test_acc, precision, recall, f1 = trainer.validate_epoch(test_loader)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start
        
        # Print results
        print(f"📊 Epoch {epoch} Results:")
        print(f"   🏋️ Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"   🧪 Test:  Loss={test_loss:.4f}, Acc={test_acc:.2f}%")
        print(f"   🎯 Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        print(f"   ⏱️ Time: {epoch_time:.1f}s")
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            print(f"   🏆 New best accuracy: {best_test_acc:.2f}%")
            save_binary_model(model, config['target_action'], config['save_dir'])
        
        # Check if we've achieved target accuracy
        if test_acc >= 95.0:
            print(f"\n🎉 TARGET ACHIEVED! {test_acc:.2f}% >= 95%")
            print("🏀 Basketball binary classifier mastered!")
            break
    
    # Training summary
    total_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("🏆 TRAINING COMPLETED!")
    print("=" * 60)
    print(f"🏀 Action: {config['target_action']}")
    print(f"🎯 Best Test Accuracy: {best_test_acc:.2f}%")
    print(f"⏱️ Total Training Time: {total_time/60:.1f} minutes")
    print(f"💾 Model saved to: {config['save_dir']}")
    
    # Evolution Strategy Status
    print("\n🚀 EVOLUTION STRATEGY STATUS:")
    if best_test_acc >= 95.0:
        print("✅ Phase 1: Basketball mastery COMPLETE!")
        print("🎯 Ready for Phase 2: Scale to other sports")
        print("   - Tennis binary classifier")
        print("   - Golf binary classifier") 
        print("   - Soccer binary classifier")
        print("   - etc...")
    else:
        print(f"⚠️ Phase 1: Basketball accuracy {best_test_acc:.2f}% < 95%")
        print("💡 Consider: More epochs, enhanced model, or data augmentation")
    
    return {
        'best_accuracy': best_test_acc,
        'training_time': total_time,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies
    }

def quick_test_basketball_model():
    """Quick test of trained basketball model"""
    
    print("\n🧪 QUICK TEST: Basketball Model")
    print("-" * 40)
    
    try:
        from binary_model import load_binary_model
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_binary_model("Basketball").to(device)
        model.eval()
        
        # Test with dummy data
        dummy_clip = torch.randn(1, 3, 16, 112, 112).to(device)
        
        with torch.no_grad():
            output = model(dummy_clip)
            prob = torch.softmax(output, dim=1)
            pred_class = output.argmax(1).item()
            confidence = prob.max().item()
        
        action_name = "Basketball" if pred_class == 1 else "Not Basketball"
        print(f"🎯 Prediction: {action_name}")
        print(f"🎲 Confidence: {confidence:.3f}")
        print("✅ Model test successful!")
        
    except Exception as e:
        print(f"⚠️ Model test failed: {e}")
        print("💡 Train the model first!")

def main():
    """Main execution"""
    
    try:
        # Train basketball binary classifier
        results = train_basketball_binary_classifier()
        
        # Quick test
        if results['best_accuracy'] >= 90:  # If reasonably good
            quick_test_basketball_model()
        
        # Next steps guidance
        print("\n🚀 NEXT STEPS:")
        print("1. 🧪 Run: python train_basketball.py  # This script")
        print("2. 🎾 Train tennis: python train_binary.py --action TennisSwing")
        print("3. ⛳ Train golf: python train_binary.py --action GolfSwing") 
        print("4. 🏟️ Combine into sports category")
        print("5. 🎯 Build confidence frontend")
        
        print(f"\n🏀 Basketball Individual Action Mastery: {'✅ COMPLETE' if results['best_accuracy'] >= 95 else '⏳ IN PROGRESS'}")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print("💡 Check GPU availability and data paths")

if __name__ == "__main__":
    main()