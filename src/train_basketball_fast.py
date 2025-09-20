"""
Fast Basketball Binary Classifier Training Script
⚡ NO HANGING VERSION - Optimized for immediate training!

This skips the slow sample counting and gets straight to training.
Expected: 95%+ accuracy in 1-2 hours on RTX 3050!
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import time
from datetime import datetime

from binary_dataset_fast import get_fast_binary_dataloaders
from binary_model import get_binary_model, BinaryModelTrainer, save_binary_model

def train_basketball_binary_classifier_fast():
    """
    FAST Basketball Binary Classifier Training - No Hanging!
    """
    
    print("⚡" * 25)
    print("⚡ FAST BASKETBALL BINARY CLASSIFIER")  
    print("⚡ Individual Action Mastery - Phase 1")
    print("⚡ NO HANGING VERSION!")
    print("⚡" * 25)
    
    # Configuration
    config = {
        'data_root': r"C:\ASH_PROJECT\data\UCF101",
        'annotation_path': r"C:\ASH_PROJECT\data\ucfTrainTestlist",
        'target_action': 'Basketball',
        'model_type': 'simple',
        'batch_size': 8,
        'num_epochs': 10,
        'save_dir': r"C:\ASH_PROJECT\outputs\binary_models"
    }
    
    print(f"🎯 Target Action: {config['target_action']}")
    print(f"📊 Batch Size: {config['batch_size']}")
    print(f"📈 Epochs: {config['num_epochs']}")
    
    # Check GPU
    if not torch.cuda.is_available():
        print("⚠️ GPU not available - using CPU (will be slower)")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🚀 Using GPU: {gpu_name}")
    
    # Create FAST binary datasets (no hanging!)
    print("\n⚡ Loading FAST Binary Classification Data...")
    train_loader, test_loader, class_weights = get_fast_binary_dataloaders(
        config['data_root'], 
        config['annotation_path'],
        target_action=config['target_action'],
        batch_size=config['batch_size'],
        num_workers=0
    )
    
    print(f"✅ Fast data loading completed!")
    
    # Create binary model
    print(f"\n🔧 Creating Binary Model...")
    model = get_binary_model(config['model_type']).to(device)
    
    # Create trainer
    trainer = BinaryModelTrainer(model, device, class_weights)
    
    # Training tracking
    best_test_acc = 0
    
    print(f"\n🚀 Starting FAST Training...")
    print("=" * 60)
    
    start_time = time.time()
    
    for epoch in range(1, config['num_epochs'] + 1):
        epoch_start = time.time()
        
        print(f"\n📈 Epoch {epoch}/{config['num_epochs']}")
        print("-" * 40)
        
        # Training with progress bar
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}")
        for clips, labels in pbar:
            clips, labels = clips.to(device), labels.to(device)
            
            trainer.optimizer.zero_grad()
            outputs = model(clips)
            loss = trainer.criterion(outputs, labels)
            loss.backward()
            trainer.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Acc': f"{100.*correct/total:.2f}%"
            })
        
        train_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Validation
        print("🧪 Validating...")
        test_loss, test_acc, precision, recall, f1 = trainer.validate_epoch(test_loader)
        
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
        
        # Check if target achieved
        if test_acc >= 95.0:
            print(f"\n🎉 TARGET ACHIEVED! {test_acc:.2f}% >= 95%")
            print("🏀 Basketball binary classifier MASTERED!")
            break
            
        # Early success check
        if test_acc >= 90.0 and epoch >= 5:
            print(f"\n✅ Excellent progress! {test_acc:.2f}% accuracy")
    
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
    elif best_test_acc >= 90.0:
        print("🎯 Phase 1: Basketball nearly mastered!")
        print("💡 Consider a few more epochs for 95%+")
    else:
        print(f"⚠️ Phase 1: Basketball accuracy {best_test_acc:.2f}%")
        print("💡 May need more training or enhanced model")
    
    return {
        'best_accuracy': best_test_acc,
        'training_time': total_time
    }

def main():
    """Main execution"""
    
    try:
        # Train basketball binary classifier
        results = train_basketball_binary_classifier_fast()
        
        # Success guidance
        if results['best_accuracy'] >= 90:
            print(f"\n🎉 SUCCESS! Basketball classifier achieved {results['best_accuracy']:.2f}% accuracy")
            print("\n🚀 NEXT STEPS:")
            print("1. ✅ Basketball mastery proven!")
            print("2. 🎾 Train tennis binary classifier")
            print("3. ⛳ Train golf binary classifier") 
            print("4. 🏟️ Combine into sports category")
            
        print(f"\n🏀 Basketball Individual Action Mastery: {'✅ COMPLETE' if results['best_accuracy'] >= 95 else '⚡ IN PROGRESS'}")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print("💡 Check error details above")

if __name__ == "__main__":
    main()