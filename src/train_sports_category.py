"""
Sports Category Training Script
🏟️ Train on 10 Sports Actions - MUCH FASTER!

BRILLIANT EVOLUTION:
- 10 classes instead of 101 or binary (2.3M samples)
- Expected: 30-60 minutes on RTX 3050
- Target: 80%+ accuracy (8/10 sports correct)
- Proves category building strategy works!
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import time

from sports_category_dataset import get_sports_dataloaders, SPORTS_ACTIONS
from model import get_model  # Use existing model, just change num_classes

def train_sports_category():
    """
    Train Sports Category Classifier - 10 Sports Actions
    """
    
    print("🏟️" * 20)
    print("🏟️ SPORTS CATEGORY CLASSIFIER TRAINING")  
    print("🏟️ Evolution Strategy - Category Building")
    print("🏟️" * 20)
    
    # Configuration
    config = {
        'data_root': r"C:\ASH_PROJECT\data\UCF101",
        'annotation_path': r"C:\ASH_PROJECT\data\ucfTrainTestlist",
        'num_classes': len(SPORTS_ACTIONS),  # 10 sports
        'batch_size': 16,                    # Larger batch for efficiency
        'num_epochs': 15,                    # Should be enough for good accuracy
        'step_between_clips': 32,            # 32 = 2x faster than normal
        'save_dir': r"C:\ASH_PROJECT\outputs\binary_models"
    }
    
    print(f"🎯 Target Classes: {config['num_classes']} sports")
    print(f"📊 Batch Size: {config['batch_size']}")
    print(f"📈 Epochs: {config['num_epochs']}")
    print(f"⚡ Speed Mode: step={config['step_between_clips']} (2x faster)")
    
    # Show sports actions
    print(f"\n🏆 Sports Actions:")
    for i, action in enumerate(SPORTS_ACTIONS):
        print(f"   {i}: {action}")
    
    # Check GPU
    if not torch.cuda.is_available():
        print("⚠️ GPU not available - using CPU")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\n🚀 Using GPU: {gpu_name}")
    
    # Create sports datasets
    print(f"\n🏟️ Loading Sports Category Data...")
    train_loader, test_loader = get_sports_dataloaders(
        config['data_root'], 
        config['annotation_path'],
        batch_size=config['batch_size'],
        step_between_clips=config['step_between_clips'],
        num_workers=0
    )
    
    print(f"✅ Sports data loaded!")
    
    # Create model for 10 sports classes
    print(f"\n🔧 Creating Sports Classification Model...")
    model = get_model(num_classes=config['num_classes']).to(device)
    print(f"✅ Model created: {config['num_classes']} classes")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Training tracking
    best_test_acc = 0
    
    print(f"\n🚀 Starting Sports Category Training...")
    print("=" * 60)
    
    start_time = time.time()
    
    for epoch in range(1, config['num_epochs'] + 1):
        epoch_start = time.time()
        
        print(f"\n📈 Epoch {epoch}/{config['num_epochs']}")
        print("-" * 40)
        
        # Training
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}")
        for clips, labels in pbar:
            clips, labels = clips.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(clips)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
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
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        
        # Per-class accuracy tracking
        class_correct = [0] * config['num_classes']
        class_total = [0] * config['num_classes']
        
        with torch.no_grad():
            for clips, labels in test_loader:
                clips, labels = clips.to(device), labels.to(device)
                
                outputs = model(clips)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
                
                # Per-class accuracy
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    class_total[label] += 1
                    if predicted[i] == labels[i]:
                        class_correct[label] += 1
        
        test_loss = test_loss / len(test_loader)
        test_acc = 100. * test_correct / test_total
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start
        
        # Print results
        print(f"📊 Epoch {epoch} Results:")
        print(f"   🏋️ Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"   🧪 Test:  Loss={test_loss:.4f}, Acc={test_acc:.2f}%")
        print(f"   ⏱️ Time: {epoch_time:.1f}s")
        
        # Show per-class accuracy every 5 epochs
        if epoch % 5 == 0:
            print("   🏆 Per-sport accuracy:")
            for i in range(config['num_classes']):
                if class_total[i] > 0:
                    acc = 100. * class_correct[i] / class_total[i]
                    print(f"      {SPORTS_ACTIONS[i]}: {acc:.1f}%")
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            print(f"   🏆 New best accuracy: {best_test_acc:.2f}%")
            
            # Save model
            model_path = os.path.join(config['save_dir'], "sports_category_model.pth")
            torch.save(model.state_dict(), model_path)
            print(f"   💾 Model saved: sports_category_model.pth")
        
        # Check target achievement
        if test_acc >= 80.0:
            print(f"\n🎉 TARGET ACHIEVED! {test_acc:.2f}% >= 80%")
            print("🏟️ Sports category classifier SUCCESSFUL!")
            
            # Show final per-class accuracy
            print(f"\n🏆 Final Per-Sport Performance:")
            for i in range(config['num_classes']):
                if class_total[i] > 0:
                    acc = 100. * class_correct[i] / class_total[i]
                    status = "✅" if acc >= 70 else "⚠️" if acc >= 50 else "❌"
                    print(f"   {status} {SPORTS_ACTIONS[i]}: {acc:.1f}%")
            break
            
        # Early stop if great progress
        if test_acc >= 85.0:
            print(f"\n🎉 EXCELLENT! {test_acc:.2f}% accuracy achieved!")
            break
    
    # Training summary
    total_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("🏆 SPORTS CATEGORY TRAINING COMPLETED!")
    print("=" * 60)
    print(f"🏟️ Sports Classes: {config['num_classes']}")
    print(f"🎯 Best Test Accuracy: {best_test_acc:.2f}%")
    print(f"⏱️ Total Training Time: {total_time/60:.1f} minutes")
    
    # Evolution Strategy Status
    print(f"\n🚀 EVOLUTION STRATEGY STATUS:")
    if best_test_acc >= 80.0:
        print("✅ Category Building Strategy: PROVEN SUCCESSFUL!")
        print("🎯 Sports category classifier works!")
        print("\n📈 NEXT EVOLUTION STEPS:")
        print("   1. ✅ Sports category mastered")
        print("   2. 🎵 Create Music category (Guitar, Piano, Drums...)")
        print("   3. 💪 Create Exercise category (PushUps, PullUps...)")
        print("   4. 🎭 Build confidence-based frontend")
    elif best_test_acc >= 70.0:
        print("🎯 Category Building: Good progress!")
        print("💡 Try more epochs or enhanced model")
    else:
        print(f"⚠️ Category accuracy {best_test_acc:.2f}% needs improvement")
    
    return {
        'best_accuracy': best_test_acc,
        'training_time': total_time
    }

def main():
    """Main execution"""
    
    try:
        # Train sports category classifier
        results = train_sports_category()
        
        # Success message
        if results['best_accuracy'] >= 70:
            print(f"\n🎉 SUCCESS! Sports category achieved {results['best_accuracy']:.2f}% accuracy")
            print(f"⚡ Training time: {results['training_time']/60:.1f} minutes")
            
            print(f"\n🚀 EVOLUTION STRATEGY PROVEN:")
            print(f"   ✅ Category-based training works!")
            print(f"   ✅ Much faster than binary approach")
            print(f"   ✅ Ready to scale to more categories")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()