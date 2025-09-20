"""
Phase 3: Unified Daily Activities Training (COMPLETE)
====================================================

Complete training system for 19 daily activities using R(2+1)D model.
Optimized for 75-85% accuracy in 4-6 hours on RTX 3050.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import time
from pathlib import Path
import json
from collections import Counter

from phase3_unified_dataset import get_phase3_dataloaders
from phase3_r2plus1d_model import create_phase3_model
from phase3_class_mapping import get_phase3_activities

class Phase3Trainer:
    """Advanced trainer for Phase 3 unified daily activities"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize mixed precision training
        self.scaler = GradScaler('cuda') if config['use_mixed_precision'] else None
        
        # Setup directories
        self.setup_directories()
        
        print(f"🚀 Phase 3 Unified Trainer Initialized")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {config['use_mixed_precision']}")
        
    def setup_directories(self):
        """Create output directories"""
        self.checkpoint_dir = Path(self.config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.logs_dir = Path(self.config['logs_dir'])
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Outputs: {self.checkpoint_dir}")
        
    def load_data(self):
        """Load Phase 3 unified dataset"""
        print(f"\n📂 Loading Phase 3 unified dataset...")
        
        self.train_loader, self.val_loader, self.class_info = get_phase3_dataloaders(
            video_root=self.config['video_root'],
            video_list_file=self.config['video_list_file'],
            batch_size=self.config['batch_size'],
            frames_per_clip=self.config['frames_per_clip'],
            num_workers=self.config['num_workers'],
            val_split=self.config['val_split']
        )
        
        # Compute class weights
        self._compute_class_weights()
        
        print(f"\n✅ Phase 3 data loaded successfully:")
        print(f"  • Train batches: {len(self.train_loader)}")
        print(f"  • Val batches: {len(self.val_loader)}")
        print(f"  • Total classes: {self.class_info['num_classes']}")
        
        return self.class_info['num_classes']
    
    def _compute_class_weights(self):
        """Compute class weights for balanced training"""
        print(f"\n📊 Computing class weights...")
        
        # Count classes by sampling training data
        class_counts = Counter()
        sample_size = min(50, len(self.train_loader))
        
        for i, (_, labels) in enumerate(self.train_loader):
            if i >= sample_size:
                break
            for label in labels:
                class_counts[label.item()] += 1
        
        # Compute inverse frequency weights
        total_samples = sum(class_counts.values())
        num_classes = self.class_info['num_classes']
        
        class_weights = []
        for class_idx in range(num_classes):
            count = class_counts.get(class_idx, 1)
            weight = total_samples / (num_classes * count)
            class_weights.append(weight)
        
        self.class_weights = torch.FloatTensor(class_weights).to(self.device)
        
        print(f"  • Weight range: [{min(class_weights):.2f}, {max(class_weights):.2f}]")
    
    def setup_model(self, num_classes):
        """Setup R(2+1)D model"""
        print(f"\n🏗️ Setting up Phase 3 R(2+1)D model...")
        
        self.model = create_phase3_model(
            num_classes=num_classes,
            pretrained=True,
            freeze_layers=self.config['freeze_layers']
        ).to(self.device)
        
        stats = self.model.get_parameter_stats()
        return stats['total_params'], stats['trainable_params']
    
    def setup_optimizer_and_scheduler(self):
        """Setup optimization"""
        print(f"\n⚙️ Setting up optimization...")
        
        # Group parameters for differential learning rates
        backbone_params = []
        classifier_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'fc' in name:
                    classifier_params.append(param)
                else:
                    backbone_params.append(param)
        
        # Create parameter groups
        param_groups = []
        if backbone_params:
            param_groups.append({'params': backbone_params, 'lr': self.config['backbone_lr']})
        if classifier_params:
            param_groups.append({'params': classifier_params, 'lr': self.config['learning_rate']})
        
        self.optimizer = optim.AdamW(param_groups, weight_decay=self.config['weight_decay'])
        
        # Learning rate scheduler
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=[self.config['backbone_lr'], self.config['learning_rate']],
            epochs=self.config['num_epochs'],
            steps_per_epoch=len(self.train_loader),
            pct_start=0.3
        )
        
        # Loss function with class weighting
        self.criterion = nn.CrossEntropyLoss(
            weight=self.class_weights,
            label_smoothing=0.1
        ).to(self.device)
        
        print(f"  • Differential LRs: {self.config['backbone_lr']} | {self.config['learning_rate']}")
    
    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Phase 3 Epoch {epoch}/{self.config['num_epochs']}")
        
        for batch_idx, (videos, labels) in enumerate(pbar):
            videos = videos.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            if self.config['use_mixed_precision']:
                with autocast('cuda'):
                    outputs = self.model(videos)
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(videos)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            self.scheduler.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            if batch_idx % 20 == 0:
                pbar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Acc': f"{100.*correct/total:.2f}%"
                })
        
        return total_loss / len(self.train_loader), 100. * correct / total
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for videos, labels in tqdm(self.val_loader, desc="Validation"):
                videos = videos.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                if self.config['use_mixed_precision']:
                    with autocast('cuda'):
                        outputs = self.model(videos)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(videos)
                    loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return total_loss / len(self.val_loader), 100. * correct / total
    
    def save_checkpoint(self, epoch, train_acc, val_acc, is_best=False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'class_info': self.class_info,
            'config': self.config
        }
        
        checkpoint_path = self.checkpoint_dir / f"phase3_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.checkpoint_dir / "phase3_best.pth"
            torch.save(checkpoint, best_path)
            print(f"💾 New best model: {val_acc:.2f}%")
    
    def train(self):
        """Main training loop"""
        print(f"\n🚀 Starting Phase 3 Training...")
        print(f"🎯 Target: 75%+ accuracy in {self.config['target_hours']} hours")
        
        # Setup
        num_classes = self.load_data()
        total_params, trainable_params = self.setup_model(num_classes)
        self.setup_optimizer_and_scheduler()
        
        print(f"\n📋 Training Summary:")
        print(f"  • Model: R(2+1)D for {num_classes} activities")
        print(f"  • Trainable params: {trainable_params:,}")
        print(f"  • Batch size: {self.config['batch_size']}")
        print(f"  • Epochs: {self.config['num_epochs']}")
        
        # Training loop
        best_val_acc = 0
        training_history = []
        start_time = time.time()
        
        for epoch in range(1, self.config['num_epochs'] + 1):
            epoch_start = time.time()
            
            # Train and validate
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()
            
            epoch_time = time.time() - epoch_start
            
            # Log results
            print(f"\nEpoch {epoch}/{self.config['num_epochs']} ({epoch_time:.1f}s)")
            print(f"Train: Loss {train_loss:.4f}, Acc {train_acc:.2f}%")
            print(f"Val:   Loss {val_loss:.4f}, Acc {val_acc:.2f}%")
            
            # Save checkpoint
            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
            
            self.save_checkpoint(epoch, train_acc, val_acc, is_best)
            
            # Store history
            training_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'epoch_time': epoch_time
            })
            
            # Check stopping conditions
            current_time = time.time() - start_time
            if val_acc >= 75.0:
                print(f"🎉 Target accuracy achieved!")
                break
            elif current_time > self.config['target_hours'] * 3600:
                print(f"⏰ Target time reached")
                break
        
        total_time = time.time() - start_time
        
        # Save history
        history_path = self.logs_dir / "phase3_history.json"
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        
        # Final results
        print(f"\n✅ Phase 3 Training Complete!")
        print(f"⏱️ Total time: {total_time/3600:.2f} hours")
        print(f"🏆 Best accuracy: {best_val_acc:.2f}%")
        
        if best_val_acc >= 75:
            print(f"🎊 SUCCESS: Target achieved!")
        else:
            print(f"🎯 Progress made, consider more training")
        
        return training_history, best_val_acc

def get_phase3_config():
    """Get Phase 3 training configuration"""
    has_cuda = torch.cuda.is_available()
    
    return {
        # Data
        'video_root': r"C:\ASH_PROJECT\data\kinetics400\videos_val",
        'video_list_file': r"C:\ASH_PROJECT\data\kinetics400\kinetics400_val_list_videos.txt",
        'checkpoint_dir': r"C:\ASH_PROJECT\outputs\phase3_checkpoints",
        'logs_dir': r"C:\ASH_PROJECT\outputs\phase3_logs",
        
        # Model
        'freeze_layers': 2,
        
        # Training - optimized for RTX 3050
        'batch_size': 4 if has_cuda else 2,
        'frames_per_clip': 16,
        'num_epochs': 25 if has_cuda else 5,
        'learning_rate': 3e-4,
        'backbone_lr': 1e-5,
        'weight_decay': 1e-4,
        'val_split': 0.2,
        'target_hours': 6.0,
        
        # Optimization
        'use_mixed_precision': has_cuda,
        'num_workers': 0,
    }

def main():
    """Main function"""
    print("🎯 Phase 3: Unified Daily Activities Training")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available - training will be slower")
    else:
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✅ Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    config = get_phase3_config()
    
    print(f"\n🎯 Phase 3 Features:")
    print(f"  • R(2+1)D model (best video architecture)")
    print(f"  • 19 daily activities")
    print(f"  • Class-weighted training")
    print(f"  • Mixed precision (GPU)")
    print(f"  • Target: 75%+ accuracy in 6 hours")
    
    # Start training
    trainer = Phase3Trainer(config)
    history, best_acc = trainer.train()
    
    # Show results
    activities = get_phase3_activities()
    print(f"\n🎊 Phase 3 Complete!")
    print(f"🏆 Best accuracy: {best_acc:.2f}%")
    print(f"🎯 Activities: {len(activities)} daily activities")
    print(f"📁 Model: {config['checkpoint_dir']}")

if __name__ == "__main__":
    main()