"""
Phase 2 GPU Training: High Accuracy Training for 80%+ Accuracy
==============================================================

Optimized GPU training configuration for RTX 3050 (4GB VRAM).
Target: 80%+ accuracy in 3-5 hours with more epochs and better hyperparameters.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
import torchvision.models.video as video_models
from tqdm import tqdm
import time
import os
from pathlib import Path
import json
import matplotlib.pyplot as plt
import shutil

from kinetics_dataset_fixed import get_kinetics_dataloaders
from kinetics_class_mapping import get_all_activities, PHASE_1_ACTIVITIES, PHASE_2_ACTIVITIES

class Phase2Model(nn.Module):
    """Extended model for 19 classes (Phase 1 + Phase 2)"""
    
    def __init__(self, phase1_model_path, num_classes=19, freeze_backbone=True):
        super(Phase2Model, self).__init__()
        
        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone
        
        # Load Phase 1 model structure
        self.backbone = video_models.r3d_18(weights=None)
        
        # Load Phase 1 trained weights
        if os.path.exists(phase1_model_path):
            print(f"🔄 Loading Phase 1 model from: {phase1_model_path}")
            checkpoint = torch.load(phase1_model_path, map_location='cpu')
            
            # Load backbone weights (excluding final layer)
            model_state = checkpoint.get('model_state_dict', checkpoint)
            backbone_state = {k: v for k, v in model_state.items() 
                            if not k.startswith('fc') and not k.startswith('backbone.fc')}
            
            # Load backbone
            self.backbone.load_state_dict(backbone_state, strict=False)
            print(f"✅ Loaded Phase 1 backbone weights")
        else:
            print(f"⚠️ Phase 1 model not found at {phase1_model_path}")
            print(f"🔄 Using ImageNet pretrained weights instead")
            self.backbone = video_models.r3d_18(weights='DEFAULT')
        
        # Replace final layer for 19 classes
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.2),  # Add dropout for better generalization
            nn.Linear(in_features, num_classes)
        )
        
        # Freeze backbone if specified
        if freeze_backbone:
            self._freeze_backbone()
            
        print(f"🏗️ Model: R3D-18 extended to {num_classes} classes")
        print(f"🧊 Backbone frozen: {freeze_backbone}")
    
    def _freeze_backbone(self):
        """Freeze backbone layers for transfer learning"""
        # Freeze all parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Unfreeze only the final classification layer
        for param in self.backbone.fc.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        return self.backbone(x)
    
    def get_trainable_parameters(self):
        """Get count of trainable parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params

class Phase2Trainer:
    """High-accuracy GPU trainer for Phase 2 activities"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize mixed precision training
        self.scaler = GradScaler() if config['use_mixed_precision'] else None
        
        # Create output directories
        self.setup_directories()
        
        print(f"🚀 High-Accuracy Phase 2 Trainer Initialized")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {config['use_mixed_precision']}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory // (1024**3)}GB")
        
    def setup_directories(self):
        """Create necessary output directories"""
        self.checkpoint_dir = Path(self.config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.logs_dir = Path(self.config['logs_dir'])
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Backup Phase 1 model
        phase1_model = self.config['phase1_model_path']
        if os.path.exists(phase1_model):
            backup_path = self.checkpoint_dir / "phase1_model_backup.pth"
            if not backup_path.exists():  # Don't overwrite existing backup
                shutil.copy2(phase1_model, backup_path)
                print(f"✅ Phase 1 model backed up to: {backup_path}")
        
    def load_data(self):
        """Load Kinetics-400 datasets with GPU-optimized settings"""
        print(f"\n📂 Loading Kinetics-400 data for GPU training...")
        
        # Load dataloaders
        self.train_loader, self.val_loader, self.class_info = get_kinetics_dataloaders(
            video_root=self.config['kinetics_video_root'],
            video_list_file=self.config['kinetics_video_list'],
            batch_size=self.config['batch_size'],
            frames_per_clip=self.config['frames_per_clip'],
            num_workers=self.config['num_workers'],
            val_split=self.config['val_split']
        )
        
        print(f"\n✅ GPU-optimized data loaded successfully:")
        print(f"  • Train batches: {len(self.train_loader)}")
        print(f"  • Val batches: {len(self.val_loader)}")
        print(f"  • Total classes: {self.class_info['num_classes']}")
        print(f"  • Phase 2 activities: {len(self.class_info['phase_2_activities'])}")
        print(f"  • Batch size: {self.config['batch_size']}")
        print(f"  • Frames per clip: {self.config['frames_per_clip']}")
        
        return self.class_info['num_classes']
    
    def setup_model(self, num_classes):
        """Setup Phase 2 model for GPU training"""
        print(f"\n🏗️ Setting up high-accuracy Phase 2 model...")
        
        self.model = Phase2Model(
            phase1_model_path=self.config['phase1_model_path'],
            num_classes=num_classes,
            freeze_backbone=self.config['freeze_backbone']
        ).to(self.device)
        
        # Count parameters
        total_params, trainable_params = self.model.get_trainable_parameters()
        
        print(f"📊 Model Statistics:")
        print(f"  • Total parameters: {total_params:,}")
        print(f"  • Trainable parameters: {trainable_params:,}")
        print(f"  • Frozen parameters: {total_params - trainable_params:,}")
        print(f"  • Training efficiency: {trainable_params/total_params*100:.1f}% of params")
        
        return total_params, trainable_params
    
    def setup_optimizer_and_scheduler(self):
        """Setup optimizer and learning rate scheduler for high accuracy"""
        # Only optimize trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Use AdamW with optimized settings for high accuracy
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler - Cosine annealing for better convergence
        if self.config['scheduler'] == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['num_epochs'],
                eta_min=self.config['learning_rate'] * 0.01
            )
        elif self.config['scheduler'] == 'onecycle':
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config['learning_rate'],
                epochs=self.config['num_epochs'],
                steps_per_epoch=len(self.train_loader),
                pct_start=0.3,
                div_factor=25,
                final_div_factor=10000
            )
        else:
            self.scheduler = None
            
        # Loss function with label smoothing for better generalization
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=self.config.get('label_smoothing', 0.1)
        ).to(self.device)
        
        print(f"🎯 Optimizer: AdamW with LR={self.config['learning_rate']}")
        print(f"📈 Scheduler: {self.config['scheduler']}")
        print(f"🎪 Label smoothing: {self.config.get('label_smoothing', 0.1)}")
    
    def train_epoch(self, epoch):
        """Train for one epoch with high accuracy optimizations"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config['num_epochs']}")
        
        for batch_idx, (clips, labels) in enumerate(pbar):
            clips, labels = clips.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            if self.config['use_mixed_precision']:
                # Mixed precision forward pass
                with autocast():
                    outputs = self.model(clips)
                    loss = self.criterion(outputs, labels)
                
                # Mixed precision backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping for stability
                if self.config.get('gradient_clip_val', 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clip_val'])
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                outputs = self.model(clips)
                loss = self.criterion(outputs, labels)
                loss.backward()
                
                # Gradient clipping
                if self.config.get('gradient_clip_val', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clip_val'])
                
                self.optimizer.step()
            
            # Update learning rate (for OneCycleLR)
            if self.config['scheduler'] == 'onecycle':
                self.scheduler.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            if batch_idx % 5 == 0:  # Update more frequently
                current_lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Acc': f"{100.*correct/total:.2f}%",
                    'LR': f"{current_lr:.2e}"
                })
        
        # Update learning rate (for CosineAnnealingLR)
        if self.config['scheduler'] == 'cosine':
            self.scheduler.step()
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for clips, labels in tqdm(self.val_loader, desc="Validation"):
                clips, labels = clips.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                
                if self.config['use_mixed_precision']:
                    with autocast():
                        outputs = self.model(clips)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(clips)
                    loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch, train_acc, val_acc, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'class_info': self.class_info,
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"phase2_gpu_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "phase2_gpu_best.pth"
            torch.save(checkpoint, best_path)
            print(f"💾 New best model saved with validation accuracy: {val_acc:.2f}%")
    
    def train(self):
        """Main high-accuracy training loop"""
        print(f"\n🚀 Starting High-Accuracy Phase 2 GPU Training...")
        print(f"🎯 Target: 80%+ accuracy on 19 classes")
        print(f"⏱️ Estimated time: 3-5 hours")
        
        # Load data
        num_classes = self.load_data()
        
        # Setup model
        total_params, trainable_params = self.setup_model(num_classes)
        
        # Setup optimizer and scheduler
        self.setup_optimizer_and_scheduler()
        
        # Training loop
        best_val_acc = 0
        training_history = []
        patience_counter = 0
        
        print(f"\n🎯 High-Accuracy Training Configuration:")
        print(f"  • Model: R3D-18 (Phase 1 + Phase 2)")
        print(f"  • Classes: {num_classes} total (9 Phase 1 + 10 Phase 2)")
        print(f"  • Trainable params: {trainable_params:,}")
        print(f"  • Batch size: {self.config['batch_size']}")
        print(f"  • Learning rate: {self.config['learning_rate']}")
        print(f"  • Epochs: {self.config['num_epochs']}")
        print(f"  • Early stopping patience: {self.config['early_stopping_patience']}")
        
        start_time = time.time()
        
        for epoch in range(1, self.config['num_epochs'] + 1):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate()
            
            epoch_time = time.time() - epoch_start
            elapsed_time = time.time() - start_time
            
            # Log results
            print(f"\nEpoch {epoch}/{self.config['num_epochs']} ({epoch_time:.1f}s | Total: {elapsed_time/3600:.2f}h)")
            print(f"Train: Loss {train_loss:.4f}, Acc {train_acc:.2f}%")
            print(f"Val:   Loss {val_loss:.4f}, Acc {val_acc:.2f}%")
            
            # Save checkpoint
            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
            
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
            
            # Early stopping checks
            if val_acc >= 80.0:
                print(f"🎉 TARGET ACHIEVED! 80%+ accuracy reached!")
                break
            elif val_acc >= 85.0:
                print(f"🎊 EXCELLENT! 85%+ accuracy achieved!")
                break
            elif patience_counter >= self.config['early_stopping_patience']:
                print(f"⏳ Early stopping triggered after {patience_counter} epochs without improvement")
                break
        
        total_time = time.time() - start_time
        
        # Save training history
        history_path = self.logs_dir / "phase2_gpu_training_history.json"
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        
        print(f"\n✅ High-Accuracy Phase 2 Training completed!")
        print(f"⏱️ Total time: {total_time/3600:.2f} hours")
        print(f"🏆 Best validation accuracy: {best_val_acc:.2f}%")
        print(f"📁 Model saved to: {self.checkpoint_dir}")
        
        return training_history, best_val_acc

def get_high_accuracy_config():
    """Get high-accuracy GPU training configuration for RTX 3050"""
    has_cuda = torch.cuda.is_available()
    
    if not has_cuda:
        raise RuntimeError("CUDA not available! GPU training requires CUDA.")
    
    return {
        # Data configuration
        'kinetics_video_root': r"C:\ASH_PROJECT\data\kinetics400\videos_val",
        'kinetics_video_list': r"C:\ASH_PROJECT\data\kinetics400\kinetics400_val_list_videos.txt",
        'phase1_model_path': r"C:\ASH_PROJECT\outputs\daily_activities_checkpoints\daily_activities_best.pth",
        'checkpoint_dir': r"C:\ASH_PROJECT\outputs\phase2_gpu_checkpoints",
        'logs_dir': r"C:\ASH_PROJECT\outputs\phase2_gpu_logs",
        
        # Model configuration
        'freeze_backbone': True,       # Freeze backbone for transfer learning
        
        # High-accuracy training hyperparameters
        'batch_size': 6,               # Optimized for RTX 3050 4GB VRAM
        'frames_per_clip': 16,         # Full temporal resolution
        'num_epochs': 50,              # More epochs for higher accuracy
        'learning_rate': 5e-4,         # Lower LR for better convergence
        'weight_decay': 1e-4,
        'label_smoothing': 0.1,
        'gradient_clip_val': 1.0,
        'val_split': 0.2,
        'early_stopping_patience': 15, # Stop if no improvement
        
        # Optimization settings
        'use_mixed_precision': True,   # Essential for RTX 3050
        'scheduler': 'cosine',         # Better than onecycle for long training
        
        # Data loading
        'num_workers': 2,              # Optimized for GPU training
    }

def estimate_training_time():
    """Estimate training time for RTX 3050"""
    print("⏱️ Training Time Estimation for RTX 3050:")
    print("=" * 50)
    print(f"• GPU: RTX 3050 (4GB VRAM)")
    print(f"• Batch size: 6")
    print(f"• Frames per clip: 16") 
    print(f"• Total batches per epoch: ~172")
    print(f"• Estimated time per epoch: 6-8 minutes")
    print(f"• Target epochs for 80% accuracy: 25-40")
    print(f"• Estimated total time: 2.5-5.3 hours")
    print(f"• Expected accuracy: 80-85%")
    print("=" * 50)

def main():
    """Main high-accuracy training function"""
    print("🎯 Phase 2 High-Accuracy GPU Training")
    print("=" * 60)
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA not available! GPU training requires CUDA.")
        print("💡 Please ensure your GPU drivers are installed correctly.")
        return
    
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ CUDA: {torch.version.cuda}")
    print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory // (1024**3)}GB")
    
    # Show time estimation
    estimate_training_time()
    
    # Get configuration
    config = get_high_accuracy_config()
    
    print(f"\n🚀 Starting high-accuracy training...")
    print(f"Press Ctrl+C to stop training early if needed.")
    
    try:
        # Create trainer and start training
        trainer = Phase2Trainer(config)
        history, best_acc = trainer.train()
        
        # Final summary
        print(f"\n🎊 High-Accuracy Phase 2 Training Complete!")
        print(f"🏆 Final accuracy: {best_acc:.2f}%")
        print(f"🎯 Model now supports all 19 daily activities!")
        
        print(f"\n📊 Activities supported:")
        print(f"Phase 1 (9): {', '.join(PHASE_1_ACTIVITIES)}")
        print(f"Phase 2 (10): {', '.join(PHASE_2_ACTIVITIES)}")
        
        if best_acc >= 80.0:
            print(f"\n✅ SUCCESS: Target 80%+ accuracy achieved! 🎉")
            print(f"🚀 Ready for production deployment!")
        elif best_acc >= 75.0:
            print(f"\n✅ GOOD: 75%+ accuracy achieved!")
            print(f"💡 Consider fine-tuning for even higher accuracy")
        else:
            print(f"\n⚠️ Consider extending training or adjusting hyperparameters")
            
    except KeyboardInterrupt:
        print(f"\n⏹️ Training stopped by user")
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()