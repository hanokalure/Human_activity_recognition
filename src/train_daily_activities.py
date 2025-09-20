"""
Daily Activities Transfer Learning Training
==========================================

Optimized training script for daily activities recognition using transfer learning.
Uses pretrained models on UCF-101 data with frozen backbone for fast, high-accuracy training.

Features:
- Transfer learning with frozen backbone
- Mixed precision training
- 14 daily activity classes from UCF-101
- Target: 85-90% accuracy in 2-4 hours
- RTX 3050 optimized
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

from daily_activities_dataset import get_daily_activities_dataloaders, print_dataset_statistics
from daily_activities_config import ACTIVITY_DESCRIPTIONS

class DailyActivitiesModel(nn.Module):
    """Transfer learning model for daily activities"""
    
    def __init__(self, num_classes=14, model_name='r3d_18', freeze_backbone=True):
        super(DailyActivitiesModel, self).__init__()
        
        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        
        # Load pretrained model
        if model_name == 'r3d_18':
            self.backbone = video_models.r3d_18(pretrained=True)
            in_features = self.backbone.fc.in_features
            # Replace the final layer
            self.backbone.fc = nn.Linear(in_features, num_classes)
            
        elif model_name == 'r2plus1d_18':
            self.backbone = video_models.r2plus1d_18(pretrained=True)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
            
        elif model_name == 'mc3_18':
            self.backbone = video_models.mc3_18(pretrained=True)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
            
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Freeze backbone if specified
        if freeze_backbone:
            self._freeze_backbone()
            
        print(f"🏗️ Model: {model_name}")
        print(f"🧊 Backbone frozen: {freeze_backbone}")
        print(f"📊 Output classes: {num_classes}")
    
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

class DailyActivitiesTrainer:
    """Optimized trainer for daily activities"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize mixed precision training
        self.scaler = GradScaler() if config['use_mixed_precision'] else None
        
        # Create output directories
        self.setup_directories()
        
        print(f"🚀 Daily Activities Trainer")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {config['use_mixed_precision']}")
        
    def setup_directories(self):
        """Create necessary output directories"""
        self.checkpoint_dir = Path(self.config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.logs_dir = Path(self.config['logs_dir'])
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self):
        """Load daily activities datasets"""
        print(f"\n📂 Loading daily activities data...")
        
        # Print dataset statistics
        print_dataset_statistics(self.config['data_root'])
        
        # Load dataloaders
        self.train_loader, self.val_loader, self.class_info = get_daily_activities_dataloaders(
            data_root=self.config['data_root'],
            batch_size=self.config['batch_size'],
            frames_per_clip=self.config['frames_per_clip'],
            num_workers=self.config['num_workers'],
            use_ucf101=True,
            use_kinetics=False  # UCF-101 only for now
        )
        
        print(f"\n✅ Data loaded successfully:")
        print(f"  • Train batches: {len(self.train_loader)}")
        print(f"  • Val batches: {len(self.val_loader)}")
        print(f"  • Classes: {self.class_info['num_classes']}")
        
        return self.class_info['num_classes']
    
    def setup_model(self, num_classes):
        """Setup transfer learning model"""
        print(f"\n🏗️ Setting up transfer learning model...")
        
        self.model = DailyActivitiesModel(
            num_classes=num_classes,
            model_name=self.config['model_name'],
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
        """Setup optimizer and learning rate scheduler"""
        # Only optimize trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        if self.config['scheduler'] == 'onecycle':
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config['learning_rate'],
                epochs=self.config['num_epochs'],
                steps_per_epoch=len(self.train_loader),
                pct_start=0.3,  # Warmup for 30% of training
                div_factor=25,  # Start with lr/25
                final_div_factor=10000  # End with lr/10000
            )
        elif self.config['scheduler'] == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['num_epochs'] * len(self.train_loader),
                eta_min=self.config['learning_rate'] * 0.001
            )
        else:
            self.scheduler = None
            
        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=self.config.get('label_smoothing', 0.1)
        ).to(self.device)
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config['num_epochs']}")
        
        for batch_idx, (videos, labels) in enumerate(pbar):
            videos, labels = videos.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            if self.config['use_mixed_precision']:
                with autocast():
                    outputs = self.model(videos)
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.get('gradient_clip_val', 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clip_val'])
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(videos)
                loss = self.criterion(outputs, labels)
                loss.backward()
                
                if self.config.get('gradient_clip_val', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clip_val'])
                
                self.optimizer.step()
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            if batch_idx % 10 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Acc': f"{100.*correct/total:.2f}%",
                    'LR': f"{current_lr:.2e}"
                })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        class_correct = torch.zeros(self.class_info['num_classes'])
        class_total = torch.zeros(self.class_info['num_classes'])
        
        with torch.no_grad():
            for videos, labels in tqdm(self.val_loader, desc="Validation"):
                videos, labels = videos.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                
                if self.config['use_mixed_precision']:
                    with autocast():
                        outputs = self.model(videos)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(videos)
                    loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Per-class accuracy
                c = (predicted == labels)
                for i in range(labels.size(0)):
                    label = labels[i]
                    if labels.size(0) == 1:
                        class_correct[label] += c.item()
                    else:
                        class_correct[label] += c[i].item()
                    class_total[label] += 1
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        # Print per-class accuracy
        print(f"\n📊 Per-class validation accuracy:")
        for i, class_name in enumerate(self.class_info['class_names']):
            if i < len(class_correct) and class_total[i] > 0:
                class_acc = 100. * class_correct[i] / class_total[i]
                print(f"  • {class_name}: {class_acc:.1f}%")
        
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
        checkpoint_path = self.checkpoint_dir / f"daily_activities_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "daily_activities_best.pth"
            torch.save(checkpoint, best_path)
            print(f"💾 New best model saved with validation accuracy: {val_acc:.2f}%")
    
    def plot_training_history(self, history):
        """Plot training curves"""
        epochs = [h['epoch'] for h in history]
        train_loss = [h['train_loss'] for h in history]
        val_loss = [h['val_loss'] for h in history]
        train_acc = [h['train_acc'] for h in history]
        val_acc = [h['val_acc'] for h in history]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss plot
        ax1.plot(epochs, train_loss, label='Train Loss')
        ax1.plot(epochs, val_loss, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(epochs, train_acc, label='Train Acc')
        ax2.plot(epochs, val_acc, label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plot_path = self.logs_dir / "training_curves.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Training curves saved to: {plot_path}")
    
    def train(self):
        """Main training loop"""
        print(f"\n🚀 Starting Daily Activities Training...")
        print(f"Target: 85-90% accuracy on {self.config['num_epochs']} epochs")
        
        # Load data
        num_classes = self.load_data()
        
        # Setup model
        total_params, trainable_params = self.setup_model(num_classes)
        
        # Setup optimizer and scheduler
        self.setup_optimizer_and_scheduler()
        
        # Training loop
        best_val_acc = 0
        training_history = []
        
        print(f"\n🎯 Training Configuration:")
        print(f"  • Model: {self.config['model_name']} (transfer learning)")
        print(f"  • Classes: {num_classes}")
        print(f"  • Trainable params: {trainable_params:,}")
        print(f"  • Batch size: {self.config['batch_size']}")
        print(f"  • Learning rate: {self.config['learning_rate']}")
        print(f"  • Epochs: {self.config['num_epochs']}")
        
        start_time = time.time()
        
        for epoch in range(1, self.config['num_epochs'] + 1):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
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
            
            # Early stopping check
            if val_acc >= 90.0:
                print(f"🎉 Excellent accuracy achieved! Stopping early.")
                break
        
        total_time = time.time() - start_time
        
        # Save training history
        history_path = self.logs_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        
        # Plot training curves
        self.plot_training_history(training_history)
        
        print(f"\n✅ Training completed!")
        print(f"⏱️ Total time: {total_time/3600:.2f} hours")
        print(f"🏆 Best validation accuracy: {best_val_acc:.2f}%")
        print(f"📁 Model saved to: {self.checkpoint_dir}")
        
        return training_history, best_val_acc

def get_daily_activities_config():
    """Get optimized configuration for daily activities training"""
    # Detect if CUDA is available and adjust settings
    has_cuda = torch.cuda.is_available()
    
    return {
        # Data configuration
        'data_root': r"C:\ASH_PROJECT\data",
        'checkpoint_dir': r"C:\ASH_PROJECT\outputs\daily_activities_checkpoints",
        'logs_dir': r"C:\ASH_PROJECT\outputs\daily_activities_logs",
        
        # Model configuration - Transfer Learning
        'model_name': 'r3d_18',        # Options: 'r3d_18', 'r2plus1d_18', 'mc3_18'
        'freeze_backbone': True,       # Freeze backbone for transfer learning
        
        # Training hyperparameters - Auto-adjust for CPU/GPU
        'batch_size': 8 if has_cuda else 2,              # Smaller batch for CPU
        'frames_per_clip': 16 if has_cuda else 8,        # Fewer frames for CPU
        'num_epochs': 25 if has_cuda else 3,             # Fewer epochs for testing on CPU
        'learning_rate': 1e-3,        # Higher LR since only training classifier
        'weight_decay': 1e-4,         # Moderate regularization
        'label_smoothing': 0.1,       # Helps with generalization
        'gradient_clip_val': 1.0,     # Prevent gradient explosion
        
        # Optimization settings
        'use_mixed_precision': has_cuda,  # Only use mixed precision with CUDA
        'scheduler': 'onecycle',          # Great for transfer learning
        
        # Data loading - Windows optimized
        'num_workers': 0,             # Single threaded for Windows stability
    }

def main():
    """Main training function"""
    print("🎯 Daily Activities Recognition Training")
    print("=" * 50)
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available. Training will be slow on CPU.")
        print("💡 Continuing with CPU-optimized settings...")
    else:
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✅ CUDA: {torch.version.cuda}")
    
    # Get configuration
    config = get_daily_activities_config()
    
    # Create trainer and start training
    trainer = DailyActivitiesTrainer(config)
    history, best_acc = trainer.train()
    
    # Final summary
    print("\n" + "=" * 50)
    print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print(f"🏆 Best accuracy: {best_acc:.2f}%")
    
    if best_acc >= 85:
        print("✅ Target accuracy (85%+) achieved!")
    elif best_acc >= 80:
        print("⚡ Good accuracy achieved - close to target!")
    else:
        print("📈 Consider training longer or adjusting hyperparameters")
    
    print(f"📁 Model saved to: {config['checkpoint_dir']}")
    print(f"📊 Logs saved to: {config['logs_dir']}")
    print("\n🎯 Next step: Use the trained model for video-to-text inference!")

if __name__ == "__main__":
    main()