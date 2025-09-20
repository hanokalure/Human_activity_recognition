import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from tqdm import tqdm
import time
import os
from pathlib import Path

from dataset_optimized import get_optimized_dataloaders
from models_optimized import get_optimized_model, count_parameters, get_model_comparison

class OptimizedTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize mixed precision training
        self.scaler = GradScaler() if config['use_mixed_precision'] else None
        
        # Create output directories
        self.setup_directories()
        
        print(f"🚀 Initialized Optimized Trainer")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {config['use_mixed_precision']}")
        
    def setup_directories(self):
        """Create necessary output directories"""
        self.checkpoint_dir = Path(self.config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self):
        """Load optimized datasets and dataloaders"""
        print("\n📂 Loading optimized datasets...")
        
        self.train_loader, self.test_loader, self.class_names = get_optimized_dataloaders(
            data_root=self.config['data_root'],
            annotation_path=self.config['annotation_path'],
            batch_size=self.config['batch_size'],
            frames_per_clip=self.config['frames_per_clip'],
            num_workers=self.config['num_workers']
        )
        
        print(f"✅ Train batches: {len(self.train_loader)}")
        print(f"✅ Test batches: {len(self.test_loader)}")
        print(f"✅ Classes: {len(self.class_names)}")
        
        return len(self.class_names)
        
    def setup_model(self, num_classes):
        """Setup optimized model"""
        print("\n🏗️ Setting up optimized model...")
        
        self.model = get_optimized_model(
            model_type=self.config['model_type'],
            num_classes=num_classes,
            pretrained=self.config['pretrained'],
            freeze_backbone=self.config.get('freeze_backbone', False)
        ).to(self.device)
        
        # Count parameters
        total_params, trainable_params = count_parameters(self.model)
        
        return total_params
        
    def setup_optimizer_and_scheduler(self):
        """Setup optimizer and learning rate scheduler"""
        # Optimizer with weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        if self.config['scheduler'] == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config['num_epochs'] * len(self.train_loader)
            )
        elif self.config['scheduler'] == 'onecycle':
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config['learning_rate'],
                epochs=self.config['num_epochs'],
                steps_per_epoch=len(self.train_loader)
            )
        else:
            self.scheduler = None
            
        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=self.config.get('label_smoothing', 0.1)
        ).to(self.device)
        
    def train_epoch(self, epoch):
        """Train for one epoch with mixed precision"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config['num_epochs']}")
        
        for batch_idx, (clips, labels) in enumerate(pbar):
            clips, labels = clips.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            if self.config['use_mixed_precision']:
                # Mixed precision forward pass
                with autocast():
                    outputs = self.model(clips)
                    loss = self.criterion(outputs, labels)
                
                # Mixed precision backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping (optional)
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
        
        with torch.no_grad():
            for clips, labels in tqdm(self.test_loader, desc="Validation"):
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
        
        avg_loss = total_loss / len(self.test_loader)
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
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"💾 New best model saved with validation accuracy: {val_acc:.2f}%")
    
    def train(self):
        """Main training loop"""
        print("\n🚀 Starting optimized training...")
        
        # Load data
        num_classes = self.load_data()
        
        # Setup model
        total_params = self.setup_model(num_classes)
        
        # Setup optimizer and scheduler
        self.setup_optimizer_and_scheduler()
        
        # Training statistics
        best_val_acc = 0
        training_history = []
        
        print(f"\n📊 Training Configuration:")
        print(f"Model: {self.config['model_type']}")
        print(f"Parameters: {total_params:,}")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Frames per clip: {self.config['frames_per_clip']}")
        print(f"Learning rate: {self.config['learning_rate']}")
        print(f"Epochs: {self.config['num_epochs']}")
        print(f"Mixed precision: {self.config['use_mixed_precision']}")
        
        start_time = time.time()
        
        for epoch in range(1, self.config['num_epochs'] + 1):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate()
            
            epoch_time = time.time() - epoch_start
            
            # Log results
            print(f"\nEpoch {epoch}/{self.config['num_epochs']} completed in {epoch_time:.2f}s")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
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
        
        total_time = time.time() - start_time
        
        print(f"\n✅ Training completed!")
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        
        return training_history

def get_optimized_config():
    """Transfer learning configuration with frozen backbone"""
    return {
        # Data paths
        'data_root': r"C:\ASH_PROJECT\data\UCF101",
        'annotation_path': r"C:\ASH_PROJECT\data\ucfTrainTestlist",
        'checkpoint_dir': r"C:\ASH_PROJECT\outputs\checkpoints",
        
        # Transfer Learning Model Settings
        'model_type': 'r3d18_transfer',  # NEW: Transfer learning model
        'pretrained': True,              # Load Kinetics-400 weights
        'freeze_backbone': True,         # NEW: Freeze backbone layers
        
        # Transfer Learning Hyperparameters
        'batch_size': 32,                # Larger batches (can use more since less params to train)
        'frames_per_clip': 16,           # More frames for better temporal info
        'num_epochs': 50,                # More epochs since training is faster
        'learning_rate': 1e-3,           # Higher LR for classifier only
        'weight_decay': 1e-4,            # Moderate regularization
        'label_smoothing': 0.1,          # Keep label smoothing
        'gradient_clip_val': 1.0,        # Standard gradient clipping
        
        # Optimization settings
        'use_mixed_precision': True,
        'scheduler': 'onecycle',         # OneCycle for transfer learning
        
        # Data loading - Windows safe settings
        'num_workers': 0,  # Set to 0 for Windows stability
    }

def main():
    """Main training function"""
    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. GPU is required for optimized training.")
    
    print(f"🎯 GPU: {torch.cuda.get_device_name(0)}")
    print(f"🎯 CUDA Version: {torch.version.cuda}")
    
    # Show model comparison
    get_model_comparison()
    
    # Get configuration
    config = get_optimized_config()
    
    # Create trainer and start training
    trainer = OptimizedTrainer(config)
    history = trainer.train()
    
    print("\n🎉 Training completed successfully!")
    return history

if __name__ == "__main__":
    main()