"""
Phase 4 Training - Separate UCF-101 + HMDB-51 Activities
=======================================================

Trains R(2+1)D model on 17 separate daily activities for 80+ accuracy.
No mixing of UCF-101 and HMDB-51 classes for optimal performance.

Features:
- 17 separate high-quality daily activities 
- 10 UCF-101 + 7 HMDB-51 activities (no overlap)
- Advanced data augmentation
- Progressive learning rate scheduling
- Mixed precision training
- Target accuracy: 80-85%

Usage:
    python phase4_train.py --epochs 50 --batch_size 8
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms as transforms

import os
import cv2
import numpy as np
import json
import random
from pathlib import Path
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from collections import defaultdict
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UnifiedVideoDataset(Dataset):
    """Unified UCF-101 + HMDB-51 video dataset"""
    
    def __init__(self, root_dir, split_file, transform=None, frames_per_clip=16, 
                 frame_rate=1, augment=True):
        self.root_dir = Path(root_dir)
        self.frames_per_clip = frames_per_clip
        self.frame_rate = frame_rate
        self.transform = transform
        self.augment = augment
        
        # Load split file
        self.samples = []
        self.classes = set()
        
        with open(split_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    video_path = parts[0]
                    class_name = parts[1]
                    
                    full_path = self.root_dir / video_path
                    if full_path.exists():
                        self.samples.append((str(full_path), class_name))
                        self.classes.add(class_name)
        
        # Create class mapping
        self.classes = sorted(list(self.classes))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        logger.info(f"Dataset loaded: {len(self.samples)} samples, {len(self.classes)} classes")
        
        # Count samples per class
        class_counts = defaultdict(int)
        for _, class_name in self.samples:
            class_counts[class_name] += 1
        
        for class_name in sorted(class_counts.keys()):
            logger.info(f"  {class_name}: {class_counts[class_name]} videos")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        video_path, class_name = self.samples[idx]
        class_idx = self.class_to_idx[class_name]
        
        # Load video frames
        frames = self._load_video_frames(video_path)
        
        if frames is None or len(frames) == 0:
            # Return a zero tensor if video loading fails
            frames = torch.zeros(3, self.frames_per_clip, 112, 112)
            return frames, class_idx
        
        # Convert to tensor and apply transforms
        frames = torch.stack(frames)  # (T, H, W, C)
        frames = frames.permute(3, 0, 1, 2)  # (C, T, H, W)
        
        if self.transform:
            frames = self.transform(frames)
        
        return frames, class_idx
    
    def _load_video_frames(self, video_path):
        """Load frames from video file"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.warning(f"Failed to open video: {video_path}")
                return None
            
            frames = []
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate frame indices to sample
            if total_frames < self.frames_per_clip:
                # If video is too short, repeat frames
                frame_indices = np.linspace(0, total_frames - 1, self.frames_per_clip, dtype=int)
            else:
                # Sample frames uniformly
                frame_indices = np.linspace(0, total_frames - 1, self.frames_per_clip, dtype=int)
            
            # Read specific frames
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # Resize frame
                    frame = cv2.resize(frame, (112, 112))
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Normalize to [0, 1]
                    frame = frame.astype(np.float32) / 255.0
                    frames.append(torch.from_numpy(frame))
                else:
                    # If frame reading fails, use the last good frame or zeros
                    if frames:
                        frames.append(frames[-1].clone())
                    else:
                        frames.append(torch.zeros(112, 112, 3))
            
            cap.release()
            return frames
            
        except Exception as e:
            logger.error(f"Error loading video {video_path}: {e}")
            return None

class VideoTransforms:
    """Custom video transformations"""
    
    def __init__(self, is_training=True, augment_strength=0.3):
        self.is_training = is_training
        self.augment_strength = augment_strength
    
    def __call__(self, video):
        """Apply transforms to video tensor (C, T, H, W)"""
        
        if self.is_training:
            # Random horizontal flip
            if random.random() < 0.5:
                video = torch.flip(video, [3])  # Flip width dimension
            
            # Random brightness/contrast adjustment
            if random.random() < 0.3:
                brightness_factor = 1.0 + random.uniform(-0.2, 0.2)
                video = torch.clamp(video * brightness_factor, 0, 1)
            
            # Random crop and resize (spatial augmentation)
            if random.random() < 0.4:
                video = self._random_crop_resize(video)
        
        # Normalize (same as ImageNet)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
        video = (video - mean) / std
        
        return video
    
    def _random_crop_resize(self, video):
        """Random spatial crop and resize"""
        C, T, H, W = video.shape
        
        # Random crop size (80% to 100% of original)
        crop_size = random.randint(int(0.8 * H), H)
        
        # Random crop position
        crop_y = random.randint(0, H - crop_size)
        crop_x = random.randint(0, W - crop_size)
        
        # Crop all frames
        cropped = video[:, :, crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]
        
        # Resize back to original size using interpolation
        resized = torch.nn.functional.interpolate(
            cropped, size=(H, W), mode='bilinear', align_corners=False
        )
        
        return resized

class R2Plus1DModel(nn.Module):
    """R(2+1)D model for video classification"""
    
    def __init__(self, num_classes=17, dropout_rate=0.5):
        super(R2Plus1DModel, self).__init__()
        
        # Load pretrained R(2+1)D model
        from torchvision.models.video import r2plus1d_18
        self.backbone = r2plus1d_18(pretrained=True)
        
        # Replace the classifier
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, num_classes)
        )
        
        # Initialize new layers
        for m in self.backbone.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.backbone(x)

class Phase4Trainer:
    """Phase 4 trainer for unified dataset"""
    
    def __init__(self, model, train_loader, val_loader, device, num_classes):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_classes = num_classes
        
        # Training components
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = GradScaler()
        
        # Metrics tracking
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.learning_rates = []
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_model_state = None
    
    def setup_optimizer(self, base_lr=1e-4, weight_decay=1e-4):
        """Setup optimizer with different learning rates for different parts"""
        
        # Different learning rates for backbone and classifier
        backbone_params = []
        classifier_params = []
        
        for name, param in self.model.named_parameters():
            if 'fc' in name:  # Classifier layers
                classifier_params.append(param)
            else:  # Backbone layers
                backbone_params.append(param)
        
        self.optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': base_lr * 0.1, 'weight_decay': weight_decay},
            {'params': classifier_params, 'lr': base_lr, 'weight_decay': weight_decay * 0.5}
        ])
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer, 
            max_lr=[base_lr * 0.1, base_lr],
            epochs=50,
            steps_per_epoch=len(self.train_loader),
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        logger.info(f"Optimizer setup: backbone_lr={base_lr*0.1:.2e}, classifier_lr={base_lr:.2e}")
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        
        for batch_idx, (videos, labels) in enumerate(pbar):
            videos, labels = videos.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed precision forward pass
            with autocast():
                outputs = self.model(videos)
                loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct_predictions += predicted.eq(labels).sum().item()
            total_samples += labels.size(0)
            
            # Update progress bar
            current_lr = self.scheduler.get_last_lr()[1]  # Classifier LR
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100*correct_predictions/total_samples:.2f}%',
                'LR': f'{current_lr:.2e}'
            })
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100 * correct_predictions / total_samples
        
        self.train_losses.append(epoch_loss)
        self.train_accuracies.append(epoch_acc)
        self.learning_rates.append(self.scheduler.get_last_lr()[1])
        
        return epoch_loss, epoch_acc
    
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]")
            
            for videos, labels in pbar:
                videos, labels = videos.to(self.device), labels.to(self.device)
                
                with autocast():
                    outputs = self.model(videos)
                    loss = self.criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                correct_predictions += predicted.eq(labels).sum().item()
                total_samples += labels.size(0)
                
                # Store for detailed metrics
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100*correct_predictions/total_samples:.2f}%'
                })
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100 * correct_predictions / total_samples
        
        self.val_losses.append(epoch_loss)
        self.val_accuracies.append(epoch_acc)
        
        # Save best model
        if epoch_acc > self.best_val_acc:
            self.best_val_acc = epoch_acc
            self.best_model_state = self.model.state_dict().copy()
            logger.info(f"🎯 New best validation accuracy: {epoch_acc:.2f}%")
        
        return epoch_loss, epoch_acc, all_predictions, all_labels
    
    def train(self, num_epochs=50, save_dir="C:/ASH_PROJECT/models/phase4"):
        """Full training loop"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🚀 Starting Phase 4 training for {num_epochs} epochs")
        logger.info(f"📁 Model save directory: {save_dir}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Training
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validation
            val_loss, val_acc, val_preds, val_labels = self.validate(epoch)
            
            epoch_time = time.time() - epoch_start
            
            # Log epoch results
            logger.info(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)")
            logger.info(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
            logger.info(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
            logger.info(f"  Best Val Acc: {self.best_val_acc:.2f}%")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint_path = save_dir / f"checkpoint_epoch_{epoch+1}.pth"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_val_acc': self.best_val_acc,
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'train_accuracies': self.train_accuracies,
                    'val_accuracies': self.val_accuracies
                }, checkpoint_path)
                logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
        
        # Save final model
        total_time = time.time() - start_time
        
        # Load best model state
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        # Save best model
        final_model_path = save_dir / "phase4_best_model.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'class_names': self.train_loader.dataset.classes,
            'class_to_idx': self.train_loader.dataset.class_to_idx,
            'best_val_acc': self.best_val_acc,
            'num_classes': self.num_classes,
            'training_time': total_time
        }, final_model_path)
        
        # Save training history
        self.save_training_plots(save_dir)
        
        logger.info(f"🎉 Training completed in {total_time/3600:.2f} hours")
        logger.info(f"📊 Best validation accuracy: {self.best_val_acc:.2f}%")
        logger.info(f"💾 Final model saved: {final_model_path}")
        
        return final_model_path
    
    def save_training_plots(self, save_dir):
        """Save training history plots"""
        
        # Loss plot
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss', color='blue', alpha=0.7)
        plt.plot(self.val_losses, label='Val Loss', color='red', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label='Train Acc', color='blue', alpha=0.7)
        plt.plot(self.val_accuracies, label='Val Acc', color='red', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / "training_history.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Learning rate plot
        plt.figure(figsize=(8, 5))
        plt.plot(self.learning_rates, color='green', alpha=0.7)
        plt.xlabel('Step')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.savefig(save_dir / "learning_rate.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Training plots saved in {save_dir}")

def create_data_loaders(dataset_root, batch_size=8, num_workers=4):
    """Create train and validation data loaders"""
    
    dataset_root = Path(dataset_root)
    train_split = dataset_root / "train_split.txt"
    test_split = dataset_root / "test_split.txt"
    
    if not train_split.exists() or not test_split.exists():
        raise FileNotFoundError(f"Split files not found in {dataset_root}")
    
    # Create datasets
    train_dataset = UnifiedVideoDataset(
        dataset_root, train_split, 
        transform=VideoTransforms(is_training=True),
        augment=True
    )
    
    val_dataset = UnifiedVideoDataset(
        dataset_root, test_split,
        transform=VideoTransforms(is_training=False),
        augment=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, 
        shuffle=True, num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    logger.info(f"Data loaders created:")
    logger.info(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    logger.info(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    logger.info(f"  Classes: {len(train_dataset.classes)}")
    
    return train_loader, val_loader

def main():
    parser = argparse.ArgumentParser(description='Phase 4 Training - Unified Dataset')
    parser.add_argument('--dataset_root', type=str, 
                       default='C:/ASH_PROJECT/data/Phase4_Separate',
                       help='Root directory of Phase 4 separate dataset')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Base learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--save_dir', type=str, default='C:/ASH_PROJECT/models/phase4',
                       help='Directory to save models')
    
    args = parser.parse_args()
    
    # Check CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🔧 Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    try:
        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            args.dataset_root, args.batch_size, args.num_workers
        )
        
        num_classes = len(train_loader.dataset.classes)
        class_names = train_loader.dataset.classes
        
        logger.info(f"📊 Dataset Info:")
        logger.info(f"  Classes: {num_classes}")
        logger.info(f"  Class names: {class_names}")
        
        # Create model
        model = R2Plus1DModel(num_classes=num_classes, dropout_rate=0.5)
        model = model.to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"🤖 Model: R(2+1)D-18")
        logger.info(f"   Total params: {total_params:,}")
        logger.info(f"   Trainable params: {trainable_params:,}")
        
        # Create trainer
        trainer = Phase4Trainer(model, train_loader, val_loader, device, num_classes)
        trainer.setup_optimizer(base_lr=args.lr)
        
        # Start training
        final_model_path = trainer.train(num_epochs=args.epochs, save_dir=args.save_dir)
        
        logger.info(f"✅ Phase 4 training completed successfully!")
        logger.info(f"📁 Final model: {final_model_path}")
        
    except KeyboardInterrupt:
        logger.info("⚠️  Training interrupted by user")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()