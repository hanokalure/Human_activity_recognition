"""
Phase 3: Unified Daily Activities Training
=========================================

Complete training system for 19 daily activities using R(2+1)D model.
Optimized for 75-85% accuracy in 4-6 hours on RTX 3050.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
from tqdm import tqdm
import time
import os
from pathlib import Path
import json
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

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
        print(f"Target Training Time: {config['target_hours']} hours")
        
    def setup_directories(self):
        """Create output directories"""
        self.checkpoint_dir = Path(self.config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.logs_dir = Path(self.config['logs_dir'])
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Outputs: {self.checkpoint_dir}")
        
    def load_data(self):
        """Load Phase 3 unified dataset with class balancing"""
        print(f"\n📂 Loading Phase 3 unified dataset...")
        
        self.train_loader, self.val_loader, self.class_info = get_phase3_dataloaders(
            video_root=self.config['video_root'],
            video_list_file=self.config['video_list_file'],
            batch_size=self.config['batch_size'],
            frames_per_clip=self.config['frames_per_clip'],
            num_workers=self.config['num_workers'],
            val_split=self.config['val_split']
        )
        
        # Analyze class distribution for weighted loss
        self._compute_class_weights()
        
        print(f"\n✅ Phase 3 data loaded successfully:")
        print(f"  • Train batches: {len(self.train_loader)}")
        print(f"  • Val batches: {len(self.val_loader)}")
        print(f"  • Total classes: {self.class_info['num_classes']}")
        
        return self.class_info['num_classes']
    
    def _compute_class_weights(self):
        """Compute class weights for balanced training"""
        print(f"\n📊 Computing class weights for balanced training...")
        
        # Count class occurrences in training data
        class_counts = Counter()
        
        # Sample some batches to estimate class distribution
        sample_size = min(100, len(self.train_loader))
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
        
        print(f"  • Computed weights for {len(class_weights)} classes")
        print(f"  • Weight range: [{min(class_weights):.2f}, {max(class_weights):.2f}]")
        
        # Show most/least weighted classes
        activities = self.class_info['class_names']
        weighted_activities = list(zip(activities, class_weights))
        weighted_activities.sort(key=lambda x: x[1], reverse=True)
        
        print(f"  • Highest weighted: {weighted_activities[:3]}")
        print(f"  • Lowest weighted: {weighted_activities[-3:]}")
    
    def setup_model(self, num_classes):
        """Setup R(2+1)D model with optimal configuration"""
        print(f"\n🏗️ Setting up Phase 3 R(2+1)D model...")
        
        self.model = create_phase3_model(
            num_classes=num_classes,
            pretrained=True,
            freeze_layers=self.config['freeze_layers']
        ).to(self.device)
        
        # Get parameter statistics
        stats = self.model.get_parameter_stats()
        
        return stats['total_params'], stats['trainable_params']
    
    def setup_optimizer_and_scheduler(self):
        """Setup advanced optimization with differential learning rates"""
        print(f"\n⚙️ Setting up advanced optimization...")
        
        # Group parameters by layer for differential learning rates
        param_groups = []
        
        # Different learning rates for different parts
        backbone_params = []
        classifier_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'fc' in name:  # Final classifier
                    classifier_params.append(param)
                else:  # Backbone layers
                    backbone_params.append(param)
        
        # Add parameter groups with different learning rates
        if backbone_params:
            param_groups.append({
                'params': backbone_params,
                'lr': self.config['backbone_lr'],
                'name': 'backbone'
            })
        
        if classifier_params:
            param_groups.append({
                'params': classifier_params, 
                'lr': self.config['learning_rate'],
                'name': 'classifier'
            })
        
        # Create optimizer
        self.optimizer = optim.AdamW(
            param_groups,
            weight_decay=self.config['weight_decay'],
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        print(f"  • Optimizer: AdamW with differential LRs")
        print(f"  • Backbone LR: {self.config['backbone_lr']}")
        print(f"  • Classifier LR: {self.config['learning_rate']}")
        
        # Setup learning rate scheduler
        if self.config['scheduler'] == 'onecycle':
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=[self.config['backbone_lr'], self.config['learning_rate']],
                epochs=self.config['num_epochs'],
                steps_per_epoch=len(self.train_loader),
                pct_start=0.3,
                div_factor=25,
                final_div_factor=10000
            )
            print(f"  • Scheduler: OneCycleLR")
        else:
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config['num_epochs']//3,
                T_mult=2,
                eta_min=1e-6
            )
            print(f"  • Scheduler: CosineAnnealingWarmRestarts")
        
        # Setup loss function with class weighting
        self.criterion = nn.CrossEntropyLoss(
            weight=self.class_weights,
            label_smoothing=self.config.get('label_smoothing', 0.1)
        ).to(self.device)
        
        print(f"  • Loss: Weighted CrossEntropy + Label Smoothing")
        print(f"  • Gradient clipping: {self.config['gradient_clip_val']}")
    
    def train_epoch(self, epoch):
        """Train one epoch with advanced features"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Phase 3 Epoch {epoch}/{self.config['num_epochs']}")
        
        for batch_idx, (videos, labels) in enumerate(pbar):
            videos, labels = videos.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            if self.config['use_mixed_precision']:
                # Mixed precision training
                with autocast('cuda'):
                    outputs = self.model(videos)
                    loss = self.criterion(outputs, labels)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config['gradient_clip_val'] > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clip_val'])\n                \n                self.scaler.step(self.optimizer)\n                self.scaler.update()\n            else:\n                # Standard training\n                outputs = self.model(videos)\n                loss = self.criterion(outputs, labels)\n                loss.backward()\n                \n                # Gradient clipping\n                if self.config['gradient_clip_val'] > 0:\n                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clip_val'])\n                \n                self.optimizer.step()\n            \n            # Update scheduler\n            if self.config['scheduler'] == 'onecycle':\n                self.scheduler.step()\n            \n            # Statistics\n            total_loss += loss.item()\n            _, predicted = outputs.max(1)\n            total += labels.size(0)\n            correct += predicted.eq(labels).sum().item()\n            \n            # Update progress bar\n            if batch_idx % 20 == 0:\n                current_lrs = [group['lr'] for group in self.optimizer.param_groups]\n                pbar.set_postfix({\n                    'Loss': f\"{loss.item():.4f}\",\n                    'Acc': f\"{100.*correct/total:.2f}%\",\n                    'LRs': f\"{current_lrs[0]:.2e},{current_lrs[-1]:.2e}\"\n                })\n        \n        # Update scheduler (for non-OneCycle schedulers)\n        if self.config['scheduler'] != 'onecycle':\n            self.scheduler.step()\n        \n        avg_loss = total_loss / len(self.train_loader)\n        accuracy = 100. * correct / total\n        \n        return avg_loss, accuracy\n    \n    def validate(self):\n        \"\"\"Validate the model\"\"\"\n        self.model.eval()\n        total_loss = 0\n        correct = 0\n        total = 0\n        \n        # Per-class accuracy tracking\n        class_correct = list(0. for i in range(self.class_info['num_classes']))\n        class_total = list(0. for i in range(self.class_info['num_classes']))\n        \n        with torch.no_grad():\n            for videos, labels in tqdm(self.val_loader, desc=\"Phase 3 Validation\"):\n                videos, labels = videos.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)\n                \n                if self.config['use_mixed_precision']:\n                    with autocast('cuda'):\n                        outputs = self.model(videos)\n                        loss = self.criterion(outputs, labels)\n                else:\n                    outputs = self.model(videos)\n                    loss = self.criterion(outputs, labels)\n                \n                total_loss += loss.item()\n                _, predicted = outputs.max(1)\n                total += labels.size(0)\n                correct += predicted.eq(labels).sum().item()\n                \n                # Per-class accuracy\n                c = (predicted == labels).squeeze()\n                for i in range(labels.size(0)):\n                    label = labels[i]\n                    class_correct[label] += c[i].item()\n                    class_total[label] += 1\n        \n        avg_loss = total_loss / len(self.val_loader)\n        accuracy = 100. * correct / total\n        \n        # Calculate per-class accuracies\n        class_accuracies = []\n        for i in range(self.class_info['num_classes']):\n            if class_total[i] > 0:\n                class_acc = 100 * class_correct[i] / class_total[i]\n                class_accuracies.append(class_acc)\n            else:\n                class_accuracies.append(0.0)\n        \n        return avg_loss, accuracy, class_accuracies\n    \n    def save_checkpoint(self, epoch, train_acc, val_acc, class_accuracies, is_best=False):\n        \"\"\"Save comprehensive checkpoint\"\"\"\n        checkpoint = {\n            'epoch': epoch,\n            'model_state_dict': self.model.state_dict(),\n            'optimizer_state_dict': self.optimizer.state_dict(),\n            'scheduler_state_dict': self.scheduler.state_dict(),\n            'train_accuracy': train_acc,\n            'val_accuracy': val_acc,\n            'class_accuracies': class_accuracies,\n            'class_info': self.class_info,\n            'config': self.config,\n            'class_weights': self.class_weights\n        }\n        \n        # Save regular checkpoint\n        checkpoint_path = self.checkpoint_dir / f\"phase3_unified_epoch_{epoch}.pth\"\n        torch.save(checkpoint, checkpoint_path)\n        \n        # Save best model\n        if is_best:\n            best_path = self.checkpoint_dir / \"phase3_unified_best.pth\"\n            torch.save(checkpoint, best_path)\n            print(f\"💾 New best model saved: {val_acc:.2f}% validation accuracy\")\n    \n    def train(self):\n        \"\"\"Main training loop with comprehensive tracking\"\"\"\n        print(f\"\\n🚀 Starting Phase 3 Unified Training...\")\n        print(f\"🎯 Target: 75-85% accuracy in {self.config['target_hours']} hours\")\n        \n        # Load data and setup model\n        num_classes = self.load_data()\n        total_params, trainable_params = self.setup_model(num_classes)\n        self.setup_optimizer_and_scheduler()\n        \n        # Training configuration summary\n        print(f\"\\n📋 Training Configuration:\")\n        print(f\"  • Model: R(2+1)D-18 for {num_classes} activities\")\n        print(f\"  • Total parameters: {total_params:,}\")\n        print(f\"  • Trainable parameters: {trainable_params:,}\")\n        print(f\"  • Batch size: {self.config['batch_size']}\")\n        print(f\"  • Learning rates: {self.config['backbone_lr']} | {self.config['learning_rate']}\")\n        print(f\"  • Epochs: {self.config['num_epochs']}\")\n        print(f\"  • Mixed precision: {self.config['use_mixed_precision']}\")\n        print(f\"  • Frozen layers: {self.config['freeze_layers']}\")\n        \n        # Training loop\n        best_val_acc = 0\n        training_history = []\n        target_accuracy = self.config.get('target_accuracy', 75.0)\n        \n        start_time = time.time()\n        \n        for epoch in range(1, self.config['num_epochs'] + 1):\n            epoch_start = time.time()\n            \n            # Train\n            train_loss, train_acc = self.train_epoch(epoch)\n            \n            # Validate\n            val_loss, val_acc, class_accuracies = self.validate()\n            \n            epoch_time = time.time() - epoch_start\n            \n            # Log results\n            print(f\"\\nPhase 3 Epoch {epoch}/{self.config['num_epochs']} ({epoch_time:.1f}s)\")\n            print(f\"Train: Loss {train_loss:.4f}, Acc {train_acc:.2f}%\")\n            print(f\"Val:   Loss {val_loss:.4f}, Acc {val_acc:.2f}%\")\n            \n            # Show per-class performance for key activities\n            activities = self.class_info['class_names']\n            worst_classes = sorted(enumerate(class_accuracies), key=lambda x: x[1])[:3]\n            best_classes = sorted(enumerate(class_accuracies), key=lambda x: x[1], reverse=True)[:3]\n            \n            print(f\"Best: {[(activities[i], f'{acc:.1f}%') for i, acc in best_classes]}\")\n            print(f\"Worst: {[(activities[i], f'{acc:.1f}%') for i, acc in worst_classes]}\")\n            \n            # Save checkpoint\n            is_best = val_acc > best_val_acc\n            if is_best:\n                best_val_acc = val_acc\n            \n            self.save_checkpoint(epoch, train_acc, val_acc, class_accuracies, is_best)\n            \n            # Store history\n            training_history.append({\n                'epoch': epoch,\n                'train_loss': train_loss,\n                'train_acc': train_acc,\n                'val_loss': val_loss,\n                'val_acc': val_acc,\n                'class_accuracies': class_accuracies,\n                'epoch_time': epoch_time\n            })\n            \n            # Early stopping checks\n            current_time = time.time() - start_time\n            if val_acc >= target_accuracy:\n                print(f\"🎉 Target accuracy {target_accuracy}% achieved! Stopping early.\")\n                break\n            elif current_time > self.config['target_hours'] * 3600:\n                print(f\"⏰ Target training time reached. Stopping.\")\n                break\n        \n        total_time = time.time() - start_time\n        \n        # Save training history\n        history_path = self.logs_dir / \"phase3_unified_training_history.json\"\n        with open(history_path, 'w') as f:\n            json.dump(training_history, f, indent=2)\n        \n        # Final results\n        print(f\"\\n✅ Phase 3 Unified Training Complete!\")\n        print(f\"⏱️ Total time: {total_time/3600:.2f} hours\")\n        print(f\"🏆 Best validation accuracy: {best_val_acc:.2f}%\")\n        print(f\"📊 Final per-class accuracies:\")\n        \n        if training_history:\n            final_class_accs = training_history[-1]['class_accuracies']\n            activities = self.class_info['class_names']\n            for activity, acc in zip(activities, final_class_accs):\n                print(f\"  • {activity}: {acc:.1f}%\")\n        \n        # Success evaluation\n        if best_val_acc >= 80:\n            print(f\"\\n🎊 EXCELLENT: Exceeded 80% accuracy target!\")\n        elif best_val_acc >= 75:\n            print(f\"\\n✅ SUCCESS: Achieved 75%+ accuracy target!\")\n        elif best_val_acc >= 65:\n            print(f\"\\n🎯 GOOD PROGRESS: Close to target, consider more training\")\n        else:\n            print(f\"\\n⚠️ Consider hyperparameter tuning or more data\")\n        \n        return training_history, best_val_acc\n\ndef get_phase3_config():\n    \"\"\"Get optimized Phase 3 training configuration\"\"\"\n    has_cuda = torch.cuda.is_available()\n    \n    return {\n        # Data configuration\n        'video_root': r\"C:\\ASH_PROJECT\\data\\kinetics400\\videos_val\",\n        'video_list_file': r\"C:\\ASH_PROJECT\\data\\kinetics400\\kinetics400_val_list_videos.txt\",\n        'checkpoint_dir': r\"C:\\ASH_PROJECT\\outputs\\phase3_unified_checkpoints\",\n        'logs_dir': r\"C:\\ASH_PROJECT\\outputs\\phase3_unified_logs\",\n        \n        # Model configuration\n        'freeze_layers': 2,  # Freeze first 2 layer groups for efficiency\n        \n        # Training hyperparameters - optimized for RTX 3050\n        'batch_size': 4 if has_cuda else 2,\n        'frames_per_clip': 16,\n        'num_epochs': 30 if has_cuda else 5,\n        'learning_rate': 3e-4,        # Higher LR for classifier\n        'backbone_lr': 1e-5,          # Lower LR for pretrained backbone\n        'weight_decay': 1e-4,\n        'label_smoothing': 0.1,\n        'gradient_clip_val': 1.0,\n        'val_split': 0.2,\n        'target_accuracy': 75.0,      # Target accuracy\n        'target_hours': 6.0,          # Maximum training time\n        \n        # Optimization settings\n        'scheduler': 'onecycle',      # 'onecycle' or 'cosine'\n        'use_mixed_precision': has_cuda,\n        \n        # Data loading\n        'num_workers': 0,  # Single threaded for Windows stability\n    }\n\ndef main():\n    \"\"\"Main training function\"\"\"\n    print(\"🎯 Phase 3: Unified Daily Activities Training\")\n    print(\"=\" * 50)\n    \n    # Check system\n    if not torch.cuda.is_available():\n        print(\"⚠️ CUDA not available. Training will be slow on CPU.\")\n    else:\n        print(f\"✅ GPU: {torch.cuda.get_device_name(0)}\")\n        print(f\"✅ CUDA: {torch.version.cuda}\")\n        print(f\"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\")\n    \n    # Get configuration\n    config = get_phase3_config()\n    \n    print(f\"\\n🎯 Phase 3 Optimization Features:\")\n    print(f\"  • R(2+1)D-18 model (state-of-the-art video architecture)\")\n    print(f\"  • Class-weighted loss (handles imbalance)\")\n    print(f\"  • Differential learning rates (backbone vs classifier)\")\n    print(f\"  • Mixed precision training (faster + less memory)\")\n    print(f\"  • Advanced scheduling (OneCycle or Cosine)\")\n    print(f\"  • Progressive unfreezing (selective layer training)\")\n    print(f\"  • Comprehensive monitoring (per-class accuracies)\")\n    \n    # Create trainer and start training\n    trainer = Phase3Trainer(config)\n    history, best_acc = trainer.train()\n    \n    # Final summary\n    activities = get_phase3_activities()\n    print(f\"\\n🎊 Phase 3 Unified Model Complete!\")\n    print(f\"🏆 Best accuracy: {best_acc:.2f}%\")\n    print(f\"🎯 Model supports {len(activities)} daily activities:\")\n    \n    # Show activities in groups\n    activity_groups = {\n        \"Basic\": [\"eating\", \"drinking\", \"sleeping\", \"sitting\", \"walking\"],\n        \"Work\": [\"typing\", \"reading\", \"using_computer\"],\n        \"Home\": [\"cooking\"],\n        \"Exercise\": [\"running\", \"yoga\", \"exercising\", \"stretching\"],\n        \"Transport\": [\"driving\", \"biking\"],\n        \"Personal\": [\"brushing_teeth\"],\n        \"Entertainment\": [\"watching_tv\", \"playing_games\", \"dancing\"]\n    }\n    \n    for group, group_activities in activity_groups.items():\n        group_acts = [act for act in group_activities if act in activities]\n        if group_acts:\n            print(f\"  {group}: {', '.join(group_acts)}\")\n    \n    print(f\"\\n📁 Model saved to: {config['checkpoint_dir']}\")\n    print(f\"🚀 Ready for deployment!\")\n\nif __name__ == \"__main__\":\n    main()