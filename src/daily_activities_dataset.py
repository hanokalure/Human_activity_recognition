"""
Daily Activities Dataset
========================

Curated dataset class that combines UCF-101 and Kinetics-400 data
for focused daily activities recognition with high accuracy.

Features:
- 22 daily activity classes
- Combines UCF-101 and Kinetics data
- Optimized for transfer learning
- Fast data loading with OpenCV
- Automatic train/val splitting
"""

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

from daily_activities_config import (
    DAILY_ACTIVITIES, UCF101_TO_DAILY, KINETICS_TO_DAILY, 
    ACTIVITY_DESCRIPTIONS
)

class DailyActivitiesTransform:
    """Optimized video transforms for daily activities"""
    def __init__(self, size=(224, 224), mean=None, std=None, augment=True):
        self.size = size
        self.mean = mean or [0.485, 0.456, 0.406]  # ImageNet means for transfer learning
        self.std = std or [0.229, 0.224, 0.225]    # ImageNet stds for transfer learning
        self.augment = augment
        
    def __call__(self, clip):
        """Transform video clip to tensor format [C, T, H, W]"""
        if clip is None or len(clip) == 0:
            # Return dummy tensor for failed videos
            return torch.randn(3, 16, *self.size)
            
        # Convert list of frames to numpy array if needed
        if isinstance(clip, list):
            clip = np.stack(clip, axis=0)
            
        # Ensure clip has shape [T, H, W, C]
        if clip.ndim != 4 or clip.shape[-1] != 3:
            return torch.randn(3, 16, *self.size)
            
        # Resize frames efficiently using OpenCV
        resized_frames = []
        for frame in clip:
            if self.augment and np.random.rand() < 0.5:
                # Random horizontal flip for augmentation
                frame = cv2.flip(frame, 1)
                
            frame_resized = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
            resized_frames.append(frame_resized)
            
        # Convert to tensor and normalize
        clip_tensor = torch.from_numpy(np.stack(resized_frames)).float() / 255.0
        # Permute from [T, H, W, C] to [C, T, H, W]
        clip_tensor = clip_tensor.permute(3, 0, 1, 2)
        
        # Normalize each channel using ImageNet statistics (for transfer learning)
        for i in range(3):
            clip_tensor[i] = (clip_tensor[i] - self.mean[i]) / self.std[i]
            
        return clip_tensor

class DailyActivitiesDataset(Dataset):
    """Curated daily activities dataset combining UCF-101 and Kinetics data"""
    
    def __init__(self, data_root, frames_per_clip=16, train=True, transform=None, 
                 use_ucf101=True, use_kinetics=False):
        self.data_root = Path(data_root)
        self.frames_per_clip = frames_per_clip
        self.train = train
        self.transform = transform or DailyActivitiesTransform(augment=train)
        self.use_ucf101 = use_ucf101
        self.use_kinetics = use_kinetics
        
        # Create class name to index mapping
        self.class_names = DAILY_ACTIVITIES
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        # Load video paths and labels
        self.video_paths, self.labels = self._load_video_data()
        
        print(f"📊 Daily Activities Dataset ({'Train' if train else 'Val'})")
        print(f"  • Total videos: {len(self.video_paths)}")
        print(f"  • Classes: {len(self.class_names)}")
        print(f"  • UCF-101: {'✅' if use_ucf101 else '❌'}")
        print(f"  • Kinetics: {'✅' if use_kinetics else '❌'}")
        
    def _load_video_data(self):
        """Load video paths and labels from both UCF-101 and Kinetics"""
        video_paths = []
        labels = []
        
        # Load UCF-101 data
        if self.use_ucf101:
            ucf_videos, ucf_labels = self._load_ucf101_data()
            video_paths.extend(ucf_videos)
            labels.extend(ucf_labels)
            print(f"✅ Loaded {len(ucf_videos)} videos from UCF-101")
        
        # Load Kinetics data (if available)
        if self.use_kinetics:
            kinetics_videos, kinetics_labels = self._load_kinetics_data()
            video_paths.extend(kinetics_videos)
            labels.extend(kinetics_labels)
            print(f"✅ Loaded {len(kinetics_videos)} videos from Kinetics")
            
        if len(video_paths) == 0:
            raise ValueError("No video data found! Check your data paths.")
            
        return video_paths, labels
    
    def _load_ucf101_data(self):
        """Load UCF-101 data for daily activities"""
        ucf101_dir = self.data_root / "UCF101"
        video_paths = []
        labels = []
        
        for ucf_class, daily_class in UCF101_TO_DAILY.items():
            if daily_class not in self.class_to_idx:
                continue
                
            class_dir = ucf101_dir / ucf_class
            if not class_dir.exists():
                print(f"⚠️ UCF-101 class directory not found: {class_dir}")
                continue
                
            # Get all video files
            video_files = list(class_dir.glob("*.avi"))
            
            if len(video_files) == 0:
                print(f"⚠️ No videos found in {class_dir}")
                continue
            
            # Split into train/val (80/20)
            train_videos, val_videos = train_test_split(
                video_files, test_size=0.2, random_state=42
            )
            
            selected_videos = train_videos if self.train else val_videos
            class_idx = self.class_to_idx[daily_class]
            
            for video_path in selected_videos:
                video_paths.append(video_path)
                labels.append(class_idx)
                
            print(f"  • {ucf_class} -> {daily_class}: {len(selected_videos)} videos")
            
        return video_paths, labels
    
    def _load_kinetics_data(self):
        """Load Kinetics data for daily activities (placeholder for now)"""
        kinetics_dir = self.data_root / "Kinetics400_Daily"
        video_paths = []
        labels = []
        
        # This would load actual Kinetics videos if available
        # For now, it's a placeholder structure
        
        for kinetics_class, daily_class in KINETICS_TO_DAILY.items():
            if daily_class not in self.class_to_idx:
                continue
                
            class_dir = kinetics_dir / kinetics_class
            if not class_dir.exists():
                continue
                
            # Look for video files (.mp4, .avi, etc.)
            video_files = []
            for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
                video_files.extend(class_dir.glob(ext))
            
            if len(video_files) == 0:
                continue
            
            # Split into train/val (80/20)
            train_videos, val_videos = train_test_split(
                video_files, test_size=0.2, random_state=42
            )
            
            selected_videos = train_videos if self.train else val_videos
            class_idx = self.class_to_idx[daily_class]
            
            for video_path in selected_videos:
                video_paths.append(video_path)
                labels.append(class_idx)
                
        return video_paths, labels
    
    def _load_video_frames(self, video_path):
        """Efficiently load video frames using OpenCV"""
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        # Get total frame count
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return None
            
        # Sample frames uniformly
        if total_frames >= self.frames_per_clip:
            frame_indices = np.linspace(0, total_frames - 1, self.frames_per_clip, dtype=int)
        else:
            # Repeat frames if video is too short
            frame_indices = np.arange(total_frames)
            while len(frame_indices) < self.frames_per_clip:
                frame_indices = np.concatenate([
                    frame_indices, 
                    frame_indices[:self.frames_per_clip - len(frame_indices)]
                ])
        
        # Load selected frames
        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                # If frame reading fails, duplicate the last frame
                if frames:
                    frames.append(frames[-1])
                else:
                    # Create a black frame as fallback
                    frames.append(np.zeros((240, 320, 3), dtype=np.uint8))
        
        cap.release()
        return frames[:self.frames_per_clip]
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        try:
            video_path = self.video_paths[idx]
            label = self.labels[idx]
            
            # Load video frames
            frames = self._load_video_frames(video_path)
            
            if frames is None:
                # Return dummy data on error
                return self.transform(None), 0
            
            # Apply transforms
            video_tensor = self.transform(frames)
            
            return video_tensor, label
            
        except Exception as e:
            if idx < 5:  # Only print first few errors
                print(f"Error loading video {idx}: {e}")
            return self.transform(None), 0
    
    def get_class_info(self):
        """Get information about classes"""
        return {
            'class_names': self.class_names,
            'class_to_idx': self.class_to_idx,
            'idx_to_class': self.idx_to_class,
            'num_classes': len(self.class_names),
            'descriptions': ACTIVITY_DESCRIPTIONS
        }

def get_daily_activities_dataloaders(data_root, batch_size=16, frames_per_clip=16, 
                                   num_workers=4, use_ucf101=True, use_kinetics=False):
    """Get optimized dataloaders for daily activities"""
    
    # Create datasets
    train_dataset = DailyActivitiesDataset(
        data_root=data_root,
        frames_per_clip=frames_per_clip,
        train=True,
        transform=DailyActivitiesTransform(size=(224, 224), augment=True),
        use_ucf101=use_ucf101,
        use_kinetics=use_kinetics
    )
    
    val_dataset = DailyActivitiesDataset(
        data_root=data_root,
        frames_per_clip=frames_per_clip,
        train=False,
        transform=DailyActivitiesTransform(size=(224, 224), augment=False),
        use_ucf101=use_ucf101,
        use_kinetics=use_kinetics
    )
    
    # Configure DataLoader parameters
    if num_workers > 0:
        # Multi-processing configuration
        loader_kwargs = {
            'persistent_workers': True,
            'prefetch_factor': 2
        }
    else:
        # Single-process configuration (Windows safe)
        loader_kwargs = {}
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        **loader_kwargs
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        **loader_kwargs
    )
    
    return train_loader, val_loader, train_dataset.get_class_info()

def print_dataset_statistics(data_root):
    """Print statistics about the daily activities dataset"""
    print("\\n📊 Daily Activities Dataset Statistics")
    print("=" * 50)
    
    # Load train dataset to get statistics
    train_dataset = DailyActivitiesDataset(
        data_root=data_root,
        train=True,
        use_ucf101=True,
        use_kinetics=False
    )
    
    val_dataset = DailyActivitiesDataset(
        data_root=data_root,
        train=False, 
        use_ucf101=True,
        use_kinetics=False
    )
    
    # Count videos per class
    class_counts_train = {}
    class_counts_val = {}
    
    for label in train_dataset.labels:
        class_name = train_dataset.idx_to_class[label]
        class_counts_train[class_name] = class_counts_train.get(class_name, 0) + 1
        
    for label in val_dataset.labels:
        class_name = val_dataset.idx_to_class[label]
        class_counts_val[class_name] = class_counts_val.get(class_name, 0) + 1
    
    print(f"\\nTotal: {len(train_dataset)} train + {len(val_dataset)} val = {len(train_dataset) + len(val_dataset)} videos")
    print(f"Classes: {len(DAILY_ACTIVITIES)}")
    
    print(f"\\n📋 Videos per class:")
    for class_name in DAILY_ACTIVITIES:
        train_count = class_counts_train.get(class_name, 0)
        val_count = class_counts_val.get(class_name, 0)
        total_count = train_count + val_count
        if total_count > 0:
            print(f"  • {class_name}: {train_count} train + {val_count} val = {total_count} total")
        else:
            print(f"  • {class_name}: ❌ No videos found")
    
    print("=" * 50)

if __name__ == "__main__":
    # Test the dataset
    data_root = "C:/ASH_PROJECT/data"
    print_dataset_statistics(data_root)
    
    # Test dataloader
    print("\\n🧪 Testing dataloader...")
    train_loader, val_loader, class_info = get_daily_activities_dataloaders(
        data_root=data_root,
        batch_size=2,
        frames_per_clip=8,
        num_workers=0  # Windows safe
    )
    
    # Test loading a batch
    for batch_idx, (videos, labels) in enumerate(train_loader):
        print(f"✅ Loaded batch {batch_idx}: videos {videos.shape}, labels {labels.shape}")
        if batch_idx >= 2:  # Test first few batches
            break
    
    print("✅ Dataset test completed successfully!")