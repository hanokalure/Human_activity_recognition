"""
Phase 3: Unified Daily Activities Dataset
========================================

Optimized dataset loader for R(2+1)D-34 training on 19 daily activities.
Uses Kinetics-400 VAL split with advanced preprocessing and class balancing.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np
from pathlib import Path
import random
from collections import defaultdict, Counter
import json

from phase3_class_mapping import (
    get_phase3_activities,
    get_phase3_class_mapping, 
    create_activity_to_index_mapping
)

class Phase3UnifiedDataset(Dataset):
    """Unified dataset for 19 daily activities from Kinetics-400"""
    
    def __init__(self, video_root, video_list_file, transform=None, 
                 frames_per_clip=16, clip_duration=2.0):
        """
        Args:
            video_root: Path to Kinetics videos directory
            video_list_file: Path to kinetics400_val_list_videos.txt
            transform: Video transforms optimized for R(2+1)D
            frames_per_clip: Number of frames to sample per video
            clip_duration: Duration in seconds for each clip
        """
        self.video_root = Path(video_root)
        self.transform = transform
        self.frames_per_clip = frames_per_clip
        self.clip_duration = clip_duration
        
        # Load Phase 3 mappings
        self.kinetics_to_activity = get_phase3_class_mapping()
        self.phase3_activities = get_phase3_activities()
        self.activity_to_idx = create_activity_to_index_mapping()
        self.idx_to_activity = {idx: activity for activity, idx in self.activity_to_idx.items()}
        
        # Load and filter video data
        self.video_data = self._load_video_list(video_list_file)
        
        print(f"✅ Phase 3 Dataset loaded: {len(self.video_data)} videos")
        print(f"📊 Activities: {len(self.phase3_activities)}")
        
        # Analyze dataset statistics
        self._print_dataset_stats()
    
    def _load_video_list(self, video_list_file):
        """Load and filter videos for Phase 3 activities"""
        video_data = []
        activity_counts = defaultdict(int)
        
        print(f"🔍 Scanning Kinetics videos for Phase 3 activities...")
        
        with open(video_list_file, 'r') as f:
            for line_num, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) >= 2:
                    video_filename = parts[0]
                    kinetics_class_id = int(parts[1])
                    
                    # Check if this Kinetics class maps to our Phase 3 activities
                    if kinetics_class_id in self.kinetics_to_activity:
                        activity = self.kinetics_to_activity[kinetics_class_id]
                        video_path = self.video_root / video_filename
                        
                        # Verify video file exists
                        if video_path.exists():
                            video_data.append({
                                'path': video_path,
                                'activity': activity,
                                'activity_idx': self.activity_to_idx[activity],
                                'kinetics_class_id': kinetics_class_id
                            })
                            activity_counts[activity] += 1
                
                # Progress update
                if line_num % 1000 == 0:
                    print(f"  Processed {line_num} lines, found {len(video_data)} videos...")
        
        print(f"\n📊 Videos found per activity:")
        for activity, count in sorted(activity_counts.items()):
            print(f"  • {activity}: {count} videos")
        
        return video_data
    
    def _print_dataset_stats(self):
        """Print comprehensive dataset statistics"""
        activity_counts = Counter([item['activity'] for item in self.video_data])
        
        print(f"\n📈 Phase 3 Dataset Statistics:")
        print(f"  • Total videos: {len(self.video_data)}")
        print(f"  • Activities: {len(activity_counts)}")
        print(f"  • Frames per clip: {self.frames_per_clip}")
        
        # Class imbalance analysis
        counts = list(activity_counts.values())
        if counts:
            min_count = min(counts)
            max_count = max(counts)
            mean_count = np.mean(counts)
            
            print(f"  • Videos per class: {min_count} - {max_count} (avg: {mean_count:.1f})")
            
            if min_count > 0:
                imbalance_ratio = max_count / min_count
                print(f"  • Class imbalance ratio: {imbalance_ratio:.2f}")
                
                if imbalance_ratio > 3.0:
                    print(f"  ⚠️  High class imbalance detected")
                elif imbalance_ratio > 2.0:
                    print(f"  ⚠️  Moderate class imbalance")
                else:
                    print(f"  ✅ Good class balance")
        
        # Find missing activities
        missing_activities = []
        for activity in self.phase3_activities:
            if activity not in activity_counts:
                missing_activities.append(activity)
        
        if missing_activities:
            print(f"  ❌ Missing activities: {missing_activities}")
        else:
            print(f"  ✅ All {len(self.phase3_activities)} activities have data")
    
    def __len__(self):
        return len(self.video_data)
    
    def __getitem__(self, idx):
        """Load and process a video clip"""
        video_info = self.video_data[idx]
        video_path = video_info['path']
        label = video_info['activity_idx']
        
        try:
            # Load video frames
            frames = self._load_video_frames(video_path)
            
            if frames is None or len(frames) == 0:
                # Fallback to random video if current fails
                return self.__getitem__(random.randint(0, len(self.video_data) - 1))
            
            # Apply transforms
            if self.transform:
                frames = self.transform(frames)
            
            return frames, label
            
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            # Fallback to random video
            return self.__getitem__(random.randint(0, len(self.video_data) - 1))
    
    def _load_video_frames(self, video_path):
        """Load frames from video optimized for R(2+1)D"""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            return None
        
        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps <= 0 or total_frames <= 0:
            cap.release()
            return None
        
        # Calculate temporal sampling for R(2+1)D
        clip_frames = int(fps * self.clip_duration)
        if clip_frames > total_frames:
            clip_frames = total_frames
        
        # Uniform temporal sampling
        if clip_frames >= self.frames_per_clip:
            frame_indices = np.linspace(0, clip_frames - 1, self.frames_per_clip, dtype=int)
        else:
            # Repeat frames if video is too short
            frame_indices = list(range(clip_frames))
            while len(frame_indices) < self.frames_per_clip:
                frame_indices.extend(frame_indices[:min(self.frames_per_clip - len(frame_indices), len(frame_indices))])
            frame_indices = frame_indices[:self.frames_per_clip]
        
        # Load selected frames
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Convert BGR to RGB for consistency
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                # Use last valid frame if available
                if frames:
                    frames.append(frames[-1])
        
        cap.release()
        
        # Ensure we have exactly the right number of frames
        while len(frames) < self.frames_per_clip:
            if frames:
                frames.append(frames[-1])
            else:
                # Create black frame as last resort
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
        
        return np.array(frames[:self.frames_per_clip])

def get_r2plus1d_transforms(train=True, input_size=224):
    """Get optimized transforms for R(2+1)D model"""
    
    def video_normalize(x):
        """Normalize video tensor with shape (C, T, H, W) for R(2+1)D"""
        C, T, H, W = x.shape
        x_reshaped = x.view(C, -1)  # (C, T*H*W)
        
        # ImageNet normalization (standard for video models)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1)
        
        x_normalized = (x_reshaped - mean) / std
        return x_normalized.view(C, T, H, W)
    
    if train:
        # Training transforms with augmentation
        transform = transforms.Compose([
            transforms.Lambda(lambda x: torch.FloatTensor(x)),  # Convert to tensor
            transforms.Lambda(lambda x: x.permute(3, 0, 1, 2)),  # (T,H,W,C) -> (C,T,H,W)
            transforms.Lambda(lambda x: torch.nn.functional.interpolate(
                x, size=(input_size, input_size), mode='bilinear', align_corners=False
            )),  # Spatial resize
            transforms.Lambda(lambda x: x / 255.0),  # Normalize to [0,1]
            transforms.Lambda(video_normalize),  # ImageNet normalization
            # Temporal data augmentation (random start position)
            transforms.Lambda(lambda x: temporal_augment(x)),
        ])
    else:
        # Validation transforms (no augmentation)
        transform = transforms.Compose([
            transforms.Lambda(lambda x: torch.FloatTensor(x)),
            transforms.Lambda(lambda x: x.permute(3, 0, 1, 2)),  # (T,H,W,C) -> (C,T,H,W)
            transforms.Lambda(lambda x: torch.nn.functional.interpolate(
                x, size=(input_size, input_size), mode='bilinear', align_corners=False
            )),
            transforms.Lambda(lambda x: x / 255.0),
            transforms.Lambda(video_normalize),
        ])
    
    return transform

def temporal_augment(video_tensor):
    """Apply temporal augmentation for training"""
    C, T, H, W = video_tensor.shape
    
    # Random temporal crop (if we have extra frames)
    if T > 16:
        start_t = random.randint(0, T - 16)
        video_tensor = video_tensor[:, start_t:start_t+16, :, :]
    
    return video_tensor

def get_phase3_dataloaders(video_root, video_list_file, batch_size=4, 
                          frames_per_clip=16, num_workers=0, val_split=0.2):
    """Create optimized dataloaders for Phase 3 training"""
    
    print(f"\n📂 Creating Phase 3 dataloaders...")
    
    # Load full dataset
    full_dataset = Phase3UnifiedDataset(
        video_root=video_root,
        video_list_file=video_list_file,
        transform=None,  # Will add transforms later
        frames_per_clip=frames_per_clip
    )
    
    if len(full_dataset) == 0:
        raise ValueError("❌ No valid videos found for Phase 3 activities!")
    
    # Split dataset
    total_size = len(full_dataset)
    val_size = int(val_split * total_size)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Reproducible split
    )
    
    # Add transforms
    train_transform = get_r2plus1d_transforms(train=True)
    val_transform = get_r2plus1d_transforms(train=False)
    
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    # Create dataloaders optimized for R(2+1)D
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=num_workers > 0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0
    )
    
    # Create class information
    class_info = {
        'num_classes': len(full_dataset.phase3_activities),
        'class_names': full_dataset.phase3_activities,
        'activity_to_idx': full_dataset.activity_to_idx,
        'idx_to_activity': full_dataset.idx_to_activity
    }
    
    print(f"\n✅ Phase 3 dataloaders created:")
    print(f"  • Train: {len(train_loader)} batches ({len(train_dataset)} videos)")
    print(f"  • Val: {len(val_loader)} batches ({len(val_dataset)} videos)")
    print(f"  • Classes: {class_info['num_classes']}")
    
    return train_loader, val_loader, class_info

if __name__ == "__main__":
    # Test the dataset loader
    video_root = r"C:\ASH_PROJECT\data\kinetics400\videos_val"
    video_list_file = r"C:\ASH_PROJECT\data\kinetics400\kinetics400_val_list_videos.txt"
    
    print("🧪 Testing Phase 3 unified dataset...")
    
    try:
        train_loader, val_loader, class_info = get_phase3_dataloaders(
            video_root=video_root,
            video_list_file=video_list_file,
            batch_size=2,
            frames_per_clip=16,
            num_workers=0
        )
        
        print(f"\n✅ Dataset test successful!")
        print(f"📊 Class info: {class_info['num_classes']} activities")
        
        # Test loading one batch
        print(f"\n🔍 Testing batch loading for R(2+1)D...")
        for batch_idx, (videos, labels) in enumerate(train_loader):
            print(f"  • Batch shape: {videos.shape}")
            print(f"  • Expected R(2+1)D input: (B, C, T, H, W) = {videos.shape}")
            print(f"  • Labels: {labels}")
            print(f"  • Video range: [{videos.min():.3f}, {videos.max():.3f}]")
            
            # Verify tensor format for R(2+1)D
            if len(videos.shape) == 5 and videos.shape[1] == 3:  # (B, C, T, H, W)
                print(f"  ✅ Correct tensor format for R(2+1)D!")
            else:
                print(f"  ❌ Incorrect tensor format!")
            break
        
        print(f"\n🎉 Phase 3 dataset ready for R(2+1)D training!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()