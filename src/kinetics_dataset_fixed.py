"""
Kinetics-400 Dataset Loader for Phase 2 Daily Activities - FIXED
================================================================

Fixed version with proper tensor dimensions for video transforms.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np
from pathlib import Path
import json
from collections import defaultdict
import random

from kinetics_class_mapping import (
    get_kinetics_class_mapping, 
    get_all_activities,
    PHASE_1_ACTIVITIES,
    PHASE_2_ACTIVITIES
)

class KineticsVideoDataset(Dataset):
    """Dataset for loading Kinetics-400 videos for daily activities"""
    
    def __init__(self, video_root, video_list_file, transform=None, 
                 frames_per_clip=16, clip_duration=2.0):
        """
        Args:
            video_root: Path to Kinetics videos directory
            video_list_file: Path to kinetics400_val_list_videos.txt
            transform: Video transforms
            frames_per_clip: Number of frames to sample per video
            clip_duration: Duration in seconds for each clip
        """
        self.video_root = Path(video_root)
        self.transform = transform
        self.frames_per_clip = frames_per_clip
        self.clip_duration = clip_duration
        
        # Load class mappings
        self.kinetics_to_daily = get_kinetics_class_mapping()
        self.all_activities = get_all_activities()
        self.phase_2_activities = PHASE_2_ACTIVITIES
        
        # Create activity to index mapping (for 19 total classes)
        self.activity_to_idx = {activity: idx for idx, activity in enumerate(self.all_activities)}
        self.idx_to_activity = {idx: activity for activity, idx in self.activity_to_idx.items()}
        
        # Load video list and filter for our target activities
        self.video_data = self._load_video_list(video_list_file)
        
        print(f"✅ Loaded {len(self.video_data)} Kinetics videos for Phase 2")
        print(f"📊 Target activities: {len(self.phase_2_activities)}")
        
        # Print dataset statistics
        self._print_dataset_stats()
    
    def _load_video_list(self, video_list_file):
        """Load and filter video list for target daily activities"""
        video_data = []
        activity_counts = defaultdict(int)
        
        with open(video_list_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    video_filename = parts[0]
                    class_id = int(parts[1])
                    
                    # Check if this class maps to one of our target activities
                    if class_id in self.kinetics_to_daily:
                        daily_activity = self.kinetics_to_daily[class_id]
                        
                        # Only include Phase 2 activities
                        if daily_activity in self.phase_2_activities:
                            video_path = self.video_root / video_filename
                            
                            # Check if video file exists
                            if video_path.exists():
                                video_data.append({
                                    'path': video_path,
                                    'activity': daily_activity,
                                    'activity_idx': self.activity_to_idx[daily_activity],
                                    'kinetics_class': class_id
                                })
                                activity_counts[daily_activity] += 1
        
        print(f"\n📊 Videos per activity:")
        for activity, count in sorted(activity_counts.items()):
            print(f"  • {activity}: {count} videos")
        
        return video_data
    
    def _print_dataset_stats(self):
        """Print dataset statistics"""
        activity_counts = defaultdict(int)
        for item in self.video_data:
            activity_counts[item['activity']] += 1
        
        print(f"\n📈 Dataset Statistics:")
        print(f"  • Total videos: {len(self.video_data)}")
        print(f"  • Activities: {len(activity_counts)}")
        print(f"  • Frames per clip: {self.frames_per_clip}")
        
        # Check for class imbalance
        counts = list(activity_counts.values())
        if counts:
            min_count, max_count = min(counts), max(counts)
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            print(f"  • Class imbalance ratio: {imbalance_ratio:.2f}")
            
            if imbalance_ratio > 3.0:
                print(f"  ⚠️  High class imbalance detected!")
    
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
                # Return dummy data to avoid infinite recursion
                frames = np.zeros((self.frames_per_clip, 224, 224, 3), dtype=np.uint8)
            
            # Apply transforms
            if self.transform:
                frames = self.transform(frames)
            
            return frames, label
            
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            # Return dummy data instead of recursive call
            frames = np.zeros((self.frames_per_clip, 224, 224, 3), dtype=np.uint8)
            if self.transform:
                frames = self.transform(frames)
            return frames, label
    
    def _load_video_frames(self, video_path):
        """Load frames from video file using OpenCV"""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            return None
        
        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps <= 0 or total_frames <= 0:
            cap.release()
            return None
        
        # Calculate frame sampling
        clip_frames = int(fps * self.clip_duration)
        if clip_frames > total_frames:
            clip_frames = total_frames
        
        # Sample frames evenly across the clip
        if clip_frames > self.frames_per_clip:
            frame_indices = np.linspace(0, clip_frames - 1, self.frames_per_clip, dtype=int)
        else:
            frame_indices = list(range(clip_frames))
            # Pad with last frame if needed
            while len(frame_indices) < self.frames_per_clip:
                frame_indices.append(frame_indices[-1] if frame_indices else 0)
        
        # Load selected frames
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Convert BGR to RGB and resize
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)
            else:
                # Use last valid frame if available, or create dummy frame
                if frames:
                    frames.append(frames[-1])
                else:
                    frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
        
        cap.release()
        
        # Ensure we have the right number of frames
        while len(frames) < self.frames_per_clip:
            if frames:
                frames.append(frames[-1])
            else:
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
        
        return np.array(frames[:self.frames_per_clip])

def process_video_tensor(frames):
    """Process video frames to correct tensor format for R3D models"""
    # Input: (T, H, W, C) numpy array
    # Output: (C, T, H, W) tensor
    
    # Convert to tensor
    frames = torch.FloatTensor(frames)  # (T, H, W, C)
    
    # Permute to (C, T, H, W) 
    frames = frames.permute(3, 0, 1, 2)  # (C, T, H, W)
    
    # Normalize to [0,1]
    frames = frames / 255.0
    
    # Apply ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
    frames = (frames - mean) / std
    
    return frames

def get_kinetics_dataloaders(video_root, video_list_file, batch_size=8, 
                           frames_per_clip=16, num_workers=0, val_split=0.2):
    """Create train/validation dataloaders for Kinetics data"""
    
    # Create transform
    transform = transforms.Lambda(process_video_tensor)
    
    # Load full dataset
    full_dataset = KineticsVideoDataset(
        video_root=video_root,
        video_list_file=video_list_file,
        transform=transform,
        frames_per_clip=frames_per_clip
    )
    
    if len(full_dataset) == 0:
        raise ValueError("No valid videos found in Kinetics dataset!")
    
    # Split dataset
    total_size = len(full_dataset)
    val_size = int(val_split * total_size)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"\n✅ Created Kinetics dataloaders:")
    print(f"  • Train: {len(train_loader)} batches ({len(train_dataset)} videos)")
    print(f"  • Val: {len(val_loader)} batches ({len(val_dataset)} videos)")
    
    # Return class information
    class_info = {
        'num_classes': len(full_dataset.all_activities),
        'class_names': full_dataset.all_activities,
        'activity_to_idx': full_dataset.activity_to_idx,
        'phase_2_activities': full_dataset.phase_2_activities
    }
    
    return train_loader, val_loader, class_info

if __name__ == "__main__":
    # Test the fixed dataset loader
    video_root = r"C:\ASH_PROJECT\data\kinetics400\videos_val"
    video_list_file = r"C:\ASH_PROJECT\data\kinetics400\kinetics400_val_list_videos.txt"
    
    print("🧪 Testing FIXED Kinetics dataset loader...")
    
    try:
        train_loader, val_loader, class_info = get_kinetics_dataloaders(
            video_root=video_root,
            video_list_file=video_list_file,
            batch_size=2,
            frames_per_clip=8,  # Smaller for testing
            num_workers=0
        )
        
        print(f"\n✅ Dataset test successful!")
        print(f"📊 Class info: {class_info['num_classes']} classes")
        
        # Test loading one batch
        print("\n🔍 Testing batch loading...")
        for batch_idx, (videos, labels) in enumerate(train_loader):
            print(f"  • Batch shape: {videos.shape}")
            print(f"  • Labels: {labels}")
            print(f"  • Video range: [{videos.min():.3f}, {videos.max():.3f}]")
            break
        
        print("\n🎉 All tests passed! Ready for training!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()