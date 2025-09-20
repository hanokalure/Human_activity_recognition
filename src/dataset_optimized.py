import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

class OptimizedVideoTransform:
    """Optimized video transforms for faster preprocessing"""
    def __init__(self, size=(112, 112), mean=None, std=None):
        self.size = size
        self.mean = mean or [0.43216, 0.394666, 0.37645]
        self.std = std or [0.22803, 0.22145, 0.216989]
        
    def __call__(self, clip):
        """Transform video clip to tensor format [C, T, H, W]"""
        if clip is None or len(clip) == 0:
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
            frame_resized = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
            resized_frames.append(frame_resized)
            
        # Convert to tensor and normalize
        clip_tensor = torch.from_numpy(np.stack(resized_frames)).float() / 255.0
        # Permute from [T, H, W, C] to [C, T, H, W]
        clip_tensor = clip_tensor.permute(3, 0, 1, 2)
        
        # Normalize each channel
        for i in range(3):
            clip_tensor[i] = (clip_tensor[i] - self.mean[i]) / self.std[i]
            
        return clip_tensor

class FastUCF101Dataset(Dataset):
    """Optimized UCF101 dataset loader with reduced frames and efficient loading"""
    def __init__(self, data_root, annotation_path, frames_per_clip=8, train=True, transform=None):
        self.data_root = Path(data_root)
        self.annotation_path = Path(annotation_path)
        self.frames_per_clip = frames_per_clip
        self.train = train
        self.transform = transform or OptimizedVideoTransform()
        
        # Load video paths and labels
        self.video_paths, self.labels = self._load_annotations()
        
        # Create class name mapping
        self.class_names = sorted(list(set([path.parent.name for path in self.video_paths])))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}
        
        print(f"Loaded {len(self.video_paths)} {'train' if train else 'test'} videos")
        print(f"Classes: {len(self.class_names)}")
    
    def _load_annotations(self):
        """Load video paths and labels from annotation files"""
        if self.train:
            annotation_file = self.annotation_path / "trainlist01.txt"
        else:
            annotation_file = self.annotation_path / "testlist01.txt"
            
        video_paths = []
        labels = []
        
        with open(annotation_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                if self.train:
                    # Train file format: "classname/videoname.avi class_index"
                    parts = line.split()
                    video_path = parts[0]
                    class_idx = int(parts[1]) - 1  # Convert to 0-based indexing
                else:
                    # Test file format: "classname/videoname.avi"
                    video_path = line
                    class_name = video_path.split('/')[0]
                    class_idx = self.class_names.index(class_name) if hasattr(self, 'class_names') else 0
                
                full_path = self.data_root / video_path
                if full_path.exists():
                    video_paths.append(full_path)
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
                frame_indices = np.concatenate([frame_indices, frame_indices[:self.frames_per_clip - len(frame_indices)]])
        
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

def get_optimized_datasets(data_root, annotation_path, frames_per_clip=8):
    """Get optimized train and test datasets"""
    transform = OptimizedVideoTransform(size=(112, 112))
    
    print("Loading optimized training dataset...")
    train_dataset = FastUCF101Dataset(
        data_root, annotation_path, 
        frames_per_clip=frames_per_clip, 
        train=True, 
        transform=transform
    )
    
    print("Loading optimized test dataset...")
    test_dataset = FastUCF101Dataset(
        data_root, annotation_path, 
        frames_per_clip=frames_per_clip, 
        train=False, 
        transform=transform
    )
    
    return train_dataset, test_dataset

def get_optimized_dataloaders(data_root, annotation_path, batch_size=16, frames_per_clip=8, num_workers=4):
    """Get optimized dataloaders with better performance settings"""
    train_dataset, test_dataset = get_optimized_datasets(data_root, annotation_path, frames_per_clip)
    
    # Configure DataLoader parameters based on num_workers
    if num_workers > 0:
        # Multi-processing configuration
        train_kwargs = {
            'persistent_workers': True,
            'prefetch_factor': 2
        }
        test_kwargs = {
            'persistent_workers': True,
            'prefetch_factor': 2
        }
    else:
        # Single-process configuration (Windows safe)
        train_kwargs = {}
        test_kwargs = {}
    
    # Create DataLoaders with correct configuration
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True if torch.cuda.is_available() else False,
        **train_kwargs
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True if torch.cuda.is_available() else False,
        **test_kwargs
    )
    
    return train_loader, test_loader, train_dataset.class_names
