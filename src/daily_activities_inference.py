"""
Daily Activities Video Inference
=================================

🎯 Test your trained daily activities model on any input video!
Supports 22 different daily activity classes with detailed predictions.

Usage:
    python daily_activities_inference.py --video "path/to/video.mp4"
    python daily_activities_inference.py --video "path/to/video.mp4" --model "path/to/model.pth"
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms, models
import os
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from daily_activities_config import DAILY_ACTIVITIES, ACTIVITY_DESCRIPTIONS

class DailyActivitiesModel(nn.Module):
    """Transfer learning model for daily activities recognition"""
    
    def __init__(self, num_classes=22, model_name='r3d_18', freeze_backbone=True):
        super().__init__()
        self.num_classes = num_classes
        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        
        # Load pre-trained 3D ResNet (match training script)
        if model_name == 'r3d_18':
            self.backbone = models.video.r3d_18(pretrained=True)
        elif model_name == 'mc3_18':
            self.backbone = models.video.mc3_18(pretrained=True)
        elif model_name == 'r2plus1d_18':
            self.backbone = models.video.r2plus1d_18(pretrained=True)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Replace the final layer (match training script architecture)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        
        # Freeze backbone if specified
        if freeze_backbone:
            self._freeze_backbone()
    
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

class DailyActivitiesInference:
    """
    Video inference for daily activities classification
    """
    
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.activities = DAILY_ACTIVITIES
        self.descriptions = ACTIVITY_DESCRIPTIONS
        
        print(f"🚀 Loading daily activities model on: {self.device}")
        
        # Load checkpoint to get actual configuration
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'class_info' in checkpoint:
            num_classes = checkpoint['class_info']['num_classes']
            self.activities = checkpoint['class_info']['class_names']
            self.descriptions = checkpoint['class_info']['descriptions']
        else:
            num_classes = len(DAILY_ACTIVITIES)
            
        # Get model configuration from checkpoint
        config = checkpoint.get('config', {})
        model_name = config.get('model_name', 'r3d_18')
        freeze_backbone = config.get('freeze_backbone', True)
        self.frames_per_clip = config.get('frames_per_clip', 16)
        
        print(f"📊 Model config: {num_classes} classes, {self.frames_per_clip} frames per clip")
        
        # Load trained model
        self.model = DailyActivitiesModel(
            num_classes=num_classes,
            model_name=model_name,
            freeze_backbone=freeze_backbone
        ).to(self.device)
        
        # Load model weights
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Model loaded from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
        else:
            print(f"❌ No model_state_dict found in checkpoint")
            return
            
        print(f"📁 Loaded from: {model_path}")
            
        self.model.eval()
        
        # Video transform (same as training) - ImageNet stats for transfer learning
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.size = (224, 224)
    
    def preprocess_frame(self, frame):
        """Preprocess individual frame"""
        # Resize frame
        frame_resized = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        # Convert to float and normalize to [0,1]
        frame_normalized = frame_resized.astype(np.float32) / 255.0
        return frame_normalized
    
    def load_video_clip(self, video_path, frames_per_clip=None):
        """
        Load video clip for inference
        """
        if frames_per_clip is None:
            frames_per_clip = self.frames_per_clip
            
        print(f"📹 Loading video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ Failed to open video")
            return None
            
        frames = []
        
        # Read all frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.preprocess_frame(frame)
            frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            print("❌ No frames found in video")
            return None
            
        print(f"📊 Video info: {len(frames)} frames")
        
        # Sample frames uniformly
        if len(frames) < frames_per_clip:
            # If video too short, repeat last frame
            while len(frames) < frames_per_clip:
                frames.append(frames[-1])
        else:
            # Sample uniformly across video
            indices = np.linspace(0, len(frames)-1, frames_per_clip, dtype=int)
            frames = [frames[i] for i in indices]
        
        # Stack to create clip [T, H, W, C]
        clip = np.stack(frames, axis=0)
        
        # Convert to tensor [T, H, W, C] -> [C, T, H, W]
        clip_tensor = torch.from_numpy(clip).permute(3, 0, 1, 2)
        
        # Normalize using ImageNet statistics
        for i in range(3):
            clip_tensor[i] = (clip_tensor[i] - self.mean[i]) / self.std[i]
        
        return clip_tensor.unsqueeze(0)  # Add batch dimension [1, C, T, H, W]
    
    def predict(self, video_path, top_k=5):
        """
        Predict daily activity in video
        """
        # Load video clip
        clip = self.load_video_clip(video_path)
        if clip is None:
            return None
        
        print(f"🔍 Analyzing clip: {clip.shape}")
        
        # Move to device
        clip = clip.to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(clip)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get top predictions
            top_probs, top_indices = torch.topk(probabilities, min(top_k, len(self.activities)))
            
        # Format results
        results = []
        for i in range(min(top_k, len(self.activities))):
            activity_idx = top_indices[0][i].item()
            confidence = top_probs[0][i].item()
            activity_name = self.activities[activity_idx]
            description = self.descriptions.get(activity_name, f"Person is {activity_name}")
            
            results.append({
                'activity': activity_name,
                'confidence': confidence,
                'percentage': confidence * 100,
                'description': description
            })
        
        return results

def print_activity_categories():
    """Print available activity categories"""
    print(f"\n🎯 Available Daily Activities ({len(DAILY_ACTIVITIES)} classes):")
    print("=" * 50)
    
    # Group activities by category
    indoor = ["brushing_teeth", "typing", "eating", "cooking", "reading", "sleeping", "cleaning"]
    fitness = ["pushups", "pullups", "squats", "jumping_jacks", "yoga", "weight_lifting", "running_treadmill", "stretching"]
    outdoor = ["walking", "running", "biking", "swimming", "walking_dog", "driving", "sitting"]
    
    print("🏠 Indoor Activities:")
    for activity in indoor:
        if activity in DAILY_ACTIVITIES:
            print(f"  • {activity}")
    
    print("\n💪 Fitness Activities:")
    for activity in fitness:
        if activity in DAILY_ACTIVITIES:
            print(f"  • {activity}")
    
    print("\n🚶 Outdoor/General Activities:")
    for activity in outdoor:
        if activity in DAILY_ACTIVITIES:
            print(f"  • {activity}")

def main():
    """
    Main inference function
    """
    parser = argparse.ArgumentParser(description='Test daily activities video classification')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--model', type=str, 
                       default=r'C:\ASH_PROJECT\outputs\daily_activities_checkpoints\daily_activities_best.pth',
                       help='Path to trained model')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top predictions to show')
    
    args = parser.parse_args()
    
    print("🏠" * 25)
    print("🏠 DAILY ACTIVITIES VIDEO INFERENCE")
    print("🏠" * 25)
    
    # Check if video exists
    if not os.path.exists(args.video):
        print(f"❌ Video file not found: {args.video}")
        return
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        print(f"Available models:")
        checkpoint_dir = Path("C:/ASH_PROJECT/outputs/daily_activities_checkpoints")
        if checkpoint_dir.exists():
            for model_file in checkpoint_dir.glob("*.pth"):
                print(f"  • {model_file}")
        return
    
    # Initialize inference
    inferencer = DailyActivitiesInference(args.model)
    
    # Predict
    results = inferencer.predict(args.video, top_k=args.top_k)
    
    if results:
        print(f"\n🎯 PREDICTIONS FOR: {os.path.basename(args.video)}")
        print("=" * 60)
        
        for i, result in enumerate(results, 1):
            confidence = result['percentage']
            activity = result['activity']
            description = result['description']
            
            if confidence > 70:
                status = "🏆 VERY HIGH CONFIDENCE"
                emoji = "✅"
            elif confidence > 50:
                status = "🎯 HIGH CONFIDENCE"
                emoji = "✅"
            elif confidence > 30:
                status = "⚠️ MEDIUM CONFIDENCE"
                emoji = "⚠️"
            elif confidence > 15:
                status = "❓ LOW CONFIDENCE"
                emoji = "❌"
            else:
                status = "❌ VERY LOW CONFIDENCE"
                emoji = "❌"
                
            print(f"{i}. {emoji} {activity.replace('_', ' ').title()}: {confidence:.1f}% {status}")
            print(f"   💬 \"{description}\"")
            
        # Best prediction summary
        best = results[0]
        print(f"\n🎯 BEST PREDICTION: {best['activity'].replace('_', ' ').title()} ({best['percentage']:.1f}%)")
        print(f"💬 {best['description']}")
        
        if best['percentage'] > 70:
            print("✅ Very confident prediction!")
        elif best['percentage'] > 50:
            print("⚠️ Good prediction - reasonable confidence")
        elif best['percentage'] > 30:
            print("❓ Moderate prediction - consider context")
        else:
            print("❌ Low confidence - might need more training or different video")
    
    # Show available activities
    print_activity_categories()
    
    print(f"\n💡 Model: {os.path.basename(args.model)}")
    print(f"📹 Video: {os.path.basename(args.video)}")

if __name__ == "__main__":
    main()