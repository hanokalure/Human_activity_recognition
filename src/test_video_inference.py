"""
Sports Category Video Inference
🎯 Test your trained model on any input video!

Run this while training is still going on - uses the saved model.
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms
import os
import argparse

from model import get_model
from sports_category_dataset import SPORTS_ACTIONS

class VideoInference:
    """
    Video inference for sports category classification
    """
    
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.sports_actions = SPORTS_ACTIONS
        
        print(f"🚀 Loading model on: {self.device}")
        
        # Load trained model
        self.model = get_model(num_classes=len(SPORTS_ACTIONS)).to(self.device)
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"✅ Model loaded from: {model_path}")
        else:
            print(f"❌ Model not found: {model_path}")
            return
            
        self.model.eval()
        
        # Video transform (same as training)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.43216, 0.394666, 0.37645],
                std=[0.22803, 0.22145, 0.216989]
            )
        ])
    
    def load_video_clip(self, video_path, frames_per_clip=16):
        """
        Load video clip for inference
        """
        print(f"📹 Loading video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        # Read all frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
        
        # Transform frames
        transformed_frames = []
        for frame in frames:
            transformed_frame = self.transform(frame)
            transformed_frames.append(transformed_frame)
        
        # Stack to create clip [C, T, H, W]
        clip = torch.stack(transformed_frames, dim=1)  # [3, 16, 112, 112]
        
        return clip.unsqueeze(0)  # Add batch dimension [1, 3, 16, 112, 112]
    
    def predict(self, video_path, top_k=3):
        """
        Predict sports activity in video
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
            top_probs, top_indices = torch.topk(probabilities, top_k)
            
        # Format results
        results = []
        for i in range(top_k):
            sport_idx = top_indices[0][i].item()
            confidence = top_probs[0][i].item()
            sport_name = self.sports_actions[sport_idx]
            results.append({
                'sport': sport_name,
                'confidence': confidence,
                'percentage': confidence * 100
            })
        
        return results

def main():
    """
    Main inference function
    """
    parser = argparse.ArgumentParser(description='Test sports video classification')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--model', type=str, default=r'C:\ASH_PROJECT\outputs\binary_models\sports_category_model.pth',help='Path to trained model')
    
    args = parser.parse_args()
    
    print("🏟️" * 20)
    print("🏟️ SPORTS CATEGORY VIDEO INFERENCE")
    print("🏟️" * 20)
    
    # Initialize inference
    inferencer = VideoInference(args.model)
    
    # Predict
    results = inferencer.predict(args.video)
    
    if results:
        print(f"\n🎯 PREDICTIONS FOR: {os.path.basename(args.video)}")
        print("=" * 50)
        
        for i, result in enumerate(results, 1):
            confidence = result['percentage']
            sport = result['sport']
            
            if confidence > 50:
                status = "🏆 HIGH CONFIDENCE"
            elif confidence > 30:
                status = "⚠️ MEDIUM CONFIDENCE"
            else:
                status = "❌ LOW CONFIDENCE"
                
            print(f"{i}. {sport}: {confidence:.1f}% {status}")
        
        # Best prediction
        best = results[0]
        print(f"\n🎯 BEST PREDICTION: {best['sport']} ({best['percentage']:.1f}%)")
        
        if best['percentage'] > 70:
            print("✅ Very confident prediction!")
        elif best['percentage'] > 50:
            print("⚠️ Reasonable prediction")
        else:
            print("❌ Low confidence - might need more training")
    
    print(f"\n💡 Available sports: {', '.join(SPORTS_ACTIONS)}")

if __name__ == "__main__":
    main()