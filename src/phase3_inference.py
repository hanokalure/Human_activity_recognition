"""
Phase 3: Daily Activities Inference Script
==========================================

Test your trained Phase 3 model on videos to predict daily activities.
Supports single videos, batch processing, and webcam input.

Usage:
    python phase3_inference.py --video path/to/video.mp4
    python phase3_inference.py --video path/to/folder/ --batch
    python phase3_inference.py --webcam
"""

import torch
import cv2
import numpy as np
import argparse
import os
from pathlib import Path
import time
import json

from phase3_r2plus1d_model import load_phase3_checkpoint
from phase3_unified_dataset import get_r2plus1d_transforms
from phase3_class_mapping import get_phase3_activities

class Phase3Predictor:
    """Phase 3 model predictor for daily activities"""
    
    def __init__(self, model_path, device=None):
        """Initialize predictor with trained model"""
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🔧 Initializing Phase 3 Predictor...")
        print(f"Device: {self.device}")
        
        # Load trained model
        print(f"📥 Loading model from: {model_path}")
        self.model = load_phase3_checkpoint(model_path, num_classes=19)
        self.model.to(self.device)
        self.model.eval()
        
        # Load class names
        self.activities = get_phase3_activities()
        self.num_classes = len(self.activities)
        
        # Setup transforms
        self.transform = get_r2plus1d_transforms(train=False)
        
        print(f"✅ Model loaded successfully!")
        print(f"📊 Activities: {self.num_classes} classes")
        print(f"🎯 Ready for inference!")
    
    def preprocess_video(self, video_path, frames_per_clip=16):
        """Preprocess video for model input"""
        print(f"🎬 Processing video: {os.path.basename(video_path)}")
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps <= 0 or total_frames <= 0:
            cap.release()
            raise ValueError(f"Invalid video file: {video_path}")
        
        print(f"  📹 Video info: {total_frames} frames, {fps:.1f} FPS, {total_frames/fps:.1f}s")
        
        # Sample frames uniformly
        if total_frames >= frames_per_clip:
            frame_indices = np.linspace(0, total_frames - 1, frames_per_clip, dtype=int)
        else:
            # Repeat frames if video is too short
            frame_indices = list(range(total_frames))
            while len(frame_indices) < frames_per_clip:
                frame_indices.extend(frame_indices[:min(frames_per_clip - len(frame_indices), len(frame_indices))])
            frame_indices = frame_indices[:frames_per_clip]
        
        # Load selected frames
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                if frames:
                    frames.append(frames[-1])  # Use last valid frame
        
        cap.release()
        
        # Ensure we have enough frames
        while len(frames) < frames_per_clip:
            if frames:
                frames.append(frames[-1])
            else:
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
        
        frames = np.array(frames[:frames_per_clip])
        print(f"  ✅ Preprocessed: {frames.shape} -> applying transforms...")
        
        # Apply transforms
        frames_tensor = self.transform(frames)  # Should be (C, T, H, W)
        frames_tensor = frames_tensor.unsqueeze(0)  # Add batch dimension -> (1, C, T, H, W)
        
        print(f"  🔧 Final tensor shape: {frames_tensor.shape}")
        return frames_tensor
    
    def predict(self, video_path, top_k=5):
        """Predict activity for a single video"""
        
        # Preprocess video
        video_tensor = self.preprocess_video(video_path)
        video_tensor = video_tensor.to(self.device)
        
        # Run inference
        print(f"🤖 Running inference...")
        start_time = time.time()
        
        with torch.no_grad():
            if torch.cuda.is_available():
                with torch.amp.autocast('cuda'):
                    outputs = self.model(video_tensor)
            else:
                outputs = self.model(video_tensor)
        
        inference_time = time.time() - start_time
        
        # Get predictions
        probabilities = torch.softmax(outputs, dim=1)
        confidence_scores = probabilities[0]  # Remove batch dimension
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(confidence_scores, top_k)
        
        predictions = []
        for i in range(top_k):
            activity = self.activities[top_indices[i].item()]
            confidence = top_probs[i].item() * 100
            predictions.append({
                'activity': activity,
                'confidence': confidence,
                'rank': i + 1
            })
        
        result = {
            'video_path': str(video_path),
            'video_name': os.path.basename(video_path),
            'inference_time': inference_time,
            'predictions': predictions,
            'top_prediction': predictions[0] if predictions else None
        }
        
        return result
    
    def predict_batch(self, video_folder, output_file=None):
        """Predict activities for all videos in a folder"""
        video_folder = Path(video_folder)
        
        # Find video files
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(video_folder.glob(f"*{ext}"))
            video_files.extend(video_folder.glob(f"*{ext.upper()}"))
        
        if not video_files:
            print(f"❌ No video files found in: {video_folder}")
            return []
        
        print(f"📁 Found {len(video_files)} video files")
        
        results = []
        for i, video_file in enumerate(video_files, 1):
            print(f"\n🎬 Processing video {i}/{len(video_files)}: {video_file.name}")
            
            try:
                result = self.predict(video_file)
                results.append(result)
                
                # Show result
                top_pred = result['top_prediction']
                print(f"  🎯 Result: {top_pred['activity']} ({top_pred['confidence']:.1f}%)")
                
            except Exception as e:
                print(f"  ❌ Error processing {video_file.name}: {e}")
                continue
        
        # Save results if requested
        if output_file:
            output_path = Path(output_file)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Results saved to: {output_path}")
        
        return results
    
    def print_result(self, result, detailed=True):
        """Print prediction result in a nice format"""
        video_name = result['video_name']
        inference_time = result['inference_time']
        predictions = result['predictions']
        
        print(f"\n🎬 Video: {video_name}")
        print(f"⏱️  Inference time: {inference_time:.3f} seconds")
        print(f"🎯 Predictions:")
        
        if detailed:
            # Show all top-k predictions
            for pred in predictions:
                activity = pred['activity']
                confidence = pred['confidence']
                rank = pred['rank']
                
                # Add emoji based on confidence
                if confidence > 70:
                    emoji = "🟢"
                elif confidence > 50:
                    emoji = "🟡"
                elif confidence > 30:
                    emoji = "🟠"
                else:
                    emoji = "🔴"
                
                print(f"  {rank}. {emoji} {activity}: {confidence:.1f}%")
        else:
            # Show only top prediction
            top_pred = predictions[0] if predictions else None
            if top_pred:
                activity = top_pred['activity']
                confidence = top_pred['confidence']
                print(f"  🏆 {activity} ({confidence:.1f}%)")

def setup_webcam_capture(predictor):
    """Setup webcam capture for real-time inference"""
    print(f"📹 Setting up webcam capture...")
    print(f"Press 'q' to quit, 'space' to capture and predict")
    
    cap = cv2.VideoCapture(0)  # Default webcam
    
    if not cap.isOpened():
        print(f"❌ Could not open webcam")
        return
    
    # Set webcam properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    frames_buffer = []
    recording = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"❌ Failed to capture frame")
            break
        
        # Display frame
        display_frame = frame.copy()
        
        # Add status text
        if recording:
            cv2.putText(display_frame, f"Recording... {len(frames_buffer)}/16", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(display_frame, "Press SPACE to start recording", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Phase 3 Webcam Inference', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):  # Quit
            break
        elif key == ord(' '):  # Space - start/stop recording
            if not recording:
                frames_buffer = []
                recording = True
                print(f"🔴 Recording started...")
            else:
                recording = False
                print(f"⏹️  Recording stopped, processing...")
                
                if len(frames_buffer) >= 8:  # Minimum frames
                    try:
                        # Process captured frames
                        frames_array = np.array(frames_buffer[:16])  # Take first 16 frames
                        
                        # Convert BGR to RGB
                        frames_rgb = []
                        for frame in frames_array:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frames_rgb.append(frame_rgb)
                        
                        frames_array = np.array(frames_rgb)
                        
                        # Apply transforms
                        frames_tensor = predictor.transform(frames_array)
                        frames_tensor = frames_tensor.unsqueeze(0).to(predictor.device)
                        
                        # Predict
                        with torch.no_grad():
                            outputs = predictor.model(frames_tensor)
                            probabilities = torch.softmax(outputs, dim=1)
                            top_prob, top_idx = torch.max(probabilities, dim=1)
                            
                            activity = predictor.activities[top_idx.item()]
                            confidence = top_prob.item() * 100
                            
                            print(f"🎯 Prediction: {activity} ({confidence:.1f}%)")
                    
                    except Exception as e:
                        print(f"❌ Error during prediction: {e}")
                else:
                    print(f"❌ Not enough frames captured: {len(frames_buffer)}")
        
        # Collect frames when recording
        if recording and len(frames_buffer) < 16:
            frames_buffer.append(frame.copy())
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Phase 3 Daily Activities Inference')
    parser.add_argument('--video', type=str, help='Path to video file or folder')
    parser.add_argument('--model', type=str, 
                       default=r'C:\ASH_PROJECT\outputs\phase3_checkpoints\phase3_best.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--batch', action='store_true', 
                       help='Process all videos in folder')
    parser.add_argument('--webcam', action='store_true', 
                       help='Use webcam for real-time inference')
    parser.add_argument('--output', type=str, 
                       help='Output file for batch results (JSON)')
    parser.add_argument('--top-k', type=int, default=5, 
                       help='Number of top predictions to show')
    parser.add_argument('--detailed', action='store_true', 
                       help='Show detailed predictions')
    
    args = parser.parse_args()
    
    # Print header
    print("🎯 Phase 3: Daily Activities Inference")
    print("=" * 50)
    
    # Check model path
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print(f"💡 Make sure you've trained Phase 3 model first!")
        return
    
    # Initialize predictor
    try:
        predictor = Phase3Predictor(model_path)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Run inference based on mode
    if args.webcam:
        # Webcam mode
        setup_webcam_capture(predictor)
        
    elif args.video:
        video_path = Path(args.video)
        
        if not video_path.exists():
            print(f"❌ Video path not found: {video_path}")
            return
        
        if args.batch or video_path.is_dir():
            # Batch mode
            print(f"📁 Batch processing mode")
            results = predictor.predict_batch(video_path, args.output)
            
            # Summary
            if results:
                print(f"\n📊 Batch Summary:")
                print(f"  • Total videos: {len(results)}")
                print(f"  • Average confidence: {np.mean([r['top_prediction']['confidence'] for r in results]):.1f}%")
                
                # Activity distribution
                activities = [r['top_prediction']['activity'] for r in results]
                from collections import Counter
                activity_counts = Counter(activities)
                print(f"  • Most common: {activity_counts.most_common(3)}")
        
        else:
            # Single video mode
            print(f"🎬 Single video mode")
            try:
                result = predictor.predict(video_path, top_k=args.top_k)
                predictor.print_result(result, detailed=args.detailed)
                
            except Exception as e:
                print(f"❌ Error processing video: {e}")
    
    else:
        # Show help
        print(f"❌ Please specify --video, --webcam, or --batch")
        print(f"\n📖 Examples:")
        print(f"  Single video:  python phase3_inference.py --video path/to/video.mp4")
        print(f"  Batch folder:  python phase3_inference.py --video path/to/folder/ --batch")
        print(f"  Webcam:        python phase3_inference.py --webcam")
        
        # Show available activities
        activities = get_phase3_activities()
        print(f"\n🎯 Supported Activities ({len(activities)}):")
        for i, activity in enumerate(activities, 1):
            print(f"  {i:2d}. {activity}")

if __name__ == "__main__":
    main()