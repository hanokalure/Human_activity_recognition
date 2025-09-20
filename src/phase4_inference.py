"""
Phase 4 Inference - Separate UCF-101 + HMDB-51 Model
===================================================

Inference script for the Phase 4 model trained on 17 separate daily activities.
Supports single videos, batch processing, and real-time webcam input.

Activities (17 Total):
- UCF-101 (10): brushing_teeth, typing, biking_ucf, pullups_ucf, pushups_ucf,
                writing, walking_dog, cooking_ucf, breast_stroke, front_crawl
- HMDB-51 (7): walking, running, sitting, eating, brushing_hair, talking, pouring

Usage:
    python phase4_inference.py --video path/to/video.mp4
    python phase4_inference.py --webcam
    python phase4_inference.py --batch_dir path/to/videos/
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
import argparse
from pathlib import Path
import json
import time
from collections import deque
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class R2Plus1DModel(nn.Module):
    """R(2+1)D model for video classification"""
    
    def __init__(self, num_classes=17, dropout_rate=0.5):
        super(R2Plus1DModel, self).__init__()
        
        # Load pretrained R(2+1)D model
        from torchvision.models.video import r2plus1d_18
        self.backbone = r2plus1d_18(pretrained=False)  # Don't load pretrained weights
        
        # Replace the classifier
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class VideoProcessor:
    """Video processing utilities for inference"""
    
    def __init__(self, frames_per_clip=16, frame_size=(112, 112)):
        self.frames_per_clip = frames_per_clip
        self.frame_size = frame_size
        
        # Normalization parameters (ImageNet)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
    
    def process_video_file(self, video_path):
        """Process a video file and extract frames"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                return None
            
            frames = []
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            if total_frames < self.frames_per_clip:
                # If video is too short, sample with repetition
                frame_indices = np.linspace(0, total_frames - 1, self.frames_per_clip, dtype=int)
            else:
                # Sample frames uniformly
                frame_indices = np.linspace(0, total_frames - 1, self.frames_per_clip, dtype=int)
            
            # Extract frames
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # Preprocess frame
                    frame = cv2.resize(frame, self.frame_size)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = frame.astype(np.float32) / 255.0
                    frames.append(torch.from_numpy(frame))
                else:
                    # Use last good frame or zeros
                    if frames:
                        frames.append(frames[-1].clone())
                    else:
                        frames.append(torch.zeros(*self.frame_size, 3))
            
            cap.release()
            
            # Convert to tensor format (C, T, H, W)
            if len(frames) == self.frames_per_clip:
                video_tensor = torch.stack(frames)  # (T, H, W, C)
                video_tensor = video_tensor.permute(3, 0, 1, 2)  # (C, T, H, W)
                
                # Normalize
                video_tensor = (video_tensor - self.mean) / self.std
                
                return video_tensor.unsqueeze(0)  # Add batch dimension
            else:
                logger.warning(f"Could not extract {self.frames_per_clip} frames from video")
                return None
                
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {e}")
            return None
    
    def process_frame_sequence(self, frames):
        """Process a sequence of frames (for webcam input)"""
        try:
            if len(frames) != self.frames_per_clip:
                logger.warning(f"Expected {self.frames_per_clip} frames, got {len(frames)}")
                return None
            
            # Convert frames to tensors
            frame_tensors = []
            for frame in frames:
                # Preprocess frame
                frame = cv2.resize(frame, self.frame_size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32) / 255.0
                frame_tensors.append(torch.from_numpy(frame))
            
            # Convert to tensor format (C, T, H, W)
            video_tensor = torch.stack(frame_tensors)  # (T, H, W, C)
            video_tensor = video_tensor.permute(3, 0, 1, 2)  # (C, T, H, W)
            
            # Normalize
            video_tensor = (video_tensor - self.mean) / self.std
            
            return video_tensor.unsqueeze(0)  # Add batch dimension
            
        except Exception as e:
            logger.error(f"Error processing frame sequence: {e}")
            return None

class Phase4Predictor:
    """Phase 4 model predictor"""
    
    def __init__(self, model_path, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        logger.info(f"Loading Phase 4 model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract model info
        self.num_classes = checkpoint['num_classes']
        self.class_names = checkpoint['class_names']
        self.class_to_idx = checkpoint['class_to_idx']
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        # Initialize model
        self.model = R2Plus1DModel(num_classes=self.num_classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize video processor
        self.processor = VideoProcessor()
        
        logger.info(f"Model loaded successfully")
        logger.info(f"  Classes: {self.num_classes}")
        logger.info(f"  Activities: {', '.join(self.class_names)}")
        logger.info(f"  Device: {self.device}")
        
        if 'best_val_acc' in checkpoint:
            logger.info(f"  Model accuracy: {checkpoint['best_val_acc']:.2f}%")
    
    def predict_video(self, video_path, return_probabilities=False):
        """Predict activity for a single video file"""
        
        # Process video
        video_tensor = self.processor.process_video_file(video_path)
        if video_tensor is None:
            return None
        
        # Make prediction
        with torch.no_grad():
            video_tensor = video_tensor.to(self.device)
            outputs = self.model(video_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get prediction
            predicted_idx = torch.argmax(probabilities, dim=1).item()
            predicted_class = self.idx_to_class[predicted_idx]
            confidence = probabilities[0, predicted_idx].item()
            
            result = {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'video_path': str(video_path)
            }
            
            if return_probabilities:
                result['all_probabilities'] = {
                    self.idx_to_class[i]: prob.item() 
                    for i, prob in enumerate(probabilities[0])
                }
            
            return result
    
    def predict_frames(self, frames):
        """Predict activity for a sequence of frames"""
        
        # Process frames
        video_tensor = self.processor.process_frame_sequence(frames)
        if video_tensor is None:
            return None
        
        # Make prediction
        with torch.no_grad():
            video_tensor = video_tensor.to(self.device)
            outputs = self.model(video_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get prediction
            predicted_idx = torch.argmax(probabilities, dim=1).item()
            predicted_class = self.idx_to_class[predicted_idx]
            confidence = probabilities[0, predicted_idx].item()
            
            return {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'all_probabilities': {
                    self.idx_to_class[i]: prob.item() 
                    for i, prob in enumerate(probabilities[0])
                }
            }
    
    def predict_batch(self, video_paths, output_file=None):
        """Predict activities for multiple videos"""
        
        results = []
        logger.info(f"Processing {len(video_paths)} videos...")
        
        for i, video_path in enumerate(video_paths):
            logger.info(f"Processing video {i+1}/{len(video_paths)}: {video_path}")
            
            result = self.predict_video(video_path, return_probabilities=True)
            if result:
                results.append(result)
                logger.info(f"  Predicted: {result['predicted_class']} ({result['confidence']:.2f})")
            else:
                logger.warning(f"  Failed to process video: {video_path}")
        
        # Save results if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_file}")
        
        return results

def run_webcam_inference(model_path, fps_target=10):
    """Run real-time inference on webcam feed"""
    
    predictor = Phase4Predictor(model_path)
    logger.info(f"Starting webcam inference at {fps_target} FPS")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Failed to open webcam")
        return
    
    # Set webcam properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, fps_target)
    
    # Frame buffer for temporal analysis
    frame_buffer = deque(maxlen=16)
    prediction_interval = fps_target  # Predict every second
    frame_count = 0
    
    current_prediction = "Analyzing..."
    current_confidence = 0.0
    
    logger.info("Webcam started. Press 'q' to quit, 'p' to pause/resume")
    paused = False
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read from webcam")
                break
            
            if not paused:
                frame_buffer.append(frame.copy())
                frame_count += 1
                
                # Make prediction every interval
                if frame_count % prediction_interval == 0 and len(frame_buffer) == 16:
                    frames = list(frame_buffer)
                    result = predictor.predict_frames(frames)
                    
                    if result:
                        current_prediction = result['predicted_class']
                        current_confidence = result['confidence']
                        
                        # Log top 3 predictions
                        sorted_probs = sorted(
                            result['all_probabilities'].items(),
                            key=lambda x: x[1], reverse=True
                        )
                        top_3 = sorted_probs[:3]
                        logger.info(f"Top predictions: {', '.join([f'{cls}({prob:.2f})' for cls, prob in top_3])}")
            
            # Display frame with prediction
            display_frame = frame.copy()
            
            # Add text overlay
            text = f"{current_prediction}: {current_confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            
            # Get text size for background
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            
            # Draw background rectangle
            cv2.rectangle(display_frame, (10, 10), 
                         (text_size[0] + 20, text_size[1] + 20), 
                         (0, 0, 0), -1)
            
            # Draw text
            cv2.putText(display_frame, text, (15, text_size[1] + 15),
                       font, font_scale, (0, 255, 0), thickness)
            
            # Add status
            status = "PAUSED" if paused else "ACTIVE"
            cv2.putText(display_frame, status, (15, 50),
                       font, 0.5, (0, 255, 255) if not paused else (0, 0, 255), 1)
            
            # Show frame
            cv2.imshow('Phase 4 Activity Recognition', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                logger.info(f"Webcam {'paused' if paused else 'resumed'}")
    
    except KeyboardInterrupt:
        logger.info("Webcam inference interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Webcam inference stopped")

def main():
    parser = argparse.ArgumentParser(description='Phase 4 Inference')
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--video', type=str, help='Path to video file')
    input_group.add_argument('--webcam', action='store_true', help='Use webcam input')
    input_group.add_argument('--batch_dir', type=str, help='Directory containing video files')
    
    # Model options
    parser.add_argument('--model', type=str, 
                       default='C:/ASH_PROJECT/models/phase4/phase4_best_model.pth',
                       help='Path to Phase 4 model file')
    
    # Output options
    parser.add_argument('--output', type=str, help='Output file for batch results (JSON)')
    parser.add_argument('--verbose', action='store_true', help='Show detailed predictions')
    
    args = parser.parse_args()
    
    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        logger.error("Please ensure Phase 4 model is trained and saved")
        return
    
    try:
        if args.webcam:
            # Webcam inference
            run_webcam_inference(args.model)
            
        elif args.video:
            # Single video inference
            predictor = Phase4Predictor(args.model)
            result = predictor.predict_video(args.video, return_probabilities=args.verbose)
            
            if result:
                print(f"\n🎯 Prediction for: {result['video_path']}")
                print(f"   Activity: {result['predicted_class']}")
                print(f"   Confidence: {result['confidence']:.2f}")
                
                if args.verbose and 'all_probabilities' in result:
                    print("\n📊 All predictions:")
                    sorted_probs = sorted(
                        result['all_probabilities'].items(),
                        key=lambda x: x[1], reverse=True
                    )
                    for i, (activity, prob) in enumerate(sorted_probs, 1):
                        print(f"   {i:2d}. {activity}: {prob:.3f}")
            else:
                logger.error(f"Failed to process video: {args.video}")
        
        elif args.batch_dir:
            # Batch processing
            batch_dir = Path(args.batch_dir)
            if not batch_dir.exists():
                logger.error(f"Batch directory not found: {batch_dir}")
                return
            
            # Find video files
            video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}
            video_files = [
                f for f in batch_dir.rglob('*') 
                if f.suffix.lower() in video_extensions
            ]
            
            if not video_files:
                logger.error(f"No video files found in {batch_dir}")
                return
            
            predictor = Phase4Predictor(args.model)
            results = predictor.predict_batch(video_files, args.output)
            
            print(f"\n📊 Batch Processing Results:")
            print(f"   Total videos: {len(video_files)}")
            print(f"   Successfully processed: {len(results)}")
            
            if results:
                # Show summary
                activity_counts = {}
                for result in results:
                    activity = result['predicted_class']
                    activity_counts[activity] = activity_counts.get(activity, 0) + 1
                
                print("\n📈 Activity Distribution:")
                for activity, count in sorted(activity_counts.items()):
                    print(f"   {activity}: {count} videos")
    
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise

if __name__ == "__main__":
    main()