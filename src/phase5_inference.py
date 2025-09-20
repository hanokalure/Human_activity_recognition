"""
Phase 5 Inference - 25 Daily Life Activities
============================================

Inference script for Phase 5 model trained on 25 daily activities.
Supports single videos, batch processing, and webcam real-time input.

Usage:
    python phase5_inference.py --video path/to/video.mp4
    python phase5_inference.py --webcam
    python phase5_inference.py --batch_dir path/to/videos/ --output results.json
"""

import argparse
import json
import logging
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class R2Plus1DModel(nn.Module):
    """R(2+1)D wrapper for Phase 5 (25 classes)"""

    def __init__(self, num_classes=25, dropout_rate=0.6):
        super(R2Plus1DModel, self).__init__()
        from torchvision.models.video import r2plus1d_18
        # Use the same backbone as training (pretrained=True was used during training)
        self.backbone = r2plus1d_18(pretrained=False)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


class VideoProcessor:
    """Video processing utilities (file + frame sequences)

    Matches preprocessing used in Phase5 training: 16 frames, 112x112, ImageNet normalization.
    """

    def __init__(self, frames_per_clip=16, frame_size=(112, 112)):
        self.frames_per_clip = frames_per_clip
        self.frame_size = frame_size
        # ImageNet mean/std
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)

    def process_video_file(self, video_path):
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                return None

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                # fallback to reading sequentially
                frames = []
                while len(frames) < self.frames_per_clip:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)
                cap.release()
                if len(frames) == 0:
                    logger.error(f"No frames read from {video_path}")
                    return None
                # pad/repeat if needed
                while len(frames) < self.frames_per_clip:
                    frames.append(frames[-1].copy())
            else:
                # uniform sampling over available frames
                indices = np.linspace(0, total_frames - 1, self.frames_per_clip, dtype=int)
                frames = []
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                    ret, frame = cap.read()
                    if not ret:
                        # use last frame or zeros
                        if frames:
                            frames.append(frames[-1].copy())
                        else:
                            frames.append(np.zeros((self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8))
                    else:
                        frames.append(frame)
                cap.release()

            # Preprocess frames
            frame_tensors = []
            for f in frames[: self.frames_per_clip]:
                f = cv2.resize(f, self.frame_size)
                f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                f = f.astype(np.float32) / 255.0
                frame_tensors.append(torch.from_numpy(f))

            # Stack and permute to (C, T, H, W)
            video_tensor = torch.stack(frame_tensors)  # (T, H, W, C)
            video_tensor = video_tensor.permute(3, 0, 1, 2)  # (C, T, H, W)
            video_tensor = (video_tensor - self.mean) / self.std
            return video_tensor.unsqueeze(0)  # (1, C, T, H, W)

        except Exception as e:
            logger.error(f"Error processing video {video_path}: {e}")
            return None

    def process_frame_sequence(self, frames):
        try:
            if len(frames) != self.frames_per_clip:
                logger.warning(f"Expected {self.frames_per_clip} frames, got {len(frames)}")
                return None

            frame_tensors = []
            for f in frames:
                f = cv2.resize(f, self.frame_size)
                f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                f = f.astype(np.float32) / 255.0
                frame_tensors.append(torch.from_numpy(f))

            video_tensor = torch.stack(frame_tensors)
            video_tensor = video_tensor.permute(3, 0, 1, 2)
            video_tensor = (video_tensor - self.mean) / self.std
            return video_tensor.unsqueeze(0)

        except Exception as e:
            logger.error(f"Error processing frame sequence: {e}")
            return None


class Phase5Predictor:
    """Predictor that loads a Phase 5 checkpoint and runs inference"""

    def __init__(self, model_path, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_path = Path(model_path)
        logger.info(f"Loading Phase 5 model from: {model_path}")

        checkpoint = torch.load(str(model_path), map_location=self.device)

        # Read class mapping robustly (handle different key names)
        if 'class_to_idx' in checkpoint:
            self.class_to_idx = checkpoint['class_to_idx']
        elif 'class_idx' in checkpoint:
            self.class_to_idx = checkpoint['class_idx']
        else:
            # Try to build from class_names
            if 'class_names' in checkpoint:
                class_names = checkpoint['class_names']
                self.class_to_idx = {c: i for i, c in enumerate(class_names)}
            else:
                raise KeyError('class_to_idx or class_names not found in checkpoint')

        # idx_to_class
        self.idx_to_class = {int(v): k for k, v in self.class_to_idx.items()}

        # num_classes if provided, else infer
        self.num_classes = int(checkpoint.get('num_classes', len(self.idx_to_class)))

        # Init model and load weights
        self.model = R2Plus1DModel(num_classes=self.num_classes)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        # If checkpoint directly contains state dict keys, allow that
        if 'state_dict' in state_dict and isinstance(state_dict, dict):
            state_dict = state_dict['state_dict']

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.processor = VideoProcessor()

        logger.info(f"Model loaded: {self.num_classes} classes")
        logger.info(f"Device: {self.device}")
        if 'best_val_acc' in checkpoint:
            logger.info(f"Checkpoint best validation accuracy: {checkpoint['best_val_acc']}")

    def predict_video(self, video_path, return_probabilities=False):
        video_tensor = self.processor.process_video_file(video_path)
        if video_tensor is None:
            return None

        with torch.no_grad():
            video_tensor = video_tensor.to(self.device)
            outputs = self.model(video_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_idx = int(torch.argmax(probs, dim=1).item())
            pred_cls = self.idx_to_class.get(pred_idx, str(pred_idx))
            conf = float(probs[0, pred_idx].item())

            result = {
                'video_path': str(video_path),
                'predicted_class': pred_cls,
                'confidence': conf,
            }
            if return_probabilities:
                result['all_probabilities'] = {self.idx_to_class[i]: float(probs[0, i].item()) for i in range(probs.shape[1])}
            return result

    def predict_frames(self, frames, return_probabilities=True):
        video_tensor = self.processor.process_frame_sequence(frames)
        if video_tensor is None:
            return None

        with torch.no_grad():
            video_tensor = video_tensor.to(self.device)
            outputs = self.model(video_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_idx = int(torch.argmax(probs, dim=1).item())
            pred_cls = self.idx_to_class.get(pred_idx, str(pred_idx))
            conf = float(probs[0, pred_idx].item())

            result = {
                'predicted_class': pred_cls,
                'confidence': conf,
            }
            if return_probabilities:
                result['all_probabilities'] = {self.idx_to_class[i]: float(probs[0, i].item()) for i in range(probs.shape[1])}
            return result

    def predict_batch(self, video_paths, output_file=None):
        results = []
        logger.info(f"Running batch prediction on {len(video_paths)} videos")
        for p in video_paths:
            res = self.predict_video(p, return_probabilities=True)
            if res:
                results.append(res)
                logger.info(f"{p} -> {res['predicted_class']} ({res['confidence']:.2f})")
            else:
                logger.warning(f"Failed: {p}")

        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved results to {output_file}")
        return results


def run_webcam_inference(model_path, fps_target=10):
    predictor = Phase5Predictor(model_path)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error('Failed to open webcam')
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, fps_target)

    frame_buffer = deque(maxlen=16)
    frame_count = 0
    prediction_interval = max(1, int(fps_target))

    current_prediction = 'Analyzing...'
    current_confidence = 0.0
    paused = False

    logger.info("Webcam running - press 'q' to quit, 'p' to pause/resume")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error('Failed to read webcam frame')
                break

            if not paused:
                frame_buffer.append(frame.copy())
                frame_count += 1

                if frame_count % prediction_interval == 0 and len(frame_buffer) == 16:
                    res = predictor.predict_frames(list(frame_buffer))
                    if res:
                        current_prediction = res['predicted_class']
                        current_confidence = res['confidence']

            disp = frame.copy()
            text = f"{current_prediction}: {current_confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.rectangle(disp, (10, 10), (350, 55), (0, 0, 0), -1)
            cv2.putText(disp, text, (15, 40), font, 0.8, (0, 255, 0), 2)

            status = 'PAUSED' if paused else 'ACTIVE'
            cv2.putText(disp, status, (15, 80), font, 0.6, (0, 255, 255) if not paused else (0, 0, 255), 1)

            cv2.imshow('Phase5 Activity Recognition', disp)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                logger.info('Paused' if paused else 'Resumed')

    except KeyboardInterrupt:
        logger.info('Interrupted by user')
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info('Webcam stopped')


def main():
    parser = argparse.ArgumentParser(description='Phase 5 Inference - 25 Activities')
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--video', type=str, help='Path to single video file')
    input_group.add_argument('--webcam', action='store_true', help='Use webcam')
    input_group.add_argument('--batch_dir', type=str, help='Directory with videos for batch processing')

    # Default model path already set
    parser.add_argument('--model', type=str, default='C:/ASH_PROJECT/models/phase5/phase5_best_model.pth', help='Path to Phase5 model checkpoint')
    parser.add_argument('--output', type=str, help='Output JSON file for batch results')
    parser.add_argument('--verbose', action='store_true', help='Print full probability vector for single video')

    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return

    try:
        if args.webcam:
            run_webcam_inference(args.model)
            return

        predictor = Phase5Predictor(args.model)

        if args.video:
            res = predictor.predict_video(args.video, return_probabilities=args.verbose)
            if res:
                print(f"\n🎯 Prediction for: {res['video_path']}")
                print(f"   Activity: {res['predicted_class']}")
                print(f"   Confidence: {res['confidence']:.4f}")
                if args.verbose and 'all_probabilities' in res:
                    print('\n📊 All predictions:')
                    sorted_probs = sorted(res['all_probabilities'].items(), key=lambda x: x[1], reverse=True)
                    for i, (cls, p) in enumerate(sorted_probs, 1):
                        print(f"   {i:2d}. {cls}: {p:.4f}")
            else:
                logger.error('Failed to process video')

        elif args.batch_dir:
            batch_dir = Path(args.batch_dir)
            if not batch_dir.exists():
                logger.error(f"Batch directory not found: {batch_dir}")
                return

            exts = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}
            video_files = [str(f) for f in batch_dir.rglob('*') if f.suffix.lower() in exts]
            if not video_files:
                logger.error('No video files found for batch processing')
                return

            results = predictor.predict_batch(video_files, args.output)

            print(f"\n📊 Batch Processing Results: {len(results)} processed")
            if results:
                counts = {}
                for r in results:
                    counts[r['predicted_class']] = counts.get(r['predicted_class'], 0) + 1
                print('\n📈 Activity distribution:')
                for act, c in sorted(counts.items()):
                    print(f"   {act}: {c}")

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise


if __name__ == '__main__':
    main()
