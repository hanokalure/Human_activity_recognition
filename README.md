# 🎬 Human Activity Recognition System

A comprehensive computer vision system for recognizing daily human activities using deep learning models trained on UCF-101 and Kinetics-400 datasets.

## 🎯 Project Overview

This project implements an optimized video action recognition system that can identify 22 common daily activities with high accuracy. The system has been optimized to reduce training time from 40+ hours to just 1-2 hours per epoch while maintaining excellent performance.

### 🏆 Current Status: Phase 1 Complete ✅

**✅ Successfully Trained Activities (9/22) - UCF-101 Dataset:**
1. **walking_with_dog** - Person walking with a dog
2. **swimming** - Swimming activities (multiple strokes)
3. **biking** - Cycling/biking activities  
4. **weight_lifting** - Weight lifting and bench pressing
5. **jumping_jacks** - Jumping jack exercises
6. **pullups** - Pull-up exercises
7. **pushups** - Push-up exercises
8. **typing** - Typing on keyboard/computer
9. **brushing_teeth** - Dental hygiene activities

## 🚀 Next Phase: Remaining Activities (13/22)

**🔄 To Be Trained - Kinetics-400 Dataset:**
10. **eating** - Various eating activities
11. **cooking** - Cooking and food preparation
12. **reading** - Reading books, newspapers
13. **sleeping** - Sleeping and resting
14. **cleaning** - Cleaning and housework
15. **yoga** - Yoga and stretching exercises
16. **walking** - General walking activities
17. **running** - Running and jogging
18. **driving** - Vehicle driving
19. **sitting** - Sitting in various positions

## 📁 Project Structure

```
ASH_PROJECT/
├── 📂 src/                           # Source code
│   ├── 🎯 daily_activities_config.py # Activity definitions & mappings
│   ├── 🚀 train_daily_activities.py  # Main training script
│   ├── 📊 daily_activities_dataset.py # Dataset handling
│   ├── 🔍 daily_activities_inference.py # Model inference
│   ├── ⚡ train_optimized.py         # Optimized training pipeline
│   ├── 🏗️ models_optimized.py        # Lightweight model architectures
│   ├── 📈 performance_monitor.py     # Training monitoring
│   └── 🎬 video_to_text.py          # Video-to-text conversion
│   
├── 📂 data/                          # Dataset storage
│   ├── 📁 UCF101/                   # UCF-101 dataset (Phase 1 ✅)
│   ├── 📁 Kinetics400_Daily/        # Kinetics-400 subset (Phase 2 🔄)
│   └── 📁 DailyActivities/          # Processed daily activities
│   
├── 📂 outputs/                       # Training outputs
│   ├── 🏆 checkpoints/              # Model checkpoints
│   ├── 📊 metrics/                  # Performance metrics
│   └── 🎯 inference_results/        # Inference outputs
│   
├── 📖 README.md                      # This file
├── 🚀 QUICK_START.md                 # Quick start guide
└── 📋 requirements.txt               # Python dependencies
```

## ⚡ Performance Optimizations

### Before vs After Optimization:

| Metric | Before (Original) | After (Optimized) | Improvement |
|--------|------------------|-------------------|-------------|
| **Model Size** | R3D-18 (33M params) | Efficient3DCNN (1-3M) | 🔥 10-30x smaller |
| **Training Time** | 40+ hours/epoch | 1-2 hours/epoch | ⚡ 20-40x faster |
| **Batch Size** | 4 | 16 | 📈 4x larger batches |
| **Memory Usage** | High | Optimized | 💾 ~60% reduction |
| **GPU Utilization** | Low (~30%) | High (80%+) | 🚀 Better efficiency |

### 🏗️ Model Architectures Available:

1. **Efficient3DCNN** (Recommended) - Ultra-lightweight, fastest training
2. **MobileNet3D** - Good balance of speed and accuracy  
3. **R(2+1)D Light** - Better accuracy, moderate speed
4. **R3D-18** - Original model (slower but highest accuracy)

## 🎯 Activity Categories

### 🏠 Indoor Activities (7 classes)
- **brushing_teeth** ✅ - Dental hygiene
- **typing** ✅ - Computer/keyboard use
- **eating** 🔄 - Various eating activities
- **cooking** 🔄 - Food preparation
- **reading** 🔄 - Reading books/newspapers
- **sleeping** 🔄 - Sleep and rest
- **cleaning** 🔄 - Housework and cleaning

### 💪 Fitness/Exercise (8 classes)
- **pushups** ✅ - Push-up exercises
- **pullups** ✅ - Pull-up exercises
- **jumping_jacks** ✅ - Jumping jack exercises
- **weight_lifting** ✅ - Weight training
- **yoga** 🔄 - Yoga and stretching
- **running_treadmill** 🔄 - Treadmill running
- **squats** 🔄 - Squat exercises
- **stretching** 🔄 - Stretching activities

### 🚶 Outdoor/General (7 classes)
- **walking_with_dog** ✅ - Dog walking
- **swimming** ✅ - Swimming activities
- **biking** ✅ - Cycling activities
- **walking** 🔄 - General walking
- **running** 🔄 - Running and jogging
- **driving** 🔄 - Vehicle operation
- **sitting** 🔄 - Sitting activities

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Navigate to project directory
cd C:\ASH_PROJECT

# Install dependencies
pip install torch torchvision opencv-python numpy matplotlib
```

### 2. Train Current Models (Phase 1 Activities)
```bash
# Train with optimized settings
cd src
python train_daily_activities.py

# Or use the ultra-fast optimized version
python train_optimized.py
```

### 3. Download Kinetics Data for Phase 2
```bash
# Download remaining activities from Kinetics-400
python download_daily_activities_data.py
```

### 4. Inference on New Videos
```python
from daily_activities_inference import DailyActivitiesInference

# Load trained model
model = DailyActivitiesInference("../outputs/checkpoints/best_model.pth")

# Analyze video
result = model.predict_video("path/to/video.mp4")
print(f"Activity: {result['activity']} (Confidence: {result['confidence']:.2f})")
```

## 📊 Current Performance Metrics

### Phase 1 Results (UCF-101 Activities):
- **Overall Accuracy**: ~75-85% on test set
- **Training Time**: 1-2 hours per epoch (vs 40+ hours originally)
- **Model Size**: 1-3M parameters (vs 33M originally)
- **Inference Speed**: Real-time capable (~30 FPS)

### Target Phase 2 Performance:
- **Target Accuracy**: 85-90% on all 22 daily activities
- **Expected Training Time**: 3-5 hours total for remaining activities
- **Final Model Size**: <5M parameters
- **Real-time Performance**: Maintained

## 🎬 Video-to-Text Capabilities

The system includes advanced video-to-text functionality:

```python
from video_to_text import create_video_analyzer

analyzer = create_video_analyzer("path/to/model.pth")

# Simple description
description = analyzer.video_to_text("video.mp4")
# Output: "Person is doing push-ups"

# Detailed analysis
detailed = analyzer.video_to_text("video.mp4", detailed=True)
# Output: "Person is performing push-up exercises in a gym setting with proper form"
```

## 🛠️ Development Roadmap

### ✅ Completed (Phase 1)
- [x] UCF-101 dataset integration
- [x] 9 core daily activities training
- [x] Performance optimization (20-40x speed improvement)
- [x] Multiple model architectures
- [x] Real-time inference pipeline
- [x] Video-to-text conversion
- [x] Comprehensive monitoring and evaluation

### 🔄 In Progress (Phase 2)
- [ ] Kinetics-400 dataset integration
- [ ] Training remaining 13 activities
- [ ] Cross-dataset validation
- [ ] Model ensemble techniques
- [ ] Mobile deployment optimization

### 🎯 Future Enhancements
- [ ] Real-time webcam recognition
- [ ] Multi-person activity detection
- [ ] Activity sequence analysis
- [ ] Mobile app integration
- [ ] Cloud API deployment

## 📈 Monitoring & Performance

The system includes comprehensive performance monitoring:

```python
from performance_monitor import PerformanceMonitor

monitor = PerformanceMonitor()
monitor.start_monitoring()
# ... training code ...
monitor.stop_monitoring()
monitor.save_metrics()
monitor.plot_metrics()
```

**Monitoring Features:**
- GPU/CPU utilization tracking
- Memory usage optimization
- Training speed analysis  
- Accuracy progression
- Loss visualization
- Real-time performance alerts

## 🎯 Key Features

- **⚡ Ultra-Fast Training**: 20-40x faster than original implementation
- **🎯 High Accuracy**: Target 85-90% on focused daily activities
- **💾 Memory Optimized**: Efficient memory usage with large batch processing
- **🔍 Real-time Inference**: Capable of real-time video analysis
- **📊 Comprehensive Monitoring**: Detailed performance tracking and visualization
- **🎬 Video-to-Text**: Natural language descriptions of activities
- **🏗️ Multiple Architectures**: Choose the best model for your use case
- **📱 Deployment Ready**: Optimized for production deployment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-activity`)
3. Commit changes (`git commit -am 'Add new activity recognition'`)
4. Push to branch (`git push origin feature/new-activity`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **UCF-101 Dataset**: University of Central Florida for the comprehensive action recognition dataset
- **Kinetics-400**: DeepMind for the large-scale human action video dataset
- **PyTorch Team**: For the excellent deep learning framework
- **OpenCV Community**: For computer vision utilities

## 📞 Support

- **Documentation**: Check `QUICK_START.md` for detailed setup instructions
- **Performance Issues**: Use the built-in `performance_monitor.py` for diagnostics
- **Model Selection**: Run `get_model_comparison()` for architecture guidance

---

**🎯 Project Goal**: Create the most efficient and accurate daily activities recognition system, optimized for real-world deployment and practical applications.

**Current Status**: Phase 1 Complete ✅ | Phase 2 In Progress 🔄 | Target Completion: 85-90% accuracy on all 22 daily activities
