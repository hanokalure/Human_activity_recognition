# Human Activity Recognition - Phase 5 Project Details

## 🎯 Project Overview

This project is an **human activity recognition system** that can automatically identify and classify human activities from video footage. Using advanced deep learning techniques, our system can recognize **25 different daily life activities** with **87.34% accuracy**.

### What does it do?
- **Analyzes video files** (MP4 format) to identify human activities
- **Recognizes 25 activities** including walking, running, cooking, typing, yoga, and more
- **Provides confidence scores** for each prediction
- **Works in real-time** through webcam(upcoming) or uploaded videos
- **Accessible via web interface** and REST API

### Why did we build this?
Human activity recognition has numerous real-world applications:
- **Healthcare monitoring** - Track patient activities and mobility
- **Smart home systems** - Automate responses based on user activities
- **Security surveillance** - Detect suspicious or emergency activities
- **Fitness tracking** - Monitor exercise routines and daily movement
- **Elderly care** - Ensure safety and detect falls or unusual behavior

## 🧠 The Technology Behind It

### Model Architecture
We use a **R(2+1)D Convolutional Neural Network**, which is specifically designed for video analysis:
- **R(2+1)D model**: Separates spatial and temporal convolutions for better video understanding
- **Input**: 16 frames at 112x112 resolution per video clip
- **Output**: Classification scores for 25 different activities
- **Backbone**: PyTorch's pre-trained r2plus1d_18 model
- **Custom classifier**: Multi-layer neural network with dropout for regularization

### Training Details
- **Dataset**: Combined UCF-101 and HMDB-51 video datasets
- **Total videos**: Thousands of labeled video clips across 25 activity categories
- **Training duration**: 60 epochs
- **Hardware used**: NVIDIA GeForce RTX 3050 4GB
- **Final accuracy**: 87.34% on validation set
- **Framework**: PyTorch 2.0+ with torchvision

## 🏗️ System Architecture

### Frontend Application
- **Technology**: React with Expo (for cross-platform support)
- **Features**: 
  - Video upload interface
  - Real-time prediction display
  - Activity confidence visualization
  - Responsive design for mobile and desktop
- **Deployment**: Vercel hosting
- **Live URL**: https://human-activity-recognition6.vercel.app/

### Backend API
- **Technology**: FastAPI (Python)
- **Features**:
  - REST API endpoints for video processing
  - WebSocket support for real-time streaming
  - Automatic model downloading
  - CORS support for cross-origin requests
- **Deployment**: Hugging Face Spaces (Docker)
- **Live URL**: https://hanokalure-human-activity-backend.hf.space

### Model Serving
- **Model file**: phase5_best_model.pth (~87MB)
- **Storage**: Google Drive with automatic download
- **Inference**: CPU/GPU compatible with PyTorch
- **Processing**: OpenCV for video handling

## 📋 Recognized Activities (25 Total)

### Daily Life Activities
- **Personal care**: brushing_teeth, eating, drinking, brushing_hair
- **Communication**: talking, waving, hugging, laughing
- **Household**: cooking, cleaning, pouring

### Physical Activities & Sports
- **Exercise**: pushups, pullups, weight_lifting, yoga
- **Swimming**: breast_stroke, front_crawl
- **Movement**: walking, running, biking, climbing_stairs

### Work & Leisure
- **Office work**: typing, writing
- **Outdoor**: walking_dog
- **Basic postures**: sitting

## 🚀 How to Use the System

### Option 1: Web Interface (Easiest)
1. Visit: https://human-activity-recognition6.vercel.app/
2. Upload an MP4 video file
3. Wait for processing (usually 5-10 seconds)
4. View the predicted activity and confidence score

### Option 2: Direct API Access
```bash
# Upload video to API
curl -X POST "https://hanokalure-human-activity-backend.hf.space/predict" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_video.mp4"
```

## 🎥 Video Requirements

### Supported Formats
- **Format**: MP4 (recommended)
- **Duration**: 2-30 seconds for best results
- **Resolution**: Any (auto-resized to 112x112 for processing)
- **Content**: Clear view of human performing single activity

### Best Practices for Accuracy
- **Good lighting** - Avoid dark or poorly lit videos
- **Clear subject** - Person should be clearly visible
- **Single activity** - Focus on one main activity per video
- **Stable footage** - Minimize camera shake
- **Full body view** - Include the person's full body when possible

## 🔧 Technical Implementation

### Backend Structure
```
backend/
├── main.py              # FastAPI application with endpoints
├── requirements.txt     # Python dependencies
├── start_server.py      # Local development server
└── download_model.py    # Model downloading utility
```

### Key Components
```
src/
├── phase5_inference.py       # Main inference engine
├── models/phase5/           # Model architecture definitions
├── utils/activity_labels.py # Activity name mappings
└── preprocessing/           # Video preprocessing utilities
```

### API Endpoints
- **GET /health** - Check system status
- **POST /predict** - Upload video for prediction
- **POST /predict/video** - Alternative upload endpoint
- **WebSocket /ws/frames** - Real-time frame streaming

## 📊 Performance Metrics

### Model Performance
- **Validation Accuracy**: 87.34%
- **Total Classes**: 25 activities
- **Model Size**: ~87MB
- **Inference Time**: ~2-5 seconds per video
- **Memory Usage**: ~500MB during inference

### System Performance
- **API Response Time**: 3-8 seconds average
- **Concurrent Users**: Supports multiple simultaneous requests
- **Uptime**: 99%+ on Hugging Face Spaces
- **Error Rate**: <2% for properly formatted videos

## 🛠️ Development Process

### Why This Approach?
1. **R(2+1)D Architecture**: Better than traditional CNN-LSTM for video analysis
2. **Transfer Learning**: Started with pre-trained weights for faster training
3. **Cloud Deployment**: Accessible anywhere without local setup
4. **REST API Design**: Easy integration with any frontend technology
5. **Docker Containerization**: Consistent deployment across platforms

### Challenges Solved
- **Memory constraints** with 4GB GPU - Optimized batch sizes and model architecture
- **Video preprocessing** - Standardized frame extraction and resizing
- **Model deployment** - Automated model downloading in production
- **Cross-origin requests** - Proper CORS configuration for web access

## 🔮 Future Enhancements

### Short-term Goals
- **Live camera feed**: Real-time activity recognition from webcam
- **Mobile app**: Native iOS/Android applications
- **Batch processing**: Multiple video analysis at once
- **Activity logging**: Track activities over time

### Long-term Vision
- **More activities**: Expand to 50+ recognizable activities
- **Multi-person detection**: Handle multiple people in one video
- **Temporal analysis**: Understand activity sequences and patterns
- **Edge deployment**: Run on mobile devices and IoT hardware

## 📈 Demo and Results

### Live Demo
- **Video demonstration**: [View Project Demo](https://drive.google.com/file/d/1CRon3Fq8LqLELR79MVuSonhu5dnxvvWx/view?usp=drive_link)
- **Web interface**: https://human-activity-recognition6.vercel.app/
- **API testing**: https://hanokalure-human-activity-backend.hf.space/docs

### Sample Predictions
The system successfully recognizes activities like:
- Walking with 95% confidence
- Cooking with 89% confidence  
- Typing with 92% confidence
- Yoga poses with 84% confidence

## 🛡️ Technical Specifications

### Dependencies
```
Core Framework:
- PyTorch 2.0+ (Deep learning)
- FastAPI (Web framework)
- OpenCV (Video processing)
- NumPy (Numerical computing)

Deployment:
- Docker (Containerization)
- Hugging Face Spaces (Backend hosting)
- Vercel (Frontend hosting)
- Google Drive (Model storage)
```

### Hardware Requirements
**Development:**
- GPU: NVIDIA RTX 3050 4GB (or equivalent)
- RAM: 8GB minimum
- Storage: 10GB for datasets and models

**Production:**
- CPU: 2+ cores
- RAM: 2GB minimum
- Storage: 1GB for model and dependencies

## 🔍 Troubleshooting

### Common Issues
1. **Video not processing**: Ensure MP4 format and reasonable file size (<100MB)
2. **Low accuracy**: Check video quality and lighting conditions
3. **API timeout**: Large videos may take longer, try shorter clips
4. **CORS errors**: Use the web interface instead of direct API calls from browser

### Error Codes
- **503**: Model not loaded (temporary, retry in 30 seconds)
- **400**: Invalid video format or corrupted file
- **500**: Server error (check file size and format)

## 📝 License and Credits

### Acknowledgments
- **UCF-101 Dataset**: University of Central Florida
- **HMDB-51 Dataset**: Human Motion Database
- **PyTorch Team**: For the excellent deep learning framework
- **FastAPI**: For the modern web framework
- **Hugging Face**: For free model hosting

### Tech Stack Credits
- **React + Expo**: Cross-platform frontend development
- **FastAPI**: Modern Python web framework
- **Docker**: Containerization and deployment
- **Vercel**: Frontend hosting and deployment

---

## 🎉 Project Impact

This human activity recognition system demonstrates the practical application of **artificial intelligence in real-world scenarios**. By combining computer vision, deep learning, and web technologies, we've created a system that can understand and interpret human behavior from video footage.

The project showcases modern **deep learning with ml models**, from data collection and model training to API deployment and frontend integration. With 87.34% accuracy across 25 different activities, this system proves that sophisticated AI capabilities can be made accessible through simple, user-friendly interfaces.

**Ready to recognize activities in your videos? Try it now at [our live demo](https://human-activity-recognition6.vercel.app/)!** 🚀

---

*Built with ❤️ using Python, PyTorch, React, and modern cloud technologies*