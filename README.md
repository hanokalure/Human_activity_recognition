# 🎯 Human Activity Recognition System

A deep learning system for recognizing 25 daily human activities from video using R(2+1)D CNN architecture.

## 🏆 Model Performance

- **Phase 5 Complete** ✅
- **25 Activity Classes** trained 
- **87.34% Validation Accuracy** achieved
- **Real-time inference** capable (~30 FPS)
- **Training Time**: 8.97 hours

## 📊 Trained Activities (25 Classes)

<details>
<summary>View All 25 Activities</summary>

1. biking_ucf
2. breast_stroke  
3. brushing_hair
4. brushing_teeth
5. cleaning
6. climbing_stairs
7. cooking_ucf
8. drinking
9. eating
10. front_crawl
11. hugging
12. laughing
13. pouring
14. pullups_ucf
15. pushups_ucf
16. running
17. sitting
18. talking
19. typing
20. walking
21. walking_dog
22. waving
23. weight_lifting
24. writing
25. yoga

</details>

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Automated setup (recommended)
python install_dependencies.py

# Or manual install
pip install -r requirements.txt
```

### 2. Run Inference
```bash
# Single video prediction
python src/phase5_inference.py --video path/to/video.mp4

# Real-time webcam
python src/phase5_inference.py --webcam

# Batch processing
python src/phase5_inference.py --batch_dir videos/ --output results.json
```

### 3. Start API Server
```bash
# Backend API
python backend/main.py

# Frontend (if needed)
cd frontend && npm start
```

## 🏗️ Technical Details

**Please confirm the following details:**

1. **Model Architecture**: R(2+1)D CNN? (detected from code)
2. **Datasets Used**: UCF-101 + HMDB-51? (detected from config files)
3. **Input Format**: 16 frames, 112x112 resolution? (from inference code)
4. **Framework**: PyTorch + FastAPI? (detected from files)
5. **Training Strategy**: Transfer learning from pretrained weights?

**Model Specs:**
- Input: Video clips (16 frames, 112x112px)
- Architecture: R(2+1)D with custom classification head
- Output: 25-class probability distribution
- Model Size: 123.7 MB

## 📁 Project Structure

```
├── backend/           # FastAPI server
├── frontend/          # React/Expo app  
├── src/               # ML training & inference
│   ├── phase5_inference.py    # Main inference script
│   ├── phase5_train.py        # Training script
│   └── phase5_config.py       # Dataset configuration
├── models/phase5/     # Trained model files
├── requirements.txt   # Dependencies
└── install_dependencies.py   # Setup script
```

## 🌐 Deployment

### Render (Recommended)
```bash
# One-click deployment with Blueprint
# render.yaml included for full-stack deployment
```

### Local Development
```bash
# Backend: http://localhost:8000
python backend/main.py

# Frontend: http://localhost:3000  
cd frontend && npm start
```

## 🎮 Usage Examples

**Video Classification:**
```python
from src.phase5_inference import Phase5Predictor

predictor = Phase5Predictor('models/phase5/phase5_best_model.pth')
result = predictor.predict_video('video.mp4')

print(f"Activity: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2f}")
```

**API Endpoint:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "video=@video.mp4"
```

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- OpenCV 4.8+
- FastAPI 0.104+
- 4GB+ RAM (8GB recommended)
- CUDA GPU (optional, for training)

## 🏁 Results

- **Validation Accuracy**: 87.34%
- **Training Time**: 8.97 hours  
- **Model Size**: 123.7 MB
- **Inference Speed**: Real-time capable

---

*Built with PyTorch • FastAPI • React*