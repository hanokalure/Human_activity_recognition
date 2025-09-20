# 🎯 Phase 5 Activity Recognition - Complete Setup Guide

A complete AI-powered activity recognition system with FastAPI backend and React Native/Expo frontend supporting both web and mobile platforms.

## 🏗️ Architecture Overview

```
C:\ASH_PROJECT\
├── backend/                 # FastAPI server
│   ├── main.py             # API server with WebSocket support
│   ├── requirements.txt    # Python dependencies
│   └── start_server.py     # Server launcher script
├── frontend/               # Expo/React Native app
│   ├── src/
│   │   ├── components/     # UI components
│   │   ├── services/       # API communication
│   │   └── types/          # TypeScript definitions
│   ├── App.tsx            # Main app component
│   └── package.json       # Node dependencies
├── src/                    # Original inference code
│   └── phase5_inference.py # Your existing model
└── models/phase5/          # Trained model files
    └── phase5_best_model.pth
```

## 🚀 Quick Start

### 1. Backend Setup (FastAPI)

```bash
# Install Python dependencies
cd C:\ASH_PROJECT
pip install -r backend\requirements.txt

# Start the API server
python backend\start_server.py
```

The server will be available at: `http://localhost:8000`
API documentation: `http://localhost:8000/docs`

### 2. Frontend Setup (Expo)

```bash
# Install Node.js dependencies
cd C:\ASH_PROJECT\frontend
yarn install
# or: npm install

# Start the Expo development server
yarn start
# or: npm start
```

### 3. Running the App

#### Web Version
- Press `w` in the Expo CLI or navigate to `http://localhost:19006`

#### Mobile Version (iOS/Android)
- Install Expo Go app on your device
- Scan the QR code from Expo CLI
- Or run on emulator: `yarn android` / `yarn ios`

## 📱 Features

### 🎬 Video Upload Mode
- **File Selection**: Pick videos from device storage
- **Supported Formats**: MP4, AVI, MOV, MKV, WMV  
- **Analysis Results**: Activity prediction with confidence scores
- **Probability Breakdown**: Top 5 most likely activities

### 📷 Live Camera Mode
- **Real-time Streaming**: Live camera feed with WebSocket connection
- **Frame Collection**: Collects 16 frames for model input
- **Live Predictions**: Continuous activity recognition
- **Camera Controls**: Switch between front/back camera

### 🎯 Activity Recognition
**25 Daily Activities Supported:**
- **Movement**: walking, running, climbing stairs, biking
- **Exercise**: pullups, pushups, yoga, weight lifting  
- **Daily Care**: brushing teeth/hair, eating, drinking
- **Work**: typing, writing, talking
- **Home**: cooking, cleaning, pouring
- **Social**: hugging, waving, laughing
- **Recreation**: walking dog, swimming (breast stroke, front crawl)
- **Rest**: sitting

## 🔧 Development

### Backend API Endpoints

#### REST Endpoints
- `GET /health` - Check server and model status
- `POST /predict/video` - Upload video for analysis

#### WebSocket Endpoint
- `WS /ws/frames` - Real-time frame streaming
  - Send binary data (JPEG frames)
  - Send JSON control: `{"type": "start"}` or `{"type": "end"}`
  - Receive predictions: `{"type": "prediction", "data": {...}}`

### Environment Variables
```bash
# Optional: Set custom model path
export PHASE5_MODEL_PATH="C:\ASH_PROJECT\models\phase5\phase5_best_model.pth"
```

## 📊 Model Details

- **Architecture**: R(2+1)D ResNet
- **Input**: 16 frames, 112x112 pixels
- **Classes**: 25 daily activities
- **Model Size**: ~129MB
- **Best Accuracy**: Check training logs in `models/phase5/`

## 🛠️ Troubleshooting

### Backend Issues
```bash
# Check model file exists
ls -la C:\ASH_PROJECT\models\phase5\phase5_best_model.pth

# Test API manually
curl http://localhost:8000/health

# Check dependencies
pip list | grep -E "fastapi|uvicorn|torch"
```

### Frontend Issues
```bash
# Clear Expo cache
cd frontend
expo r -c

# Check Node dependencies
yarn list --pattern expo

# For web issues
yarn add @expo/webpack-config@latest
```

### Permission Issues (Mobile)
- **Camera**: Grant camera permissions in device settings
- **File Access**: Allow storage permissions for video upload

## 📈 Performance Tips

### Backend Optimization
- Use GPU if available (CUDA)
- Adjust frame rate in WebSocket streaming (default: 2 fps)
- Consider model quantization for faster inference

### Frontend Optimization
- Lower camera quality for streaming (currently 0.3)
- Implement frame rate limiting
- Add connection retry logic

## 🔒 Production Deployment

### Backend
```bash
# Use production ASGI server
pip install gunicorn
gunicorn backend.main:app -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000

# Or with Docker
docker build -t phase5-api .
docker run -p 8000:8000 phase5-api
```

### Frontend
```bash
# Build for production
expo build:web
expo build:android
expo build:ios
```

### Security Notes
- Update CORS settings in `backend/main.py`
- Use HTTPS in production
- Implement rate limiting
- Add authentication if needed

## 🧪 Testing

### Backend Testing
```bash
# Test video upload
curl -X POST http://localhost:8000/predict/video \
  -F "file=@test_video.mp4"

# Test WebSocket (using wscat)
npm install -g wscat
wscat -c ws://localhost:8000/ws/frames
```

### Frontend Testing
```bash
cd frontend
yarn test
```

## 🎨 Customization

### Adding New Activities
1. Update `ACTIVITY_CATEGORIES` in `src/types/index.ts`
2. Add emoji mappings in `ResultsDisplay.tsx`
3. Retrain model with new classes

### UI Theming
- Modify colors in component StyleSheets
- Update `App.tsx` for global theme
- Add dark mode support

## 📞 Support

For issues or questions:
1. Check the GitHub Issues
2. Review API logs: `http://localhost:8000/docs`
3. Expo development logs
4. Model inference logs in console

## 🔄 Updates

To update the system:
1. Pull latest changes
2. Update backend dependencies: `pip install -r backend/requirements.txt`
3. Update frontend dependencies: `cd frontend && yarn install`
4. Restart both servers

---

**🎉 Enjoy your AI-powered activity recognition app!**