# Human Activity Recognition - Local + Vercel Deployment

This project runs with a **local backend** and **Vercel-hosted frontend**.

## 🏗️ Architecture

- **Backend:** Runs locally on `http://127.0.0.1:8000`
- **Frontend:** Deployed on Vercel at https://github.com/hanokalure/Human-Activity-Recognition-Frontend.git
- **Model:** Local PyTorch model (25 classes, 87.34% accuracy)

## 🚀 Local Backend Setup

### Prerequisites
- Python 3.x with PyTorch 2.7.1+cu118
- Model file at: `C:\ASH_PROJECT\models\phase5\phase5_best_model.pth`

### Installation
```bash
# Install dependencies
pip install -r backend/requirements.txt

# Run the API server
python backend/start_server.py
```

### Verify Installation
- **API Health:** http://127.0.0.1:8000/health
- **API Docs:** http://127.0.0.1:8000/docs  
- **Expected Response:** `{"status": "ok", "model_loaded": true}`

## 🌐 Frontend Access

### Production (Vercel)
- **URL:** https://your-app.vercel.app
- **Status:** UI-only (no predictions - shows connection error)
- **Note:** Vercel frontend cannot reach your local backend

### Local Development
- **Access:** http://localhost:19006 (when running `npm run web`)
- **Status:** Full functionality with API predictions
- **Connects to:** http://127.0.0.1:8000

## 📱 Usage Instructions

### For Full Functionality (Local)
1. Start backend: `python backend/start_server.py`
2. Wait for: `INFO: Uvicorn running on http://127.0.0.1:8000`
3. Access: http://localhost:19006 (if running frontend locally)
4. Upload video and get predictions

### For Demo (Vercel)
1. Visit: https://your-app.vercel.app
2. UI loads but shows "Server Unavailable"
3. This is expected - backend runs locally only

## 🔧 Configuration

### Backend Environment Variables
Create `backend/.env` (optional):
```
PHASE5_MODEL_PATH=C:\ASH_PROJECT\models\phase5\phase5_best_model.pth
```

### Frontend Repository
- **Main repo:** https://github.com/hanokalure/Human_activity_recognition.git
- **Frontend repo:** https://github.com/hanokalure/Human-Activity-Recognition-Frontend.git
- **Vercel deploys from:** Frontend repo only

## 🎯 Expected Behavior

### When Backend is Running
- ✅ Health check passes
- ✅ Video upload works  
- ✅ Activity predictions return
- ✅ 25 classes supported
- ✅ WebSocket streaming available

### When Backend is Stopped
- ❌ Frontend shows "Server Unavailable"
- ❌ Video upload fails gracefully
- ❌ Retry button allows reconnection
- ✅ Frontend UI still loads and looks good

## 📊 Model Information
- **Classes:** 25 daily life activities
- **Accuracy:** 87.34%
- **Size:** ~130MB
- **Device:** CPU optimized
- **Framework:** PyTorch 2.7.1+cu118