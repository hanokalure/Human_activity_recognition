# 🚀 Complete Deployment Guide: Hugging Face + Vercel

This guide will help you deploy your Human Activity Recognition backend to Hugging Face Spaces and connect it to your Vercel frontend.

## 📋 Prerequisites

1. **Hugging Face Account**: Sign up at [huggingface.co](https://huggingface.co)
2. **Vercel Account**: Your frontend is already deployed
3. **Git**: Installed and configured
4. **Hugging Face Token**: Create at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

## 🎯 Architecture Overview

```
┌─────────────────┐    HTTPS API calls    ┌─────────────────────┐
│   Vercel        │ ───────────────────→  │  Hugging Face       │
│   Frontend      │                       │  Spaces Backend     │
│                 │                       │                     │
│ • React/Expo    │                       │ • FastAPI + Gradio │
│ • Static Site   │                       │ • PyTorch Model    │
│ • Free Hosting  │                       │ • Free Hosting     │
└─────────────────┘                       └─────────────────────┘
```

## 📁 What's Been Prepared

I've created the following files for your Hugging Face deployment:

```
hf_backend/
├── app.py                  # Main application (FastAPI + Gradio)
├── requirements.txt        # Python dependencies
├── README.md              # Space documentation
└── src/                   # Your model and preprocessing code
    ├── models/
    ├── preprocessing/
    └── utils/
```

## 🚀 Step-by-Step Deployment

### Step 1: Create Hugging Face Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Fill in the details:
   - **Owner**: Hanokalure
   - **Space name**: `human-activity-backend`
   - **License**: MIT
   - **SDK**: Gradio
   - **Visibility**: Public (for free hosting)
3. Click "Create Space"

### Step 2: Deploy Backend to Hugging Face

Run the deployment script:

```powershell
# Option 1: Run the script directly
.\deploy_hf_backend.ps1

# Option 2: If script doesn't run, use these commands manually:
cd C:\ASH_PROJECT\hf_backend
git init
git add .
git commit -m "Initial backend deployment"
git remote add origin https://huggingface.co/spaces/Hanokalure/human-activity-backend
git push -u origin main
```

**When prompted for credentials:**
- Username: `Hanokalure`
- Password: Your Hugging Face token (not your account password!)

### Step 3: Wait for Build

1. Go to https://huggingface.co/spaces/Hanokalure/human-activity-backend
2. Wait for the Space to build (2-5 minutes)
3. You'll see the build logs in the "Logs" tab
4. Once built, you'll see the Gradio interface

### Step 4: Test Backend API

Your API endpoints will be available at:
- **Base URL**: `https://hanokalure-human-activity-backend.hf.space`
- **Health Check**: `https://hanokalure-human-activity-backend.hf.space/health`
- **Prediction**: `https://hanokalure-human-activity-backend.hf.space/predict`

Test with curl:
```bash
curl https://hanokalure-human-activity-backend.hf.space/health
```

### Step 5: Update Vercel Frontend

#### Option A: Via Vercel Dashboard
1. Go to your Vercel project settings
2. Navigate to "Environment Variables"
3. Add or update:
   - **Name**: `EXPO_PUBLIC_API_URL`
   - **Value**: `https://hanokalure-human-activity-backend.hf.space`
   - **Environments**: Production, Preview, Development

#### Option B: Via Command Line
```bash
# If you have Vercel CLI installed
vercel env add EXPO_PUBLIC_API_URL production
# Enter: https://hanokalure-human-activity-backend.hf.space
```

### Step 6: Redeploy Frontend

Trigger a new Vercel deployment:
```bash
cd frontend
git commit --allow-empty -m "Update API URL for HF backend"
git push origin main
```

## ✅ Testing Your Deployment

### 1. Test Hugging Face Interface
- Go to: https://huggingface.co/spaces/Hanokalure/human-activity-backend
- Upload a video using the Gradio interface
- Verify predictions work

### 2. Test Vercel Frontend
- Go to your Vercel app URL
- Upload a video
- Check that it connects to the HF backend
- Verify results display correctly

### 3. Test API Directly
```bash
# Health check
curl https://hanokalure-human-activity-backend.hf.space/health

# Upload test (replace with your video file)
curl -X POST "https://hanokalure-human-activity-backend.hf.space/predict" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@test_video.mp4"
```

## 🔧 Key Features

### Backend (Hugging Face Spaces)
- **FastAPI**: RESTful API endpoints
- **Gradio**: Interactive web interface
- **CORS**: Configured for your Vercel domain
- **Auto Download**: Model downloads from Google Drive
- **Free Hosting**: No cost on HF Community tier

### Frontend (Vercel)
- **Environment Variables**: Dynamically points to HF backend
- **Error Handling**: Graceful fallbacks
- **Responsive**: Works on mobile and desktop
- **Free Hosting**: Vercel's generous free tier

## 🛠 Troubleshooting

### Backend Issues
```bash
# Check HF Space logs
# Go to Space > Logs tab in HF interface

# Common issues:
# 1. Model download fails - check Google Drive link permissions
# 2. Import errors - verify all files are uploaded
# 3. Memory issues - model is large, may take time to load
```

### Frontend Issues
```bash
# Check Vercel deployment logs
# Go to Vercel Dashboard > Deployments

# Common issues:
# 1. CORS errors - verify domain in backend CORS config
# 2. API timeout - video processing can take 30+ seconds
# 3. Environment variable not set - check Vercel settings
```

### API Connection Issues
```bash
# Test connection
curl -v https://hanokalure-human-activity-backend.hf.space/health

# Check CORS
curl -X OPTIONS -H "Origin: https://your-vercel-domain.vercel.app" \
     -v https://hanokalure-human-activity-backend.hf.space/predict
```

## 💰 Cost Breakdown

| Service | Tier | Cost | Limits |
|---------|------|------|--------|
| Hugging Face Spaces | Community | FREE | CPU-only, may sleep after inactivity |
| Vercel | Hobby | FREE | 100GB bandwidth/month |
| **Total** | | **$0/month** | Perfect for portfolio/demo projects |

## 🚨 Important Notes

1. **HF Spaces Sleeping**: Free spaces sleep after 48h of inactivity. First request after sleeping takes longer.

2. **Model Loading**: The 87MB model downloads on first run, which may take 30-60 seconds.

3. **CORS Configuration**: Backend is pre-configured for your Vercel domain.

4. **API Rate Limits**: HF Spaces has reasonable rate limits for free tier.

## 🎉 You're Done!

Your architecture:
- ✅ Backend: Hugging Face Spaces (Free)
- ✅ Frontend: Vercel (Free) 
- ✅ Model: Auto-downloaded from Google Drive
- ✅ CORS: Properly configured
- ✅ APIs: RESTful endpoints + Gradio interface

Both services offer excellent free tiers, making this a sustainable solution for your ML application!