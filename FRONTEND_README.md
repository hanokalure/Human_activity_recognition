# 🚀 Daily Activities Recognition - Frontend Implementation Guide

## 📱 React Native Expo + FastAPI Architecture

**Target Platforms**: iOS, Android, Web (Single Codebase)  
**Backend**: FastAPI + Your existing PyTorch models  
**Frontend**: React Native with Expo  

---

## 🎯 Project Architecture Overview

```
ASH_PROJECT/
├── 🐍 backend/                    # FastAPI Backend
│   ├── main.py                   # FastAPI main server
│   ├── api/
│   │   ├── video_upload.py       # Video upload endpoints
│   │   ├── inference.py          # Model inference endpoints
│   │   └── activities.py         # Activity management
│   ├── models/
│   │   └── model_manager.py      # PyTorch model loading
│   ├── utils/
│   │   ├── video_processor.py    # Video preprocessing
│   │   └── response_handler.py   # API response formatting
│   └── requirements.txt
├── 📱 frontend/                   # React Native Expo Frontend
│   ├── src/
│   │   ├── components/           # Reusable UI components
│   │   ├── screens/             # App screens
│   │   ├── services/            # API services
│   │   └── utils/               # Helper functions
│   ├── App.tsx                  # Main app entry
│   └── package.json
└── 🐳 docker/                    # Deployment configs
    ├── Dockerfile.backend
    └── docker-compose.yml
```

---

## 🔄 Application Flow Diagram

```
📱 Mobile/Web App                    🐍 FastAPI Backend                  🧠 AI Models
┌─────────────────┐                 ┌──────────────────┐                ┌─────────────────┐
│  Video Capture  │────────────────▶│  /upload-video   │───────────────▶│  Video Processor │
│  (Camera/File)  │                 │                  │                │                 │
└─────────────────┘                 └──────────────────┘                └─────────────────┘
         │                                   │                                   │
         ▼                                   ▼                                   ▼
┌─────────────────┐                 ┌──────────────────┐                ┌─────────────────┐
│  Upload Progress│◀────────────────│  WebSocket/SSE   │◀───────────────│  Model Inference │
│  & Status       │                 │  Progress Updates│                │  (PyTorch)      │
└─────────────────┘                 └──────────────────┘                └─────────────────┘
         │                                   │                                   │
         ▼                                   ▼                                   ▼
┌─────────────────┐                 ┌──────────────────┐                ┌─────────────────┐
│  Results Display│◀────────────────│  /get-results    │◀───────────────│  Activity Results│
│  & Analytics    │                 │  JSON Response   │                │  + Confidence   │
└─────────────────┘                 └──────────────────┘                └─────────────────┘
```

---

## 🛠️ Implementation Roadmap

### **Phase 1: Backend API Setup** ⚡ (1-2 days)

#### Step 1.1: Create FastAPI Structure
```bash
# In ASH_PROJECT directory
mkdir backend
cd backend
```

**Files to create:**
- `main.py` - FastAPI server with CORS
- `api/video_upload.py` - Video upload handling
- `api/inference.py` - Model inference endpoints
- `models/model_manager.py` - Load your trained models
- `requirements.txt` - Python dependencies

#### Step 1.2: Key API Endpoints to Implement

```python
# Backend API Endpoints Structure

POST /api/upload-video
├── Accept: multipart/form-data
├── Body: video file + metadata
├── Response: upload_id + status
└── Triggers: Background inference task

GET /api/inference-status/{upload_id}
├── Response: progress percentage + status
├── WebSocket alternative for real-time updates
└── Status: processing | completed | error

GET /api/results/{upload_id}
├── Response: activity predictions + confidence scores
├── Includes: video-to-text description
└── Format: JSON with activity rankings

GET /api/activities
├── Response: List of 22 supported activities
├── Includes: descriptions + categories
└── For frontend dropdowns/info

POST /api/camera-stream (Future Phase)
├── Accept: Base64 video chunks
├── Real-time processing
└── WebSocket responses
```

#### Step 1.3: Integration with Your Models
```python
# models/model_manager.py - Connect to your existing code
from src.daily_activities_inference import DailyActivitiesInference
from src.daily_activities_config import DAILY_ACTIVITIES, ACTIVITY_DESCRIPTIONS

class ModelManager:
    def __init__(self):
        model_path = "../outputs/checkpoints/best_model.pth"
        self.inference = DailyActivitiesInference(model_path)
    
    async def predict_video(self, video_path):
        result = self.inference.predict(video_path)
        return {
            "activity": result["activity"],
            "confidence": result["confidence"],
            "description": ACTIVITY_DESCRIPTIONS[result["activity"]],
            "all_predictions": result["top_predictions"]
        }
```

---

### **Phase 2: React Native Expo Frontend** 📱 (3-4 days)

#### Step 2.1: Initialize Expo Project
```bash
# Install Expo CLI
npm install -g @expo/cli

# Create new Expo project
cd ASH_PROJECT
npx create-expo-app frontend --template typescript
cd frontend

# Install dependencies
npx expo install expo-camera expo-av expo-document-picker
npm install axios react-native-elements react-native-paper
npm install @react-navigation/native @react-navigation/stack
```

#### Step 2.2: Project Structure to Create

```
frontend/src/
├── components/
│   ├── VideoUploader.tsx        # Video file picker
│   ├── CameraRecorder.tsx       # Live camera recording
│   ├── ProgressBar.tsx          # Upload/processing progress
│   ├── ResultsCard.tsx          # Activity results display
│   ├── ActivityList.tsx         # Supported activities info
│   └── LoadingSpinner.tsx       # Loading states
├── screens/
│   ├── HomeScreen.tsx           # Main landing page
│   ├── CameraScreen.tsx         # Camera recording
│   ├── UploadScreen.tsx         # File upload
│   ├── ResultsScreen.tsx        # Results display
│   └── ActivitiesScreen.tsx     # Activities info
├── services/
│   ├── api.ts                   # API calls to FastAPI
│   ├── videoService.ts          # Video handling utilities
│   └── constants.ts             # App constants
└── utils/
    ├── fileUtils.ts             # File processing
    └── formatters.ts            # Data formatting
```

#### Step 2.3: Key Frontend Components Flow

**1. Home Screen → Choice**
```typescript
// HomeScreen.tsx - Main navigation
<HomeScreen>
  ├── Upload Video File (→ UploadScreen)
  ├── Record with Camera (→ CameraScreen)  
  ├── View Activities Info (→ ActivitiesScreen)
  └── Recent Results History
</HomeScreen>
```

**2. Video Upload Flow**
```typescript
// UploadScreen.tsx + VideoUploader.tsx
const uploadFlow = {
  1. "Select video file (expo-document-picker)",
  2. "Show video preview",
  3. "Display upload progress bar",
  4. "Call API: POST /api/upload-video",
  5. "Poll status: GET /api/inference-status/{id}",
  6. "Navigate to ResultsScreen when complete"
}
```

**3. Camera Recording Flow**
```typescript
// CameraScreen.tsx + CameraRecorder.tsx  
const cameraFlow = {
  1. "Request camera permissions",
  2. "Live camera preview with expo-camera",
  3. "Record button with countdown timer",
  4. "Save video locally",
  5. "Auto-upload to backend",
  6. "Show processing status"
}
```

**4. Results Display**
```typescript
// ResultsScreen.tsx + ResultsCard.tsx
const resultsDisplay = {
  activity: "pushups",
  confidence: 0.94,
  description: "Person is doing push-ups",
  allPredictions: [
    { activity: "pushups", confidence: 0.94 },
    { activity: "exercise", confidence: 0.83 },
    // ... top 5 predictions
  ],
  processingTime: "2.3s",
  videoThumbnail: "base64_image"
}
```

---

### **Phase 3: API Integration & Services** 🔌 (1-2 days)

#### Step 3.1: API Service Layer
```typescript
// services/api.ts
class ApiService {
  private baseURL = 'http://localhost:8000/api'; // Your FastAPI server
  
  async uploadVideo(videoUri: string, metadata: object) {
    const formData = new FormData();
    formData.append('video', {
      uri: videoUri,
      type: 'video/mp4',
      name: 'video.mp4',
    } as any);
    
    return fetch(`${this.baseURL}/upload-video`, {
      method: 'POST',
      body: formData,
      headers: { 'Content-Type': 'multipart/form-data' }
    });
  }
  
  async getInferenceStatus(uploadId: string) {
    return fetch(`${this.baseURL}/inference-status/${uploadId}`);
  }
  
  async getResults(uploadId: string) {
    return fetch(`${this.baseURL}/results/${uploadId}`);
  }
  
  async getSupportedActivities() {
    return fetch(`${this.baseURL}/activities`);
  }
}
```

#### Step 3.2: Video Service Utilities
```typescript
// services/videoService.ts
class VideoService {
  async compressVideo(videoUri: string): Promise<string> {
    // Video compression for mobile upload
  }
  
  async generateThumbnail(videoUri: string): Promise<string> {
    // Generate video thumbnail
  }
  
  validateVideoFile(videoUri: string): boolean {
    // Check file size, format, duration limits
  }
  
  async uploadWithProgress(
    videoUri: string, 
    onProgress: (progress: number) => void
  ): Promise<any> {
    // Upload with progress tracking
  }
}
```

---

### **Phase 4: Advanced Features** 🚀 (2-3 days)

#### Step 4.1: Real-time Camera Analysis (Optional)
```typescript
// Real-time activity recognition through camera
const RealTimeCamera = () => {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [currentActivity, setCurrentActivity] = useState(null);
  
  const analyzeFrame = async (base64Frame: string) => {
    // Send frame to backend for real-time analysis
    const result = await api.analyzeFrame(base64Frame);
    setCurrentActivity(result);
  };
  
  return (
    <CameraView onFrameCapture={analyzeFrame}>
      <ActivityOverlay activity={currentActivity} />
    </CameraView>
  );
};
```

#### Step 4.2: Batch Processing
```typescript
// Multiple video upload and batch processing
const BatchUpload = () => {
  const [videoQueue, setVideoQueue] = useState([]);
  const [processingStatus, setProcessingStatus] = useState({});
  
  const uploadBatch = async (videos: string[]) => {
    for (const video of videos) {
      const uploadId = await api.uploadVideo(video);
      // Track each video's processing status
    }
  };
};
```

#### Step 4.3: Results History & Analytics
```typescript
// Local storage for results history
class ResultsManager {
  async saveResult(result: ActivityResult): Promise<void> {
    // Save to AsyncStorage
  }
  
  async getHistory(): Promise<ActivityResult[]> {
    // Retrieve past results
  }
  
  generateAnalytics(results: ActivityResult[]): Analytics {
    // Activity frequency, accuracy trends, etc.
  }
}
```

---

### **Phase 5: Platform-Specific Features** 🌐📱

#### Step 5.1: Web-Specific Features
```typescript
// Web platform optimizations
if (Platform.OS === 'web') {
  // Drag & drop video upload
  // Larger screen layouts
  // Keyboard shortcuts
  // File system access
}
```

#### Step 5.2: Mobile-Specific Features  
```typescript
// Mobile optimizations
if (Platform.OS !== 'web') {
  // Push notifications for processing complete
  // Background processing
  // Device storage management
  // Camera permissions handling
}
```

---

## 📦 Dependencies & Installation

### Backend Requirements
```python
# backend/requirements.txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6
python-opencv==4.8.1.78
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pillow>=10.0.0
aiofiles==23.2.1
websockets==12.0
```

### Frontend Dependencies
```json
{
  "dependencies": {
    "expo": "~49.0.0",
    "react": "18.2.0",
    "react-native": "0.72.6",
    "expo-camera": "~13.4.4",
    "expo-av": "~13.4.1",
    "expo-document-picker": "~11.5.4",
    "axios": "^1.5.0",
    "react-navigation": "^6.0.0",
    "react-native-elements": "^3.4.3",
    "react-native-paper": "^5.10.0"
  }
}
```

---

## 🚀 Development Workflow

### Day 1-2: Backend Setup
```bash
# 1. Create FastAPI backend
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# 2. Test model integration
python -c "from models.model_manager import ModelManager; print('Models loaded!')"

# 3. Start development server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Day 3-4: Frontend Setup
```bash
# 1. Initialize Expo project
cd frontend
npm install
npx expo install

# 2. Start development
npx expo start
# Choose platform: web (w), iOS simulator (i), Android (a)

# 3. Test API connection
# Update API baseURL to your backend server
```

### Day 5-6: Integration & Testing
```bash
# 1. End-to-end testing
# Upload video → Processing → Results display

# 2. Platform testing
# Test on web browser, iOS simulator, Android emulator

# 3. Performance optimization
# Video compression, API response caching
```

---

## 🌐 Deployment Options

### **Option 1: Local Development**
```bash
# Backend: http://localhost:8000
# Frontend Web: http://localhost:19006 
# Mobile: Expo Go app with QR code
```

### **Option 2: Cloud Deployment**
```yaml
# docker-compose.yml
version: '3.8'
services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    volumes:
      - ./outputs:/app/outputs
  
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    depends_on:
      - backend
```

### **Option 3: Mobile App Store**
```bash
# Build for production
npx expo build:android  # APK for Google Play
npx expo build:ios      # IPA for App Store
```

---

## 🎯 Key Features to Implement

### **Core Features (Must Have)**
- ✅ Video upload from device gallery
- ✅ Camera recording and upload
- ✅ Activity recognition with confidence scores
- ✅ Progress tracking during processing
- ✅ Results display with descriptions
- ✅ List of supported activities

### **Enhanced Features (Nice to Have)**
- 🔄 Real-time camera analysis
- 🔄 Batch video processing
- 🔄 Results history and analytics
- 🔄 Social sharing of results
- 🔄 Offline mode with local storage
- 🔄 Push notifications

### **Advanced Features (Future)**
- 🚀 Multi-person activity detection
- 🚀 Activity sequence analysis
- 🚀 Custom model training interface
- 🚀 Activity coaching/feedback
- 🚀 Integration with fitness trackers

---

## 📊 Performance Considerations

### **Backend Optimization**
- Asynchronous video processing
- Queue management for multiple uploads
- Model caching and warm-up
- Video compression before processing
- Progress tracking with WebSockets

### **Frontend Optimization**
- Video compression before upload
- Progressive image loading
- Offline-first architecture
- Efficient state management
- Platform-specific optimizations

---

## 🛡️ Security & Privacy

### **Data Protection**
- Temporary video storage (auto-delete after processing)
- No personal data collection
- Local processing options
- HTTPS/SSL encryption
- File type validation

### **API Security**
- Rate limiting on endpoints
- File size limits
- Input validation
- CORS configuration
- API key authentication (optional)

---

## 🎉 Getting Started Checklist

### **Before You Begin**
- [ ] Ensure your trained model works with `daily_activities_inference.py`
- [ ] Test video processing on a sample video
- [ ] Install Node.js (v18+) and Python (3.8+)
- [ ] Choose your development approach (web-first or mobile-first)

### **Quick Start Commands**
```bash
# 1. Clone/setup backend
mkdir backend && cd backend
# ... implement FastAPI structure

# 2. Initialize frontend
npx create-expo-app frontend --template typescript
cd frontend && npm install

# 3. Test integration
# Start backend: uvicorn main:app --reload
# Start frontend: npx expo start
```

---

## 🎯 Success Metrics

### **Technical Goals**
- Video upload success rate: >95%
- Processing time: <30 seconds per video
- App load time: <3 seconds
- Cross-platform compatibility: iOS, Android, Web

### **User Experience Goals**
- Intuitive video upload flow
- Clear progress indication
- Accurate activity recognition results
- Responsive design across devices

---

**🚀 Ready to build an amazing Daily Activities Recognition app!**

*Start with the backend API integration, then move to the React Native frontend. Focus on core video upload → processing → results flow first, then add advanced features.*

**Need help with any specific implementation step? Let me know!** 🤝