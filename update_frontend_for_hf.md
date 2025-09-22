# Update Frontend for Hugging Face Backend

After deploying your backend to Hugging Face Spaces, you need to update your Vercel frontend to use the new API URL.

## Step 1: Get Your Hugging Face Spaces URL

Your backend will be available at:
```
https://hanokalure-human-activity-backend.hf.space
```

## Step 2: Update Vercel Environment Variable

### Option A: Via Vercel Dashboard
1. Go to your Vercel project dashboard
2. Navigate to "Settings" > "Environment Variables"
3. Update or add the `EXPO_PUBLIC_API_URL` variable:
   - **Name**: `EXPO_PUBLIC_API_URL`
   - **Value**: `https://hanokalure-human-activity-backend.hf.space`
   - **Environments**: Select "Production", "Preview", and "Development"

### Option B: Via Vercel CLI
```bash
# Install Vercel CLI if you haven't already
npm i -g vercel

# Login to Vercel
vercel login

# Set the environment variable
vercel env add EXPO_PUBLIC_API_URL production
# When prompted, enter: https://hanokalure-human-activity-backend.hf.space

# Also set for preview and development
vercel env add EXPO_PUBLIC_API_URL preview
vercel env add EXPO_PUBLIC_API_URL development
```

## Step 3: Redeploy Frontend

After updating the environment variable, trigger a new deployment:

### Via Vercel Dashboard
1. Go to your project's "Deployments" tab
2. Click "Redeploy" on the latest deployment

### Via Git Push
```bash
# Make a small change to trigger redeploy
cd frontend
echo "# Updated for HF backend" >> README.md
git add .
git commit -m "Update API URL for Hugging Face backend"
git push origin main
```

## Step 4: Test the Integration

1. Wait for both deployments to complete:
   - Hugging Face Spaces build (2-5 minutes)
   - Vercel redeploy (1-2 minutes)

2. Test your Vercel frontend:
   - Go to your Vercel app URL
   - Upload a video
   - Verify it connects to the HF backend

3. Test the HF Gradio interface:
   - Go to `https://huggingface.co/spaces/Hanokalure/human-activity-backend`
   - Try the Gradio interface directly

## Troubleshooting

### If CORS errors occur:
The backend is configured with CORS for your Vercel domain. If you have a different domain, update the CORS origins in `app.py`:

```python
allow_origins=[
    "https://your-actual-vercel-domain.vercel.app",
    "https://human-activity-recognition-frontend-phi.vercel.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://*.vercel.app"
],
```

### If API calls fail:
1. Check that the HF Space is running (not sleeping)
2. Verify the API URL in Vercel environment variables
3. Check browser network tab for detailed error messages
4. Test the API directly with curl:
   ```bash
   curl https://hanokalure-human-activity-backend.hf.space/health
   ```

## Architecture Overview

```
Frontend (Vercel) ──→ Backend API (Hugging Face Spaces)
     │                        │
     │                        ├── FastAPI endpoints
     │                        ├── Model inference
     │                        └── Gradio interface
     │
└── Static React app       ── PyTorch model
```

Both platforms offer free hosting, making this a cost-effective solution for your ML application!