# Youtube Video Maker App

A powerful web application to generate AI-powered viral clips from YouTube videos.

## Project Structure
- `/frontend`: Self-contained frontend code. Can be hosted on GitHub Pages, Netlify, or Vercel.
- `/backend`: FastAPI backend server. Handles video downloading, AI analysis, and clipping.

## Local Setup

### 1. Backend
1. Go to the `backend` directory.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run the server: `python -m uvicorn main:app --reload`.
   - The API will be available at `http://localhost:8000`.

### 2. Frontend
You can simply open `frontend/index.html` in your browser.
The frontend is configured to connect to `http://localhost:8000` by default.

## Deployment Guide

### GitHub Pages (Frontend)
1. Push your `frontend/` folder to a GitHub repository.
2. In Repository Settings, go to **Pages** and set the source to the `main` branch and the folder to `/` (if you put the contents of `frontend` in the root) or keep the structure.
3. **Important**: Before deploying, open `frontend/script.js` and change `BACKEND_URL` to your live backend URL (e.g., `https://your-backend-app.render.com`).

### Backend Deployment
1. Deploy the `backend` folder to a service like **Render**, **Railway**, or **Heroku**.
2. Ensure you have `ffmpeg` installed on the server environment.
3. The server runs using: `uvicorn main:app --host 0.0.0.0 --port $PORT`.

## Features
- AI Viral Segment Detection
- High-quality Video Clipping (1080p supported)
- Preview Clips before downloading
- Gallery of processed videos
