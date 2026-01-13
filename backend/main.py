from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import os
import sys
import uuid
import asyncio
from typing import Dict

# Add backend directory to Python path for imports
sys.path.append(os.path.dirname(__file__))

# Import your upcoming modules here (e.g., from downloader import download_video)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for task status
tasks: Dict[str, Dict] = {}

# Resolve paths relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
PROCESSED_DIR = os.path.join(STATIC_DIR, "processed")
DOWNLOADS_DIR = os.path.join(BASE_DIR, "downloads")

# Ensure directories exist
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(DOWNLOADS_DIR, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def read_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"), media_type="text/html")

@app.get("/analyze.html")
async def read_analyze():
    return FileResponse(os.path.join(STATIC_DIR, "analyze.html"), media_type="text/html")

@app.get("/downloads.html")
async def read_downloads():
    return FileResponse(os.path.join(STATIC_DIR, "downloads.html"), media_type="text/html")

@app.get("/gallery.html")
async def read_gallery():
    return FileResponse(os.path.join(STATIC_DIR, "gallery.html"), media_type="text/html")

# Static file serving is removed for frontend files as they are now in /frontend
# However, we still serve processed files if needed, but via specific endpoints.

class ProcessRequest(BaseModel):
    urls: list[str]  # Support multiple URLs for batch processing
    duration_mode: str = "30" # 30, 60, 120, custom

class ClipRequest(BaseModel):
    video_id: str
    start: float
    end: float
    quality: str = "1080"
    title: str
    platform: str = None

class EditRequest(BaseModel):
    video_id: str
    operation: str  # 'trim', 'merge', 'add_text', 'add_music'
    params: dict

class SocialMediaRequest(BaseModel):
    video_path: str
    platform: str  # 'tiktok', 'instagram', 'twitter'
    title: str
    hashtags: list[str] = []

class CaptionRequest(BaseModel):
    video_id: str

class TitleDescriptionRequest(BaseModel):
    video_id: str
    segment_start: float = 0
    segment_end: float = 30
    platform: str = "youtube"

class ThumbnailRequest(BaseModel):
    video_id: str
    timestamp: float = 0

class SubtitleRequest(BaseModel):
    video_id: str
    language: str = "en"

async def analyze_video_task(task_id: str, url: str, duration_mode: str):
    tasks[task_id]["status"] = "downloading"
    tasks[task_id]["progress"] = 10

    try:
        from processing import processor

        # 1. Download
        tasks[task_id]["status"] = "downloading"
        video_path, info = await asyncio.to_thread(processor.download_video, url)
        tasks[task_id]["progress"] = 50
        tasks[task_id]["video_path"] = video_path # Store this for later clipping

        # 2. Analyze
        tasks[task_id]["status"] = "analyzing"
        duration = info.get('duration', 600)
        segments = await asyncio.to_thread(processor.analyze_viral_segments, video_path, duration, duration_mode)

        tasks[task_id]["status"] = "completed"
        tasks[task_id]["progress"] = 100
        tasks[task_id]["segments"] = segments
        tasks[task_id]["video_id"] = task_id # Use task_id as session/video id for now

    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)
        print(f"Error processing task {task_id}: {e}")

@app.post("/api/analyze")
async def start_analysis(background_tasks: BackgroundTasks, request: ProcessRequest):
    task_ids = []
    for url in request.urls:
        task_id = str(uuid.uuid4())
        tasks[task_id] = {
            "status": "queued",
            "progress": 0,
            "url": url
        }
        background_tasks.add_task(analyze_video_task, task_id, url, request.duration_mode)
        task_ids.append(task_id)

    return {"task_ids": task_ids}

@app.post("/api/generate-clip")
async def generate_clip(request: ClipRequest):
    # Find the task that holds the video path
    # In a real app, use a DB. Here we look up by ID which is the task_id
    session = tasks.get(request.video_id)
    if not session or "video_path" not in session:
        raise HTTPException(status_code=404, detail="Session expired or video not found")

    video_path = session["video_path"]
    print(f"Generating clip for video: {video_path}")
    print(f"Clip params: start={request.start}, end={request.end}, quality={request.quality}")

    try:
        from processing import processor
        # Generate on demand
        clip_url = await asyncio.to_thread(
            processor.create_single_clip,
            video_path,
            request.start,
            request.end,
            request.quality
        )
        print(f"Clip generated successfully: {clip_url}")
        return {"url": clip_url}
    except Exception as e:
        print(f"Error generating clip: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/edit-video")
async def edit_video(request: EditRequest):
    session = tasks.get(request.video_id)
    if not session or "video_path" not in session:
        raise HTTPException(status_code=404, detail="Video not found")

    video_path = session["video_path"]

    try:
        from processing import processor

        if request.operation == "trim":
            result_url = await asyncio.to_thread(
                processor.trim_video,
                video_path,
                request.params.get("start", 0),
                request.params.get("end", 10)
            )
        elif request.operation == "merge":
            result_url = await asyncio.to_thread(
                processor.merge_videos,
                request.params.get("video_paths", [])
            )
        elif request.operation == "add_text":
            result_url = await asyncio.to_thread(
                processor.add_text_overlay,
                video_path,
                request.params.get("text", ""),
                request.params.get("position", "bottom"),
                request.params.get("fontsize", 50)
            )
        elif request.operation == "add_music":
            result_url = await asyncio.to_thread(
                processor.add_background_music,
                video_path,
                request.params.get("music_path", "")
            )
        elif request.operation == "add_effects":
            result_url = await asyncio.to_thread(
                processor.add_video_effects,
                video_path,
                request.params.get("effects", [])
            )
        else:
            raise HTTPException(status_code=400, detail="Unknown operation")

        return {"url": result_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-caption")
async def generate_caption(request: CaptionRequest):
    session = tasks.get(request.video_id)
    if not session or "video_path" not in session:
        raise HTTPException(status_code=404, detail="Video not found")

    video_path = session["video_path"]

    try:
        from processing import processor
        caption = await asyncio.to_thread(processor.generate_caption, video_path)
        return {"caption": caption}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-title-description")
async def generate_title_description(request: TitleDescriptionRequest):
    session = tasks.get(request.video_id)
    if not session or "video_path" not in session:
        raise HTTPException(status_code=404, detail="Video not found")

    video_path = session["video_path"]

    try:
        from processing import processor
        result = await asyncio.to_thread(
            processor.generate_title_and_description,
            video_path,
            request.segment_start,
            request.segment_end,
            request.platform
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-thumbnail")
async def generate_thumbnail(request: ThumbnailRequest):
    session = tasks.get(request.video_id)
    if not session or "video_path" not in session:
        raise HTTPException(status_code=404, detail="Video not found")

    video_path = session["video_path"]

    try:
        from processing import processor
        thumbnail_url = await asyncio.to_thread(
            processor.generate_thumbnail,
            video_path,
            request.timestamp
        )
        return {"thumbnail_url": thumbnail_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/suggest-hashtags")
async def suggest_hashtags(request: CaptionRequest):
    session = tasks.get(request.video_id)
    if not session or "video_path" not in session:
        raise HTTPException(status_code=404, detail="Video not found")

    video_path = session["video_path"]

    try:
        from processing import processor
        hashtags = await asyncio.to_thread(processor.suggest_hashtags, video_path)
        return {"hashtags": hashtags}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload-to-social")
async def upload_to_social(request: SocialMediaRequest):
    try:
        from processing import processor
        result = await asyncio.to_thread(
            processor.upload_to_social_media,
            request.video_path,
            request.platform,
            request.title,
            request.hashtags
        )
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-subtitles")
async def generate_subtitles(request: SubtitleRequest):
    session = tasks.get(request.video_id)
    if not session or "video_path" not in session:
        raise HTTPException(status_code=404, detail="Video not found")

    video_path = session["video_path"]

    try:
        from processing import processor
        srt_url = await asyncio.to_thread(processor.generate_subtitles, video_path, request.language)
        if srt_url:
            return {"srt_url": srt_url}
        else:
            raise HTTPException(status_code=500, detail="Failed to generate subtitles")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/music-library")
async def get_music_library():
    """Get available background music files"""
    music_dir = os.path.join(STATIC_DIR, "music")
    if not os.path.exists(music_dir):
        return {"tracks": []}

    tracks = []
    for file in os.listdir(music_dir):
        if file.endswith(('.mp3', '.wav', '.m4a')):
            tracks.append({
                "name": file,
                "url": f"/static/music/{file}"
            })
    return {"tracks": tracks}

@app.get("/api/status/{task_id}")
async def get_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks[task_id]

@app.get("/api/processed-videos")
async def list_processed_videos():
    if not os.path.exists(PROCESSED_DIR):
        return {"videos": []}
    videos = []
    for file in os.listdir(PROCESSED_DIR):
        if file.endswith('.mp4'):
            videos.append({
                "filename": file,
                "url": f"/download/{file}",
                "size": os.path.getsize(os.path.join(PROCESSED_DIR, file))
            })
    return {"videos": videos}

@app.get("/download/{filename}")
async def download_clip(filename: str):
    file_path = os.path.join(PROCESSED_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type='video/mp4', headers={"Content-Disposition": f"attachment; filename={filename}"})

@app.get("/api/preview/{video_id}")
async def get_video_preview(video_id: str, start: float = 0, duration: float = 10):
    """Stream a video preview segment"""
    # Find the task that holds the video path
    session = tasks.get(video_id)
    if not session or "video_path" not in session:
        raise HTTPException(status_code=404, detail="Video not found")

    video_path = session["video_path"]
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found")

    # Create a temporary preview clip
    try:
        from processing import processor
        temp_clip_path = await asyncio.to_thread(
            processor.create_preview_clip,
            video_path,
            start,
            start + duration
        )
        return FileResponse(temp_clip_path, media_type='video/mp4')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
