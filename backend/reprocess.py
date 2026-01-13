from processing import processor
from moviepy import VideoFileClip
import os

downloads = 'backend/downloads'
processed = 'backend/static/processed'

# List videos
videos = [f for f in os.listdir(downloads) if f.endswith('.mp4')]

for video in videos:
    video_path = os.path.join(downloads, video)
    print(f"Processing {video_path}")

    # Get duration
    full_clip = VideoFileClip(video_path)
    duration = full_clip.duration
    segments = processor.analyze_viral_segments(video_path, duration)
    print(f"Segments: {segments}")

    # Create clips
    base_name = os.path.basename(video_path).split('.')[0]
    for i, seg in enumerate(segments):
        out_name = f"{base_name}_clip_{i+1}.mp4"
        out_path = os.path.join(processed, out_name)

        # Delete if exists
        if os.path.exists(out_path):
            os.remove(out_path)

        # Create with re-encoding
        full_clip = VideoFileClip(video_path)
        clip = full_clip[seg['start']:seg['end']]
        clip.write_videofile(out_path, codec='libx264', audio_codec='aac')
        print(f"Created {out_path}")
