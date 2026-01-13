from processing import processor
from moviepy import VideoFileClip

video_path = 'backend/downloads/bg9CcbjBEyA.mp4'
clip = VideoFileClip(video_path)
duration = clip.duration
segments = processor.analyze_viral_segments(video_path, duration)
print("Duration:", duration)
print("Segments:", segments)

# Test creating one clip
sub = clip.subclip(segments[0]['start'], segments[0]['end'])
sub.write_videofile('backend/static/processed/test_clip2.mp4', codec='libx264', audio_codec='aac')
print("Clip created with re-encoding")
