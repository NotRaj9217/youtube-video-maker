from moviepy.video.io.VideoFileClip import VideoFileClip

try:
    clip = VideoFileClip('downloads/H895ZcG13Hg.mp4')
    print('File is valid. Duration:', clip.duration)
except Exception as e:
    print('File is corrupt or error:', str(e))
