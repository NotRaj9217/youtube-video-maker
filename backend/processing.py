import os
import yt_dlp
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import asyncio
import uuid
import random
try:
    import speech_recognition as sr
    from textblob import TextBlob
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False
import subprocess
import re
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
import tempfile

import imageio_ffmpeg

class VideoProcessor:
    def __init__(self, download_dir="downloads", processed_dir="processed"):
        self.download_dir = download_dir
        self.processed_dir = processed_dir
        os.makedirs(download_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)
        self.ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        print(f"Using FFmpeg at: {self.ffmpeg_path}")

    def download_video(self, url):
        """Downloads video using yt-dlp"""
        # Limit to 1080p to save bandwidth/processing and ensure MP4 compatibility
        ydl_opts = {
            'format': 'bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': os.path.join(self.download_dir, '%(id)s.%(ext)s'),
            'noplaylist': True,
            'ffmpeg_location': self.ffmpeg_path,
            'merge_output_format': 'mp4',  # Ensure final merge is MP4
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_path = ydl.prepare_filename(info)
            return video_path, info

    import subprocess
    import re

    def get_silence_points(self, video_path, threshold="-30dB", duration=0.5):
        """
        Runs ffmpeg silencedetect to find silence timestamps.
        Returns a list of [start, end] tuples.
        """
        cmd = [
            self.ffmpeg_path,
            "-i", video_path,
            "-af", f"silencedetect=noise={threshold}:d={duration}",
            "-f", "null",
            "-"
        ]
        
        try:
            # silencedetect writes to stderr
            result = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True)
            output = result.stderr
            
            # Parse output
            # [silencedetect @ ...] silence_start: 462.394
            # [silencedetect @ ...] silence_end: 464.551 | silence_duration: 2.156
            
            starts = []
            ends = []
            
            for line in output.split('\n'):
                if "silence_start" in line:
                    match = re.search(r'silence_start: (\d+(\.\d+)?)', line)
                    if match:
                        starts.append(float(match.group(1)))
                elif "silence_end" in line:
                    match = re.search(r'silence_end: (\d+(\.\d+)?)', line)
                    if match:
                        ends.append(float(match.group(1)))
            
            # Pair them up nicely
            silences = []
            for s in starts:
                # Find first end after start
                valid_ends = [e for e in ends if e > s]
                if valid_ends:
                    silences.append((s, valid_ends[0]))
            
            return silences
            
        except Exception as e:
            print(f"Error checking silence: {e}")
            return []

    def extract_audio_segment(self, video_path, start, end):
        """Extract audio segment as WAV for speech recognition"""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name

        cmd = [
            self.ffmpeg_path, "-y", "-i", video_path,
            "-ss", str(start), "-t", str(end - start),
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            temp_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return temp_path

    def transcribe_and_analyze(self, audio_path):
        """Transcribe audio and analyze sentiment"""
        if not SPEECH_AVAILABLE:
            return "", 0  # Fallback when speech recognition not available

        try:
            recognizer = sr.Recognizer()
            with sr.AudioFile(audio_path) as source:
                audio = recognizer.record(source)
                text = recognizer.recognize_google(audio)
                blob = TextBlob(text)
                sentiment = blob.sentiment.polarity  # -1 to 1
                return text, sentiment
        except Exception as e:
            print(f"Speech recognition error: {e}")
            return "", 0

    def analyze_viral_segments(self, video_path, duration, duration_mode="30"):
        """
        Real AI-powered Analysis using Speech Recognition and Sentiment Analysis
        """
        clips = []

        # Determine duration config
        if duration_mode == "60":
            target_dur = 60
        elif duration_mode == "120":
            target_dur = 120
        elif duration_mode.isdigit() and int(duration_mode) > 0:
             target_dur = int(duration_mode)
        else: # Default 30s
            target_dur = 30

        if duration < target_dur:
            return [{"start": 0, "end": duration, "title": "Full Video Segment", "duration": f"{int(duration)}s", "sentiment": 0}]

        # Sample segments across the video
        segment_length = 10  # Analyze 10s chunks for efficiency
        step = max(segment_length, duration / 20)  # Sample at least 20 points or every segment_length

        segments_data = []
        current_time = 0
        while current_time < duration - segment_length:
            audio_path = self.extract_audio_segment(video_path, current_time, min(current_time + segment_length, duration))
            text, sentiment = self.transcribe_and_analyze(audio_path)
            os.unlink(audio_path)  # Clean up

            segments_data.append({
                "start": current_time,
                "end": min(current_time + segment_length, duration),
                "sentiment": sentiment,
                "text": text
            })
            current_time += step

        # Find top segments with highest sentiment (dynamic count based on video length)
        if segments_data:
            # Sort by sentiment descending
            segments_data.sort(key=lambda x: x["sentiment"], reverse=True)

            # Dynamic clip count based on video length and content quality
            # Shorter videos get fewer clips, longer videos get more
            # Videos with more engaging content get more clips
            base_clip_count = min(5, max(1, int(duration / 60)))  # 1 clip per minute, max 5

            # If we have segments with positive sentiment, prioritize them
            positive_segments = [s for s in segments_data if s["sentiment"] > 0.1]
            if positive_segments:
                top_segments = positive_segments[:base_clip_count]
            else:
                # If no positive sentiment, still provide clips but mark as random
                top_segments = segments_data[:base_clip_count]

            for i, seg in enumerate(top_segments):
                # Extend segment to target duration while maintaining sentiment
                start = max(0, seg["start"] - (target_dur - segment_length) / 2)
                end = min(duration, start + target_dur)

                # If not enough room, adjust
                if end - start < target_dur:
                    if start == 0:
                        end = min(duration, start + target_dur)
                    else:
                        start = max(0, end - target_dur)

                clips.append({
                    "start": float(f"{start:.2f}"),
                    "end": float(f"{end:.2f}"),
                    "title": f"Engaging Segment {i+1} (Sentiment: {seg['sentiment']:.2f})",
                    "duration": f"{int(end - start)}s",
                    "sentiment": seg["sentiment"]
                })

        # Fallback to random if no good segments found
        if not clips:
            for i in range(3):
                start = random.uniform(0, duration - target_dur)
                end = start + target_dur
                clips.append({
                    "start": float(f"{start:.2f}"),
                    "end": float(f"{end:.2f}"),
                    "title": f"Random Segment {i+1}",
                    "duration": f"{target_dur}s",
                    "sentiment": 0
                })

        return clips

    def create_clips(self, video_path, segments):
        """Cuts the video based on segments"""
        generated_clips = []
        base_name = os.path.basename(video_path).split('.')[0]
        
        for i, seg in enumerate(segments):
            out_name = f"{base_name}_clip_{i+1}.mp4"
            out_path = os.path.join(self.processed_dir, out_name)
            
            # Use ffmpeg_extract_subclip for speed (no re-encoding if possible)
            # Or VideoFileClip for more precision
            ffmpeg_extract_subclip(video_path, seg['start'], seg['end'], outputfile=out_path)
            
            generated_clips.append({
                "id": str(uuid.uuid4()),
                "url": f"/static/processed/{out_name}", # Serve from static
                "title": seg['title']
            })
            
        return generated_clips

    def create_single_clip(self, video_path, start, end, quality="1080", platform=None):
        """
        Cuts a single clip and optionally resizes it for specific platforms.
        """
        print(f"Creating clip: video_path={video_path}, start={start}, end={end}, quality={quality}, platform={platform}")

        # Check if video file exists
        if not os.path.exists(video_path):
            raise Exception(f"Video file does not exist: {video_path}")

        # Validate parameters
        if start >= end:
            raise Exception(f"Invalid time range: start ({start}) >= end ({end})")
        if start < 0:
            raise Exception(f"Invalid start time: {start} < 0")
        if end <= 0:
            raise Exception(f"Invalid end time: {end} <= 0")

        # Platform-specific formatting
        platform_formats = {
            "tiktok": {"width": 1080, "height": 1920, "duration": 180},  # 9:16 aspect ratio
            "instagram": {"width": 1080, "height": 1920, "duration": 90},  # Stories/Reels
            "twitter": {"width": 1920, "height": 1080, "duration": 140},  # 16:9 aspect ratio
            "youtube": {"width": 1920, "height": 1080, "duration": 60},  # Shorts
        }

        base_name = os.path.basename(video_path).split('.')[0]
        # output filename includes params to be unique
        out_name = f"{base_name}_cut_{int(start)}_{int(end)}_{quality}p.mp4"
        if platform:
            out_name = f"{base_name}_{platform}_{int(start)}_{int(end)}.mp4"
        out_path = os.path.join(self.processed_dir, out_name)

        print(f"Output path: {out_path}")
        print(f"Processed dir exists: {os.path.exists(self.processed_dir)}")

        if os.path.exists(out_path):
            print(f"Clip already exists: {out_path}")
            return f"/static/processed/{out_name}"

        # Build ffmpeg command via moviepy or directly calls
        # We use moviepy's ffmpeg_tools or just raw ffmpeg wrapper if available
        # But VideoFileClip.resize is heavy.

        # Using simple ffmpeg command wrapper for speed and resizing
        cmd = [
            self.ffmpeg_path,
            "-y", # overwrite
            "-ss", str(start),
            "-i", video_path,
            "-t", str(end - start),
            "-c:v", "libx264",
            "-c:a", "aac",
            "-strict", "experimental"
        ]

        # Add platform-specific scaling and formatting
        if platform and platform in platform_formats:
            fmt = platform_formats[platform]
            # Ensure duration doesn't exceed platform limit
            actual_duration = min(end - start, fmt["duration"])
            cmd[6] = str(actual_duration)  # Update duration

            # Add scaling filter
            scale_filter = f"scale={fmt['width']}:{fmt['height']}:force_original_aspect_ratio=decrease,pad={fmt['width']}:{fmt['height']}:(ow-iw)/2:(oh-ih)/2"
            cmd.extend(["-vf", scale_filter])
        elif quality and quality.isdigit():
            h = int(quality)
            if h < 1080: # Only downscale
                cmd.extend(["-vf", f"scale=-2:{h}"])

        cmd.append(out_path)

        import subprocess
        print(f"Running FFmpeg: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            print(f"FFmpeg completed successfully for {out_name}")
            return f"/static/processed/{out_name}"
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg failed: {e}")
            print(f"FFmpeg stderr: {e.stderr}")
            raise Exception(f"Video processing failed: {e.stderr}")

    def create_preview_clip(self, video_path, start, end):
        """Create a temporary preview clip for streaming"""
        temp_dir = os.path.join(self.processed_dir, "temp")
        os.makedirs(temp_dir, exist_ok=True)

        base_name = os.path.basename(video_path).split('.')[0]
        temp_name = f"{base_name}_preview_{int(start)}_{int(end)}.mp4"
        temp_path = os.path.join(temp_dir, temp_name)

        if os.path.exists(temp_path):
            return temp_path

        # Create a short preview clip (max 10s, lower quality for speed)
        preview_duration = min(10, end - start)

        cmd = [
            self.ffmpeg_path, "-y",
            "-ss", str(start),
            "-i", video_path,
            "-t", str(preview_duration),
            "-c:v", "libx264",
            "-c:a", "aac",
            "-vf", "scale=-2:480",  # Lower quality for preview
            "-preset", "fast",
            temp_path
        ]

        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return temp_path

    def trim_video(self, video_path, start, end):
        """Trim video to specified start and end times"""
        base_name = os.path.basename(video_path).split('.')[0]
        out_name = f"{base_name}_trim_{int(start)}_{int(end)}.mp4"
        out_path = os.path.join(self.processed_dir, out_name)

        if os.path.exists(out_path):
            return f"/static/processed/{out_name}"

        cmd = [
            self.ffmpeg_path, "-y", "-i", video_path,
            "-ss", str(start), "-t", str(end - start),
            "-c", "copy", out_path
        ]

        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return f"/static/processed/{out_name}"

    def merge_videos(self, video_paths):
        """Merge multiple videos into one"""
        if not video_paths:
            raise ValueError("No videos to merge")

        # Create a temporary file list for ffmpeg
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for path in video_paths:
                f.write(f"file '{path}'\n")
            list_file = f.name

        out_name = f"merged_{uuid.uuid4().hex[:8]}.mp4"
        out_path = os.path.join(self.processed_dir, out_name)

        cmd = [
            self.ffmpeg_path, "-y", "-f", "concat", "-safe", "0",
            "-i", list_file, "-c", "copy", out_path
        ]

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        finally:
            os.unlink(list_file)

        return f"/static/processed/{out_name}"

    def add_text_overlay(self, video_path, text, position="bottom", fontsize=50):
        """Add text overlay to video"""
        base_name = os.path.basename(video_path).split('.')[0]
        out_name = f"{base_name}_text_{uuid.uuid4().hex[:8]}.mp4"
        out_path = os.path.join(self.processed_dir, out_name)

        # Position mapping
        positions = {
            "top": "x=(w-text_w)/2:y=50",
            "bottom": "x=(w-text_w)/2:y=h-th-50",
            "center": "x=(w-text_w)/2:y=(h-text_h)/2"
        }

        filter_pos = positions.get(position, positions["bottom"])

        cmd = [
            self.ffmpeg_path, "-y", "-i", video_path,
            "-vf", f"drawtext=text='{text}':fontsize={fontsize}:{filter_pos}:fontcolor=white:box=1:boxcolor=black@0.5",
            "-c:a", "copy", out_path
        ]

        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return f"/static/processed/{out_name}"

    def add_background_music(self, video_path, music_path):
        """Add background music to video"""
        base_name = os.path.basename(video_path).split('.')[0]
        out_name = f"{base_name}_music_{uuid.uuid4().hex[:8]}.mp4"
        out_path = os.path.join(self.processed_dir, out_name)

        cmd = [
            self.ffmpeg_path, "-y", "-i", video_path, "-i", music_path,
            "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=first[aout]",
            "-map", "0:v", "-map", "[aout]", "-c:v", "copy", out_path
        ]

        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return f"/static/processed/{out_name}"

    def add_video_effects(self, video_path, effects):
        """Add video effects like filters, transitions"""
        base_name = os.path.basename(video_path).split('.')[0]
        out_name = f"{base_name}_effects_{uuid.uuid4().hex[:8]}.mp4"
        out_path = os.path.join(self.processed_dir, out_name)

        filters = []
        for effect in effects:
            if effect == "grayscale":
                filters.append("colorchannelmixer=.3:.4:.3:0:.3:.4:.3:0:.3:.4:.3")
            elif effect == "sepia":
                filters.append("colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131")
            elif effect == "blur":
                filters.append("boxblur=5:5")
            elif effect == "vintage":
                filters.append("curves=vintage")

        filter_chain = ",".join(filters) if filters else "copy"

        cmd = [
            self.ffmpeg_path, "-y", "-i", video_path,
            "-vf", filter_chain, "-c:a", "copy", out_path
        ]

        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return f"/static/processed/{out_name}"

    def extract_visual_content(self, video_path, timestamp=0, num_frames=3):
        """Extract text and visual features from video frames"""
        try:
            import cv2
            import pytesseract
            from PIL import Image
            import numpy as np

            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)

            visual_text = []
            visual_features = []

            for i in range(num_frames):
                ret, frame = cap.read()
                if not ret:
                    break

                # Extract text using OCR
                try:
                    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    text = pytesseract.image_to_string(pil_image).strip()
                    if text:
                        visual_text.append(text)
                except Exception as e:
                    print(f"OCR error: {e}")

                # Basic visual analysis (could be enhanced with ML models)
                # Check for text overlays, faces, objects, etc.
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Simple edge detection to gauge visual complexity
                edges = cv2.Canny(gray, 100, 200)
                edge_density = np.sum(edges > 0) / edges.size

                visual_features.append({
                    "edge_density": edge_density,
                    "brightness": np.mean(gray),
                    "contrast": gray.std()
                })

                # Move to next frame (every 2 seconds for variety)
                cap.set(cv2.CAP_PROP_POS_MSEC, (timestamp + i * 2) * 1000)

            cap.release()

            return {
                "text": " ".join(visual_text),
                "features": visual_features
            }
        except ImportError:
            print("Visual analysis dependencies not available")
            return {"text": "", "features": []}
        except Exception as e:
            print(f"Visual analysis error: {e}")
            return {"text": "", "features": []}

    def analyze_content_with_ai(self, transcript, visual_text, duration, platform="general"):
        """Use AI to deeply analyze content and extract insights"""
        try:
            import openai
            client = openai.OpenAI()

            # Combine audio and visual text
            combined_text = f"{transcript} {visual_text}".strip()

            # Content analysis prompt
            analysis_prompt = f"""
            Analyze this video content and extract key information:

            Transcript: {combined_text[:500]}
            Duration: {duration} seconds
            Platform: {platform}

            Provide a JSON response with:
            - main_topics: array of 3-5 key topics/subjects
            - keywords: array of 5-8 important keywords
            - content_type: (educational, entertainment, review, tutorial, vlog, etc.)
            - tone: (serious, humorous, inspirational, informative, etc.)
            - target_audience: (general, tech-savvy, beginners, professionals, etc.)
            """

            analysis_response = client.chat.completions.create(
                model="gpt-4",
                messages=[{
                    "role": "user",
                    "content": analysis_prompt
                }],
                max_tokens=200,
                temperature=0.3
            )

            # Parse the analysis
            try:
                import json
                analysis = json.loads(analysis_response.choices[0].message.content.strip())
            except:
                # Fallback if JSON parsing fails
                analysis = {
                    "main_topics": ["video content"],
                    "keywords": ["video", "content"],
                    "content_type": "general",
                    "tone": "informative",
                    "target_audience": "general"
                }

            return analysis

        except Exception as e:
            print(f"Content analysis error: {e}")
            return {
                "main_topics": ["video content"],
                "keywords": ["video", "content"],
                "content_type": "general",
                "tone": "informative",
                "target_audience": "general"
            }

    def generate_title_and_description(self, video_path, segment_start=0, segment_end=30, platform="youtube"):
        """Generate AI-powered title and description with enhanced content understanding"""
        print(f"Generating enhanced content for segment {segment_start}-{segment_end}, platform: {platform}")

        duration = segment_end - segment_start

        # 1. Enhanced transcription using Whisper
        transcript = ""
        sentiment = 0

        try:
            import whisper
            model = whisper.load_model("base")
            # Extract audio for the segment
            audio_path = self.extract_audio_segment(video_path, segment_start, segment_end)
            result = model.transcribe(audio_path)
            transcript = result["text"].strip()
            print(f"Whisper transcript: '{transcript[:200]}...'")

            # Simple sentiment analysis
            if transcript:
                try:
                    from textblob import TextBlob
                    blob = TextBlob(transcript)
                    sentiment = blob.sentiment.polarity
                except:
                    sentiment = 0

            try:
                os.unlink(audio_path)
            except:
                pass

        except Exception as e:
            print(f"Whisper transcription failed: {e}")
            # Fallback to basic speech recognition
            try:
                audio_path = self.extract_audio_segment(video_path, segment_start, min(segment_end, segment_start + 30))
                transcript, sentiment = self.transcribe_and_analyze(audio_path)
                try:
                    os.unlink(audio_path)
                except:
                    pass
            except Exception as e2:
                print(f"Fallback transcription also failed: {e2}")

        # 2. Visual content analysis
        visual_data = self.extract_visual_content(video_path, segment_start)
        visual_text = visual_data.get("text", "")
        if visual_text:
            print(f"Visual text detected: '{visual_text[:100]}...'")

        # 3. AI-powered content analysis
        content_analysis = self.analyze_content_with_ai(transcript, visual_text, duration, platform)

        # 4. Generate title and description with enhanced context
        try:
            import openai
            client = openai.OpenAI()

            # Platform-specific title generation
            platform_styles = {
                "youtube": "YouTube video title - SEO optimized, clickbait but honest, includes keywords",
                "tiktok": "TikTok video title - short, trendy, emoji-friendly, viral potential",
                "instagram": "Instagram Reel title - engaging, hashtag-ready, social-focused",
                "twitter": "Twitter/X video title - concise, shareable, conversation-starting"
            }

            style_guide = platform_styles.get(platform, "general social media video title")

            # Enhanced title generation
            title_prompt = f"""
            Create an optimized title for a {duration:.1f}-second video clip.

            Content Analysis:
            - Main topics: {', '.join(content_analysis.get('main_topics', []))}
            - Keywords: {', '.join(content_analysis.get('keywords', []))}
            - Content type: {content_analysis.get('content_type', 'general')}
            - Tone: {content_analysis.get('tone', 'informative')}
            - Target audience: {content_analysis.get('target_audience', 'general')}

            Transcript excerpt: "{transcript[:300]}..."
            Visual text: "{visual_text[:100]}..."

            Style: {style_guide}
            Requirements:
            - Max 60 characters
            - Engaging and click-worthy
            - Include 1-2 relevant keywords naturally
            - Match the content's tone and style
            """

            title_response = client.chat.completions.create(
                model="gpt-4",
                messages=[{
                    "role": "user",
                    "content": title_prompt
                }],
                max_tokens=40,
                temperature=0.8
            )

            # Enhanced description generation
            desc_prompt = f"""
            Create a compelling description for a {duration:.1f}-second video clip.

            Content Analysis:
            - Main topics: {', '.join(content_analysis.get('main_topics', []))}
            - Keywords: {', '.join(content_analysis.get('keywords', []))}
            - Content type: {content_analysis.get('content_type', 'general')}
            - Tone: {content_analysis.get('tone', 'informative')}

            Transcript excerpt: "{transcript[:400]}..."
            Visual text: "{visual_text[:100]}..."

            Style: Optimized for {platform} - engaging, informative, includes relevant hashtags
            Requirements:
            - Max 150 characters
            - Highlight key value or insights
            - Include 3-5 relevant hashtags
            - Encourage engagement (likes, comments, shares)
            """

            desc_response = client.chat.completions.create(
                model="gpt-4",
                messages=[{
                    "role": "user",
                    "content": desc_prompt
                }],
                max_tokens=60,
                temperature=0.7
            )

            title = title_response.choices[0].message.content.strip()
            description = desc_response.choices[0].message.content.strip()

            # Ensure title length limit
            if len(title) > 60:
                title = title[:57] + "..."

            # Ensure description length limit
            if len(description) > 150:
                description = description[:147] + "..."

            return {
                "title": title,
                "description": description,
                "sentiment_score": sentiment,
                "content_analysis": content_analysis,
                "transcript_length": len(transcript),
                "visual_text_detected": bool(visual_text)
            }

        except Exception as e:
            print(f"AI generation error: {e}")
            # Enhanced fallback based on content analysis
            fallback_title = self.generate_fallback_title(content_analysis, duration, platform)
            fallback_desc = self.generate_fallback_description(content_analysis, duration, platform)

            return {
                "title": fallback_title,
                "description": fallback_desc,
                "sentiment_score": sentiment,
                "content_analysis": content_analysis,
                "transcript_length": len(transcript),
                "visual_text_detected": bool(visual_text)
            }

    def generate_fallback_title(self, analysis, duration, platform):
        """Generate fallback title based on content analysis"""
        topics = analysis.get('main_topics', ['content'])
        content_type = analysis.get('content_type', 'general')
        tone = analysis.get('tone', 'informative')

        base_title = f"{topics[0].title()} {content_type.title()}"

        if duration < 15:
            base_title = f"Quick {base_title}"
        elif duration > 60:
            base_title = f"Complete {base_title}"

        # Platform-specific adjustments
        if platform == "tiktok":
            base_title += " 🎥"
        elif platform == "instagram":
            base_title += " 📱"

        return base_title[:60]

    def generate_fallback_description(self, analysis, duration, platform):
        """Generate fallback description based on content analysis"""
        topics = analysis.get('main_topics', ['content'])
        keywords = analysis.get('keywords', ['video'])

        desc = f"Check out this {analysis.get('content_type', 'engaging')} video about {', '.join(topics[:2])}!"

        # Add hashtags
        hashtags = ['#' + keyword.replace(' ', '') for keyword in keywords[:3]]
        desc += f" {' '.join(hashtags)}"

        # Platform-specific encouragement
        if platform == "youtube":
            desc += " Like & subscribe for more!"
        elif platform == "tiktok":
            desc += " Follow for daily content! 🔥"
        elif platform == "instagram":
            desc += " Double tap if you agree! 💯"

        return desc[:150]

    def generate_caption(self, video_path):
        """Generate AI caption using transcription"""
        # Extract audio and transcribe
        audio_path = self.extract_audio_segment(video_path, 0, 30)  # First 30 seconds
        try:
            text, _ = self.transcribe_and_analyze(audio_path)
            os.unlink(audio_path)

            if not text:
                return "No speech detected in the video."

            # Use OpenAI to generate a catchy caption
            import openai
            client = openai.OpenAI()  # Assumes API key is set in environment

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{
                    "role": "user",
                    "content": f"Generate a catchy, engaging caption for a video with this transcript: '{text}'. Make it suitable for social media."
                }],
                max_tokens=100
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Caption generation error: {e}")
            return "Could not generate caption. Please try again."

    def generate_thumbnail(self, video_path, timestamp=0):
        """Generate thumbnail from video frame"""
        base_name = os.path.basename(video_path).split('.')[0]
        thumb_name = f"{base_name}_thumb_{int(timestamp)}.jpg"
        thumb_path = os.path.join(self.processed_dir, thumb_name)

        cmd = [
            self.ffmpeg_path, "-y", "-i", video_path,
            "-ss", str(timestamp), "-vframes", "1",
            "-q:v", "2", thumb_path
        ]

        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return f"/static/processed/{thumb_name}"

    def suggest_hashtags(self, video_path):
        """Suggest hashtags based on video content"""
        # Transcribe and analyze content
        audio_path = self.extract_audio_segment(video_path, 0, 30)
        try:
            text, _ = self.transcribe_and_analyze(audio_path)
            os.unlink(audio_path)

            if not text:
                return ["#video", "#content"]

            # Use OpenAI to suggest hashtags
            import openai
            client = openai.OpenAI()

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{
                    "role": "user",
                    "content": f"Suggest 5-10 relevant hashtags for a video with this transcript: '{text}'. Return only the hashtags separated by spaces."
                }],
                max_tokens=50
            )

            hashtags_text = response.choices[0].message.content.strip()
            # Parse hashtags
            hashtags = [tag.strip() for tag in hashtags_text.replace('#', '').split() if tag.strip()]
            return ['#' + tag for tag in hashtags[:10]]
        except Exception as e:
            print(f"Hashtag suggestion error: {e}")
            return ["#video", "#viral", "#content"]

    def upload_to_social_media(self, video_path, platform, title, hashtags):
        """Upload video to social media platforms"""
        # This is a simplified implementation - in reality, you'd need API keys and proper authentication
        try:
            if platform == "tiktok":
                # TikTok API integration would go here
                return {"message": "TikTok upload not implemented yet", "url": None}
            elif platform == "instagram":
                # Instagram API integration
                return {"message": "Instagram upload not implemented yet", "url": None}
            elif platform == "twitter":
                # Twitter API integration
                import tweepy
                # This would require proper authentication
                return {"message": "Twitter upload not implemented yet", "url": None}
            else:
                raise ValueError(f"Unsupported platform: {platform}")
        except Exception as e:
            raise Exception(f"Social media upload failed: {str(e)}")

    def generate_subtitles(self, video_path, target_language="en"):
        """Generate subtitles using Whisper and translate if needed"""
        import whisper
        from googletrans import Translator
        import srt
        from datetime import timedelta

        try:
            # Load Whisper model (base model for speed)
            model = whisper.load_model("base")

            # Transcribe
            result = model.transcribe(video_path, language="en")

            # Convert to SRT format
            subtitles = []
            for segment in result["segments"]:
                start = timedelta(seconds=segment["start"])
                end = timedelta(seconds=segment["end"])
                text = segment["text"].strip()

                # Translate if needed
                if target_language != "en":
                    translator = Translator()
                    try:
                        translated = translator.translate(text, dest=target_language)
                        text = translated.text
                    except:
                        pass  # Keep original if translation fails

                subtitle = srt.Subtitle(
                    index=len(subtitles) + 1,
                    start=start,
                    end=end,
                    content=text
                )
                subtitles.append(subtitle)

            # Save SRT file
            base_name = os.path.basename(video_path).split('.')[0]
            srt_name = f"{base_name}_subtitles_{target_language}.srt"
            srt_path = os.path.join(self.processed_dir, srt_name)

            with open(srt_path, 'w', encoding='utf-8') as f:
                f.write(srt.compose(subtitles))

            return f"/static/processed/{srt_name}"
        except Exception as e:
            print(f"Subtitle generation error: {e}")
            return None


processor = VideoProcessor(download_dir="downloads", processed_dir="static/processed")
