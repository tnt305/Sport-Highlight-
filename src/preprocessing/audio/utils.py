import ffmpeg
import math
import os
from tqdm import tqdm

def video2audio(video: str, output_audio: str):
    """Extract audio from a video file."""
    try:
        (
            ffmpeg
            .input(video)
            .output(output_audio, acodec="pcm_s16le")
            .run(quiet=True, overwrite_output=False)
        )
    except ffmpeg.Error as e:
        print(f"Error occurred while extracting audio: {e.stderr.decode()}")

def create_audio_chunk(input: str, output: str, ss: int, duration: int):
    """
    Create an audio chunk.
    start_time with 0
    duration length [60s +30s, 60s + 60s, 60s + 90s]
    
    """
    try:
        (
            ffmpeg
            .input(input, ss=ss, t=duration)
            .output(output, acodec='pcm_s16le')
            .run(quiet=True, overwrite_output=True)
        )
        # print(f"Audio chunk created successfully: {output}")
    except ffmpeg.Error as e:
        print(f"Error occurred while creating audio chunk: {e.stderr.decode()}")