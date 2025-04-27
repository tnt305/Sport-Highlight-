import ffmpeg
import math
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import yaml

from src.preprocessing.video.utils import list_end_directories
from src.preprocessing import load_config
from src.task.register import task, register_task

def generate_time_pairs(duration, start_minute=0, min_duration=10, audio_pre=10, audio_post=20, seconds_per_segment=60):
    """Tạo danh sách cặp (start, end) cho video và audio, bỏ qua đoạn video dưới min_duration."""
    pairs = []
    current = start_minute
    while current * seconds_per_segment < duration:
        video_start = current * seconds_per_segment
        video_end = min((current + 1) * seconds_per_segment, duration)
        video_length = video_end - video_start
        if video_length >= min_duration:
            audio_start = max(0, video_start - audio_pre)
            audio_end = min(duration, video_end + audio_post)
            pairs.append({
                'video': (video_start, video_end),
                'audio': (audio_start, audio_end)
            })
        current += 1
    return pairs

def video2frames(video: str, save_dir: str, frame_interval: int):
    """Trích xuất khung hình từ video mỗi frame_interval giây."""
    try:
        if not os.path.exists(video):
            print(f"Video file {video} không tồn tại, bỏ qua trích xuất khung hình")
            return False
        os.makedirs(save_dir, exist_ok=True)
        (
            ffmpeg
            .input(video)
            .output(f"{save_dir}/frame_%04d.png", vf=f"fps=1/{frame_interval}")
            .run(quiet=True, overwrite_output=True)
        )
        return True
    except ffmpeg.Error as e:
        error_msg = e.stderr.decode('utf-8') if e.stderr else "Không có chi tiết lỗi"
        print(f"Lỗi khi trích xuất khung hình từ {video}: {error_msg}")
        return False

def cut_segment(input_video, start, length, output_file, is_audio=False):
    """Cắt video hoặc âm thanh sử dụng FFmpeg Python chainable API."""
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        probe = ffmpeg.probe(input_video)
        duration = float(probe['format']['duration'])
        if start < 0 or start + length > duration:
            return False
        if is_audio:
            has_audio = any(stream['codec_type'] == 'audio' for stream in probe['streams'])
            if not has_audio:
                print(f"File {input_video} không có luồng âm thanh, bỏ qua: {output_file}")
                return False
        stream = ffmpeg.input(input_video, ss=start, t=length)
        if is_audio:
            stream = ffmpeg.output(stream, output_file, acodec="libmp3lame", vn=True, **{'q:a': '5'}, format="mp3")
        else:
            stream = ffmpeg.output(stream, output_file, c="copy", loglevel="quiet")
        stream.run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
        return True
    except ffmpeg.Error as e:
        error_msg = e.stderr.decode('utf-8') if e.stderr else "Không có chi tiết lỗi"
        print(f"Lỗi với {output_file}: {error_msg}")
        return False

def process_chunk(chunk_data):
    """Xử lý một chunk: cắt video, trích xuất khung hình, cắt âm thanh."""
    input_video, chunk_index, video_start, video_end, audio_start, audio_end, output_dir, frame_interval, save_dir_prefix = chunk_data
    chunk_dir = os.path.join(output_dir, f"{save_dir_prefix}_{chunk_index}")
    os.makedirs(chunk_dir, exist_ok=True)
    
    video_file = os.path.join(chunk_dir, f"{save_dir_prefix}_{chunk_index}.mp4")
    video_success = cut_segment(input_video, video_start, video_end - video_start, video_file, is_audio=False)
    
    frames_dir = os.path.join(chunk_dir, "frames")
    if video_success:
        video2frames(video_file, frames_dir, frame_interval)
    else:
        print(f"Bỏ qua trích xuất khung hình cho {save_dir_prefix}_{chunk_index} vì cắt video thất bại")
    
    audio_file = os.path.join(chunk_dir, f"{save_dir_prefix}_{chunk_index}.mp3")
    cut_segment(input_video, audio_start, audio_end - audio_start, audio_file, is_audio=True)

def cut_video_and_audio(input_video, output_dir, chunk_offset=0, frame_interval=1, min_duration=10, audio_pre=10, audio_post=20, seconds_per_segment=60, save_dir_prefix="chunks"):
    """Cắt video, âm thanh và trích xuất khung hình, lưu trong chunks/chunk_i."""
    if not os.path.exists(input_video):
        print(f"File {input_video} không tồn tại!")
        return 0
    os.makedirs(output_dir, exist_ok=True)
    try:
        probe = ffmpeg.probe(input_video)
        duration = float(probe['format']['duration'])
    except ffmpeg.Error as e:
        print(f"Lỗi khi đọc file {input_video}: {e.stderr.decode('utf-8')}")
        return 0
    time_pairs = generate_time_pairs(
        duration,
        min_duration=min_duration,
        audio_pre=audio_pre,
        audio_post=audio_post,
        seconds_per_segment=seconds_per_segment
    )
    
    chunk_tasks = []
    for i, pair in enumerate(time_pairs):
        chunk_index = i + chunk_offset
        video_start, video_end = pair['video']
        audio_start, audio_end = pair['audio']
        chunk_tasks.append((
            input_video, chunk_index, video_start, video_end, audio_start, audio_end, output_dir, frame_interval, save_dir_prefix
        ))
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        executor.map(process_chunk, chunk_tasks)
    
    return len(time_pairs)

@register_task("process_video")
def process_videos(**config_overrides):
    """Xử lý tất cả video dựa trên cấu hình YAML, hỗ trợ ghi đè tham số."""
    config = load_config(**config_overrides)
    
    input_base_dir = config["input_base_dir"]
    video_names = config["video_names"]
    annotation_files = config["annotation_files"]
    processing_params = config["processing_params"]
    storage_params = config["storage_params"]
    
    frame_interval = processing_params["frame_interval"]
    min_duration = processing_params["min_duration"]
    audio_pre = processing_params["audio_pre"]
    audio_post = processing_params["audio_post"]
    seconds_per_segment = processing_params["seconds_per_segment"]
    output_base_dir = storage_params["output_base_dir"]
    save_dir_prefix = storage_params["save_dir_prefix"]
    
    game_paths = list_end_directories(input_base_dir)
    
    for game_path in tqdm(game_paths, desc="Xử lý các trận đấu"):
        game_name = os.path.basename(game_path)
        output_dir = os.path.join(output_base_dir, game_name)
        
        chunk_offset = 0
        for video_name in video_names:
            video_path = os.path.join(game_path, video_name)
            if not os.path.exists(video_path):
                print(f"Video {video_path} không tồn tại, bỏ qua.")
                continue
            
            num_segments = cut_video_and_audio(
                input_video=video_path,
                output_dir=output_dir,
                chunk_offset=chunk_offset,
                frame_interval=frame_interval,
                min_duration=min_duration,
                audio_pre=audio_pre,
                audio_post=audio_post,
                seconds_per_segment=seconds_per_segment,
                save_dir_prefix=save_dir_prefix
            )
            chunk_offset += num_segments
            print(f"Đã xử lý {video_path}: {num_segments} đoạn")
        
        for filename in annotation_files:
            src_path = os.path.join(game_path, filename)
            dst_path = os.path.join(output_dir, filename)
            if os.path.exists(src_path):
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.copy2(src_path, dst_path)
                print(f"Đã copy {filename} vào {dst_path}")
            else:
                print(f"{filename} không tồn tại trong {game_path}")

if __name__ == "__main__":
    # Chạy task process_video với overwrite seconds_per_segment
    task("process_video", processing_params={"seconds_per_segment": 45})

