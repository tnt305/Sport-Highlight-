import ffmpeg
import math
import os
import shutil
import threading
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import yaml
import json
from src.preprocessing.video.utils import list_end_directories
from src.preprocessing import load_config
from src.task.register import task, register_task

def get_max_game_time(annotation_file, half):
    """Lấy thời gian gameTime lớn nhất của hiệp đấu từ file Labels-v2.json."""
    try:
        with open(annotation_file, 'r') as f:
            label_data = json.load(f)
        
        max_position = -1
        max_game_time = None
        
        for item in label_data['annotations']:
            if item['gameTime'].startswith(half) and int(item['position']) > max_position:
                max_position = int(item['position'])
                max_game_time = item['gameTime']
        
        if max_game_time:
            # Chuyển đổi gameTime (ví dụ: "1 - 44:50") thành giây
            _, time_str = max_game_time.split(' - ')
            minutes, seconds = map(int, time_str.split(':'))
            return minutes + 1 #* 60 + seconds
        return None
    except Exception as e:
        print(f"Lỗi khi đọc file {annotation_file}: {str(e)}")
        return None

def generate_time_pairs(duration, start_minute=0, min_duration=10, audio_pre=10, audio_post=20, seconds_per_segment=60, max_segments=None):
    """Tạo danh sách cặp (start, end) cho video và audio, giới hạn số segment."""
    pairs = []
    current = start_minute
    segment_count = 0
    
    while current * seconds_per_segment < duration:
        if max_segments is not None and segment_count >= max_segments:
            break
        
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
            segment_count += 1
        
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
            stream = ffmpeg.output(stream, output_file, vn=True, **{'q:a': '5'}, format="mp3") #acodec="libmp3lame"
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

def cut_video_and_audio(input_video, output_dir, annotation_file=None, chunk_offset=0, frame_interval=1, min_duration=10, audio_pre=10, audio_post=20, seconds_per_segment=60, save_dir_prefix="chunks"):
    """Cắt video, âm thanh và trích xuất khung hình, lưu trong chunks/chunk_i, giới hạn segment theo Labels-v2.json."""
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
    
    # Xác định số segment tối đa dựa trên hiệp đấu
    max_segments = None
    if annotation_file and os.path.exists(annotation_file):
        # Xác định hiệp đấu dựa trên video_name (1_224p.mkv -> hiệp 1, 2_224p.mkv -> hiệp 2)
        video_name = os.path.basename(input_video)
        half = '1' if '1_' in video_name else '2' if '2_' in video_name else None
        if half:
            max_game_time = get_max_game_time(annotation_file, half)
            if max_game_time:
                # Tính số segment tối đa (mỗi segment dài seconds_per_segment giây)
                max_segments = max_game_time #math.ceil(max_game_time / seconds_per_segment)
                # print(f"Hiệp {half} có thời gian tối đa {max_game_time}s, giới hạn {max_segments} segment")
    
    time_pairs = generate_time_pairs(
        duration,
        min_duration=min_duration,
        audio_pre=audio_pre,
        audio_post=audio_post,
        seconds_per_segment=seconds_per_segment,
        max_segments=max_segments
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
    output_base_dir = config["storage_params"]["output_base_dir"]
    save_dir_prefix = config["storage_params"]["save_dir_prefix"]
    max_workers = min(os.cpu_count(), 16)  # Sử dụng tối đa 8 luồng hoặc số lõi CPU
    
    game_paths = list_end_directories(input_base_dir)
    # Hàm xử lý cho một trận đấu
    def process_game(game_path):
        game_name = os.path.basename(game_path)
        output_dir = os.path.join(output_base_dir, game_name)
        
        # Tìm file Labels-v2.json trong game_path
        annotation_file = None
        for filename in annotation_files:
            if filename == "Labels-v2.json":
                src_path = os.path.join(game_path, filename)
                if os.path.exists(src_path):
                    annotation_file = src_path
                    break
        
        chunk_offset = 0
        total_segments = 0
        
        # Lock để tránh các vấn đề khi in trong nhiều luồng
        print_lock = threading.Lock()
        
        for video_name in video_names:
            video_path = os.path.join(game_path, video_name)
            if not os.path.exists(video_path):
                with print_lock:
                    print(f"Video {video_path} không tồn tại, bỏ qua.")
                continue
            
            num_segments = cut_video_and_audio(
                input_video=video_path,
                output_dir=output_dir,
                annotation_file=annotation_file,
                chunk_offset=chunk_offset,
                frame_interval=frame_interval,
                min_duration=min_duration,
                audio_pre=audio_pre,
                audio_post=audio_post,
                seconds_per_segment=seconds_per_segment,
                save_dir_prefix=save_dir_prefix
            )
            chunk_offset += num_segments
            total_segments += num_segments
            
            with print_lock:
                print(f"Đã xử lý {video_path}: {num_segments} đoạn")
        
        # Copy các file annotation
        for filename in annotation_files:
            src_path = os.path.join(game_path, filename)
            dst_path = os.path.join(output_dir, filename)
            if os.path.exists(src_path):
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.copy2(src_path, dst_path)
                with print_lock:
                    print(f"Đã copy {filename} vào {dst_path}")
            else:
                with print_lock:
                    print(f"{filename} không tồn tại trong {game_path}")
        
        return game_name, total_segments
    
    # Sử dụng ThreadPoolExecutor để xử lý đồng thời
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Tạo một tqdm progress bar cho tổng số game_paths
        futures = {executor.submit(process_game, game_path): game_path for game_path in game_paths}
        
        for future in tqdm(concurrent.futures.as_completed(futures), 
                          total=len(game_paths), 
                          desc="Xử lý các trận đấu"):
            game_path = futures[future]
            try:
                game_name, segments = future.result()
                results.append((game_name, segments))
            except Exception as e:
                print(f"Lỗi khi xử lý {game_path}: {e}")
    
    # Tổng kết
    print("\nKết quả xử lý:")
    total_segments = 0
    for game_name, segments in results:
        print(f"- {game_name}: {segments} đoạn")
        total_segments += segments
    
    print(f"\nTổng cộng: {len(results)} trận đấu, {total_segments} đoạn video")
    # game_paths = list_end_directories(input_base_dir)
    # for game_path in tqdm(game_paths, desc="Xử lý các trận đấu"):
    #     game_name = os.path.basename(game_path)
    #     output_dir = os.path.join(output_base_dir, game_name)
        
    #     # Tìm file Labels-v2.json trong game_path
    #     annotation_file = None
    #     for filename in annotation_files:
    #         if filename == "Labels-v2.json":
    #             src_path = os.path.join(game_path, filename)
    #             if os.path.exists(src_path):
    #                 annotation_file = src_path
    #                 break
        
    #     chunk_offset = 0
    #     for video_name in video_names:
    #         video_path = os.path.join(game_path, video_name)
    #         if not os.path.exists(video_path):
    #             print(f"Video {video_path} không tồn tại, bỏ qua.")
    #             continue
            
    #         num_segments = cut_video_and_audio(
    #             input_video=video_path,
    #             output_dir=output_dir,
    #             annotation_file=annotation_file,
    #             chunk_offset=chunk_offset,
    #             frame_interval=frame_interval,
    #             min_duration=min_duration,
    #             audio_pre=audio_pre,
    #             audio_post=audio_post,
    #             seconds_per_segment=seconds_per_segment,
    #             save_dir_prefix=save_dir_prefix
    #         )
    #         chunk_offset += num_segments
    #         print(f"Đã xử lý {video_path}: {num_segments} đoạn")
        
    #     for filename in annotation_files:
    #         src_path = os.path.join(game_path, filename)
    #         dst_path = os.path.join(output_dir, filename)
    #         if os.path.exists(src_path):
    #             os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    #             shutil.copy2(src_path, dst_path)
    #             print(f"Đã copy {filename} vào {dst_path}")
    #         else:
    #             print(f"{filename} không tồn tại trong {game_path}")

if __name__ == "__main__":
    # Chạy task process_video với overwrite seconds_per_segment
    task("process_video")

# One item handler
# game_paths = ["D:/projects/train/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley"]
# for game_path in tqdm(game_paths, desc="Xử lý các trận đấu"):
#     game_name = os.path.basename(game_path)
#     output_dir = os.path.join(output_base_dir, game_name)
    
#     chunk_offset = 0
#     for video_name in video_names:
#         video_path = os.path.join(game_path, video_name)
#         if not os.path.exists(video_path):
#             print(f"Video {video_path} không tồn tại, bỏ qua.")
#             continue
        
#         num_segments = cut_video_and_audio(
#             input_video=video_path,
#             output_dir=output_dir,
#             chunk_offset=chunk_offset,
#             frame_interval=frame_interval,
#             min_duration=min_duration,
#             audio_pre=audio_pre,
#             audio_post=audio_post,
#             seconds_per_segment=seconds_per_segment,
#             save_dir_prefix=save_dir_prefix
#         )
#         chunk_offset += num_segments
#         print(f"Đã xử lý {video_path}: {num_segments} đoạn")