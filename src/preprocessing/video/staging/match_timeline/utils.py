
import os
import cv2
import math
import re
from PIL import Image
from datetime import timedelta
from src.logger import Logging

log = Logging()


def is_timestamp(text: str):
    """
        Function:
            Kiểm tra một giá trị bounding box có phải là timestamp không
            Một bbox được coi là timestamp khi được phân biệt bởi dấu , hoặc : hoặc có dấu .

        Output:
            Một output nếu không phải là timestamp thì sẽ trả về None
    """
    try:
        float(text)
        return text.replace(".", ":")
    except ValueError:
        if ':' in text or ',' in text or '.' in text:
            separator = ':' if ':' in text else (',' if ',' in text else '.')
            before, after = text.split(separator)
            try:
                float(before.strip())
                return text.replace(separator, ":")
            except ValueError:
                return None
        return None


def convert_to_timedelta(time_str: str) -> timedelta:
    # Tách phút và giây từ chuỗi "mm:ss"
    minutes, seconds = map(int, time_str.split(':'))
    
    # Tạo đối tượng timedelta từ phút và giây
    return timedelta(minutes=minutes, seconds=seconds)

def scoreboard_cropper(root_dir: str, image_name: str):
    image_path = os.path.join(root_dir, image_name)
    if not os.path.exists(image_path):
        log.error(f"Image not found: {image_path}")
        return None
    
    crop_area = (0,0,400,100)
    
    with Image.open(image_path) as image:
        cropped_image = image.crop(crop_area)
        cropped_image_dir = image_name.split(".")[0] + "_" + "cropped" + ".png"
        cropped_image_path = os.path.join(root_dir, cropped_image_dir)
        cropped_image.save(cropped_image_path)
    log.info(f'Ảnh scoreboard được lưu tại {cropped_image_dir}')
    return cropped_image_path

def video2frames(video_path, output_folder, interval=60, max_frames=10, order='topdown'):
    video = cv2.VideoCapture(video_path)
    
    fps = math.ceil(video.get(cv2.CAP_PROP_FPS))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"FPS: {fps}")
    print(f"Total frames: {total_frames}")
    
    save_dir = f"{output_folder}/{order}"
    os.makedirs(save_dir, exist_ok=True)
    # Số frame giữa mỗi lần lưu
    frame_interval = fps * interval
    print(f"Frame interval: {frame_interval}")
    
    # Xác định frame bắt đầu dựa vào order
    if order == 'bottomup':
        current_frame = total_frames - frame_interval
    else:  # topdown
        current_frame = 0
        
    extracted_count = 0
    
    while extracted_count < max_frames:
        # Kiểm tra điều kiện dừng
        if (order == 'bottomup' and current_frame < 0) or \
           (order == 'topdown' and current_frame >= total_frames):
            break
            
        # Di chuyển đến frame cần xử lý
        video.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = video.read()
        
        if not ret:
            break
            
        # Tính thời gian của frame
        time_in_seconds = current_frame / fps
        minutes = int(time_in_seconds // 60)
        seconds = int(time_in_seconds % 60)
        milliseconds = int((time_in_seconds - int(time_in_seconds)) * 1000)
        
        # Lưu frame
        filename = f"{minutes:03d}-{seconds:03d}-{milliseconds:03d}.png"
        filepath = os.path.join(save_dir, filename)
        cv2.imwrite(filepath, frame)
        extracted_count += 1
        
        # Cập nhật vị trí frame tiếp theo
        if order == 'bottomup':
            current_frame -= frame_interval
        else:
            current_frame += frame_interval
    
    video.release()
    print(f"Đã trích xuất {extracted_count} frames và lưu vào {save_dir}")
    
##adding 
import subprocess
import os

def convert_to_ffmpeg_time(minutes, seconds):
    """
    Chuyển đổi phút và giây thành định dạng HH:MM:SS cho FFmpeg
    Hỗ trợ số phút lớn hơn 60
    """
    hours = 0 if minutes < 60 else minutes//60
    minutes = minutes if minutes < 60 else minutes - 60
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

def split_video_ffmpeg(input_file, timestamps):
    """
    Cắt video thành nhiều phần sử dụng FFmpeg
    
    Parameters:
    input_file (str): Đường dẫn đến file video gốc
    timestamps (list): Danh sách các timestamp dạng tuple (minutes, seconds)
    """
    # Kiểm tra FFmpeg
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True)
    except FileNotFoundError:
        print("Lỗi: FFmpeg chưa được cài đặt. Vui lòng cài đặt FFmpeg trước.")
        return

    # Lấy thời lượng video
    duration_cmd = [
        'ffprobe', 
        '-v', 'error', 
        '-show_entries', 'format=duration', 
        '-of', 'default=noprint_wrappers=1:nokey=1', 
        input_file
    ]
    
    try:
        duration = float(subprocess.check_output(duration_cmd).decode().strip())
        total_minutes = int(duration // 60)
        total_seconds = int(duration % 60)
        end_time = convert_to_ffmpeg_time(total_minutes, total_seconds)
    except subprocess.CalledProcessError:
        print("Lỗi: Không thể đọc thông tin video")
        return

    # Tạo thư mục output
    output_dir = "split_videos"
    os.makedirs(output_dir, exist_ok=True)

    # Chuyển đổi timestamps sang định dạng FFmpeg
    formatted_timestamps = [convert_to_ffmpeg_time(m, s) for m, s in timestamps]
    
    # Thêm điểm bắt đầu và kết thúc
    start_times = ['00:00:00'] + formatted_timestamps
    end_times = formatted_timestamps + [end_time]

    # Cắt video
    for i, (start, end) in enumerate(zip(start_times, end_times)):
        output_file = os.path.join(output_dir, f"part_{i+1}_{start.replace(':', '')}_to_{end.replace(':', '')}.mp4")
        
        cmd = [
            'ffmpeg',
            '-i', input_file,
            '-ss', start,
            '-to', end,
            '-c', 'copy',
            '-avoid_negative_ts', '1',
            output_file
        ]

        print(f"Đang xử lý phần {i+1}: {start} đến {end}")
        
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            print(f"Đã tạo file: {output_file}")
        except subprocess.CalledProcessError as e:
            print(f"Lỗi khi xử lý phần {i+1}: {e.stderr.decode()}")