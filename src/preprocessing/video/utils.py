import os
import json
from datetime import datetime, timedelta
import re
import math
import ffmpeg  # noqa
from pathlib import Path
import subprocess


def create_video_chunk(input: str, output: str, ss: int, duration: int):
    """Create a video chunk."""
    try:
        (
            ffmpeg
            .input(input, ss=ss, t=duration)
            .output(output, c="copy")
            .run(quiet=True, overwrite_output=False)
        )
    except ffmpeg.Error as e:
        print(f"Error occurred while creating video chunk: {e.stderr.decode()}")


# def create_video_chunk(input: str, output: str, ss: int, duration: int):
#     """
#     Create a video chunk using OpenCV.
#     Args:
#         input: Input video path
#         output: Output video path
#         ss: Start time in seconds
#         duration: Duration in seconds
#     """
#     try:
#         # Mở video input
#         cap = cv2.VideoCapture(input)
#         if not cap.isOpened():
#             print(f"Error: Could not open video {input}")
#             return False

#         # Lấy thông tin video
#         fps = int(cap.get(cv2.CAP_PROP_FPS))
#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#         # Tính toán frame bắt đầu và số frame cần lấy
#         start_frame = ss * fps
#         frames_to_capture = duration * fps

#         # Kiểm tra nếu start_frame vượt quá tổng số frame
#         if start_frame >= total_frames:
#             print(f"Start position ({ss}s) exceeds video duration ({total_frames/fps}s)")
#             cap.release()
#             return False

#         # Di chuyển đến frame bắt đầu
#         cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

#         # Tạo VideoWriter object
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # hoặc 'XVID' cho avi
#         out = cv2.VideoWriter(output, fourcc, fps, (width, height))

#         frames_processed = 0
        
#         # Đọc và ghi frames
#         while frames_processed < frames_to_capture:
#             ret, frame = cap.read()
#             if not ret:
#                 print("Reached end of video before completing chunk")
#                 break
                
#             out.write(frame)
#             frames_processed += 1

#             # Print progress mỗi 100 frames
#             if frames_processed % 100 == 0:
#                 progress = (frames_processed / frames_to_capture) * 100
#                 print(f"Progress: {progress:.2f}%")

#         # Giải phóng resources
#         cap.release()
#         out.release()

#         print(f"Successfully created video chunk: {frames_processed} frames processed")
#         return True

#     except Exception as e:
#         print(f"Error occurred while creating video chunk: {str(e)}")
#         return False



# def create_video_chunk(input: str, output: str, ss: int, duration: int):
#     """
#     Create a video chunk and convert from mkv to mp4.
#     Args:
#         input: Input file path (.mkv)
#         output: Output file path (should end with .mp4)
#         ss: Start time in seconds
#         duration: Duration in seconds
#     """
#     try:
#         # Đảm bảo output file có đuôi .mp4
#         if not output.endswith('.mp4'):
#             output = output.rsplit('.', 1)[0] + '.mp4'

#         (
#             ffmpeg
#             .input(input, ss=ss)
#             .output(output,
#                    t=duration,
#                    # Video codec settings
#                    vcodec='libx264',
#                    video_bitrate='1000k',
#                    # Audio codec settings
#                    acodec='aac',
#                    audio_bitrate='128k',
#                    # Container format
#                    f='mp4',
#                    # Additional encoding parameters
#                    preset='medium',  # Cân bằng giữa tốc độ encode và chất lượng
#             )
#             .run(quiet=True, overwrite_output=False)
#         )
#     except ffmpeg.Error as e:
#         print(f"Error occurred while creating video chunk: {e.stderr.decode()}")

def video2frames(video: str, save_dir: str, frame_interval: int):
    """Extract frames from a video file."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    try:
        (
            ffmpeg
            .input(video)
            .output(f"{save_dir}/frame_%04d.png", vf=f"fps=1/{frame_interval}")
            .run(quiet=True, overwrite_output=True)
        )
    except ffmpeg.Error as e:
        print(f"Error occurred while extracting frames: {e.stderr.decode()}")


def video2frames_v2(video: str,
                    save_dir: str,
                    frame_interval: int,
                    start_time: int,
                    end_time: int):
    """
        Extract frames from a video file.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    command = [
        "ffmpeg",
        "-i", video,
        "-loglevel", "error",
        "-vf", f"fps=1/{frame_interval}",
        "-ss", str(start_time),
        "-t", str(end_time),
        os.path.join(save_dir, "frame_%04d.png")
    ]
    subprocess.run(command, check=True)
    # print("Hoàn thành trích xuất các frame.")


def get_video_length(video_path: str) -> float:   
    try:
        # Lấy metadata của video
        probe = ffmpeg.probe(video_path)
        # Tìm thông tin duration trong streams
        duration = math.floor((float(probe['format']['duration'])))
        return duration
    except ffmpeg.Error as e:
        raise Exception(f"Error occurred while checking video length: {e.stderr.decode()}")
    except KeyError:
        raise Exception("Duration information not found in video metadata.")


def normalize_path(path: str):
    return Path(path).as_posix()


def extract_timestamps(filename):
    """
    Trích xuất start_time và end_time từ tên file video.
    :param filename: Tên file video (chuỗi)
    :return: (start_time, end_time) dưới dạng số nguyên
    """
    match = re.search(r"start_(\d+)_end_(\d+)", filename)
    if match:
        start_time = int(match.group(1))
        end_time = int(match.group(2))
        return start_time, end_time
    else:
        raise Exception("Không tìm thấy timestamp trong tên file!")


def has_exactly_two_mkv_files(directory: str) -> bool:
    count = 0
    with os.scandir(directory) as entries:
        for entry in entries:
            # Check if the current entry is a file and ends with ".mkv"
            if entry.is_file() and entry.name.endswith(".mkv"):
                count += 1
                # Exit early if more than two such files are found
                if count > 2:
                    return False
    return count == 2


def list_end_directories(path: str) -> list:
    """
    Recursively traverse a directory and 
    return all paths that are files (end of branch).
    """
    end_files = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.startswith('SoccerNetV2'):                
                pass
            else:
                refined_path = normalize_path(os.path.join(root, file))
                # print(refined_path)
                # if has_exactly_two_mkv_files(refined_path):
                end_files.append(refined_path)
    end_files = ["/".join(file.split("/")[:-1]) for file in end_files]
    end_files = [item for item in end_files if has_exactly_two_mkv_files(item)]
    end_files = sorted(list(set(end_files)))
    return end_files


def json_reader(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def round_down_to_minute(time_str):
    time_obj = datetime.strptime(time_str, "%M:%S")
    rounded_obj = time_obj.replace(second=0)  # Bỏ giây
    
    return rounded_obj.strftime("%M:%S")

def round_down_to_minute_v2(time_str):
    time_obj = datetime.strptime(time_str, "%M:%S")
    result_time = time_obj + timedelta(minutes=45)
    
    # Handle case where minutes overflow to hours
    total_minutes = result_time.hour * 60 + result_time.minute
    rounded_minutes = total_minutes  # Keep total minutes as is
    # seconds = result_time.second
    
    # Format back to "MM:SS"
    formatted_result = f"{rounded_minutes}:00"
    
    return formatted_result


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):

        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count