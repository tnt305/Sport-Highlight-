import os
import math
import shutil
import ffmpeg  # noqa
from tqdm import tqdm  # noqa
from pydantic import BaseModel, field_validator

from src.logger import Logging
from src.preprocessing.video.utils import (
    video2frames,
    create_video_chunk,
    normalize_path,
)  # noqa

from concurrent.futures import ThreadPoolExecutor  
from src.preprocessing.audio import create_audio_chunk

logger = Logging(log_file="logs/video_processor.log")


class VideoProcessorConfig(BaseModel):
    input_file: str
    segment_duration: int
    base_output_dir: str
    output_dir: str | None = None
    session: str
    overlapping: int
    frame_interval: int
    video_name: str | None = None

    @field_validator("input_file")
    def validate_inut_file(cls, v: str):
        if not (v.endswith("1_224p.mkv") or v.endswith("2_224p.mkv")):
            logger.debug(
                f"""INPUT_FILE is NOT training set of SoccerNetv2. \n
                Sample test {v} is activated
                """
            )
        if not os.path.isfile(v):
            raise ValueError(f"INPUT_FILE is not A FILE. You got {v}")
        if not os.path.exists(v):
            raise ValueError(f"INPUT_FILE is not EXIST. You got {v}")
        return v

    @field_validator("video_name")
    def set_video_name(cls, v, values):
        if v is None and "input_file" in values:
            return os.path.splitext(os.path.basename(values["input_file"]))[0]
        return v

    @field_validator("session")
    def validator_session(cls, v: str):
        valid_sessions = ["prematch", "first_half", "second_half", "extra_first_half", "extra_second_half", "penalty"]
        if v not in valid_sessions:
            raise ValueError(
                f"Session must be one of {', '.join(valid_sessions)}. You got {v}"
                )
        return v

    @field_validator("segment_duration")
    def validator_segment_duration(cls, v: int):
        if v <= 20:
            raise ValueError(f"Segment duration must be greater than 20. You got {v}")
        return v

    @field_validator("output_dir")
    def validator_output_dir(cls, v: str | None, info):
        if v is None or v == "None" or not os.path.exists(v):
            logger.warning(f"Output directory does not exist. You got {v}")
            if v is not None and not v.endswith(".mkv"):
                logger.info(f"Creating folder {v}")
                os.makedirs(v, exist_ok=True)
            else:
                pass
                # logger.debug(
                # "This has structured as video format, no make directory for this"
                # )
            base_dir = info.data["base_output_dir"]
            input_file = info.data["input_file"]
            v = os.path.join(base_dir, os.path.basename(os.path.dirname(input_file)))
            # logger.debug(f"Using replacement outputdir with {v}")
        else:
            logger.warning("There is no attribute base_output_dir or input_file")

        if not os.path.exists(v):
            # logger.debug(f"Output directory does not exist. Creating: {v}")
            if "224p.mkv" in v:
                pass
            elif v is None:
                pass
            elif v == "None":
                pass
            else:
                os.makedirs(v, exist_ok=True)
        else:
            logger.warning(f"Output directory will automatically created as {v}")
        return v


class VideoProcessor:
    """
        input_file: video đầu vào
        output_dir: đầu ra lưu các fiel video, audio và frames
        Segment duration = thời lượng của một chunk
        overlapping =  lấy trùng (trong thời lượng âm thanh hoặc frames)
        Do số lượng segments tăng sử dụng overlapping giữa 2 chunk liên tục,
        điều đó đồng nghĩa với việc số lượng frames và âm thanh tăng lên
    """
    def __init__(self, config: VideoProcessorConfig):
        self.config = config

    def ffmpeg_cmd(cls,
                   input_file,
                   output_dir,
                   segment_duration,
                   overlapping: int,
                   do_sep=True):
        logger.info(f'Đây là output_dir for ffmpeg: {output_dir}')
        probe = ffmpeg.probe(input_file)
        video_duration = math.ceil(float(probe['format']['duration']))
        # 2*(video_duration / segment_duration) - 1 for formulation
        num_segments = math.ceil((video_duration / segment_duration))
        # testcase
        logger.info(f"Đang xử lý {input_file}")
        # start time offset xử lý vấn đề về sự thay đổi trong hiệp 1 và hiệp 2
        if input_file.split("/")[-1].startswith("2_"):
    
            start_time_offset = len(os.listdir(output_dir))
        else:
            start_time_offset = 0

        # một video dài 90 phút sẽ chia làm 90 chunk + 89 chunk do lấy middle
        # một chunk 60s
        # v1 = 60+ 0
        # v2 = 2* 30
        for i in tqdm(range(num_segments), desc='Chunking video into parts'):
            start_time = max(0, i * (segment_duration - 0) + start_time_offset)

            if start_time + segment_duration > video_duration:
                duration = video_duration - start_time
                if duration < 10:
                    print(f"Duration is less than 10, pass at {i} chunk for {input_file}")
                    break
            else:
                duration = segment_duration

            # Create chunk directory
            chunk_dir = f"{output_dir}/chunk_{i+start_time_offset}"
            os.makedirs(chunk_dir, exist_ok=True)
            if os.listdir(chunk_dir) != []:
                for item in os.listdir(chunk_dir):
                    os.remove(os.path.join(chunk_dir, item))

            # Create output file
            output_file = f"{chunk_dir}/video_{i+start_time_offset}_start_{start_time}_end_{int(start_time + duration)}.mp4"
            create_video_chunk(input_file, output_file, start_time, duration)

            if do_sep:
                start_time_for_audio = max(0, start_time - overlapping)  # replaced i*segment_duration with start_time
                remaining_time = video_duration - start_time_for_audio

                if remaining_time < 10:
                    continue

                durationv2 = remaining_time if remaining_time < segment_duration else segment_duration * 1.2 if i == 0 else segment_duration + 60
        
                output_audio = f"{chunk_dir}/audio_{i+start_time_offset}_start_{start_time_for_audio}_end_{int(start_time_for_audio + durationv2)}.wav"
                frames_dir = f"{chunk_dir}/frames"
                # Generate audio file from chunk (retain original quality)                
                create_audio_chunk(input_file, output_audio,
                                   start_time_for_audio, durationv2)
                # Extract frames from video chunk
                video2frames(output_file, frames_dir, 2)

    def chunk_video(self, do_sep=False):
        """
            Chia video thành các đoạn
        """
        if self.config.input_file.split("/")[-1].startswith("1_"):
            self.ffmpeg_cmd(self.config.input_file,
                            self.config.output_dir,
                            self.config.segment_duration,
                            self.config.overlapping,
                            do_sep)
            logger.info(f"Đã chia video cho HIỆP_I thành các đoạn trong thư mục: {self.config.output_dir}")

            input_for_2nd_half = normalize_path(os.path.join("/".join(self.config.input_file.split("/")[:-1]), "2_224p.mkv"))
            self.ffmpeg_cmd(input_for_2nd_half,
                            self.config.output_dir,
                            self.config.segment_duration,
                            self.config.overlapping,
                            do_sep)
            logger.info(f"Đã chia video cho HIỆP_II thành các đoạn trong thư mục: {self.config.output_dir}")
    
    @classmethod
    def post_process(cls, game_path):
        game_chunk_paths = sorted(
        [
            os.path.join(game_path, i, 'frames')
            for i in os.listdir(game_path)
            if not i.endswith('.json')
        ]
        )
        
        for item in game_chunk_paths:
            if len(os.listdir(item)) < 10:
                parent_dir = os.path.dirname(item)
                shutil.rmtree(parent_dir)

    @classmethod
    def create_and_copy_path(self):
        if os.path.isfile(os.path.join(os.path.dirname(self.config.input_file), 'Labels-v2.json')):
            game_name = self.config.input_file.split("/")[-2]
            new_game_path = os.path.join(self.config.base_output_dir, game_name)
            # if not os.path.exists(new_game_path) and not (new_game_path.endswith(".mkvv") or new_game_path.endswith(".mp4")) :
            #     os.makedirs(new_game_path, exist_ok=True)
        
            # Check if the path does not exist
            path_does_not_exist = not (game_name == "1_224.mkv" or game_name == "2_224.mkv")
            # Check if the path does not end with the specified extensions
            invalid_extension = not (
                new_game_path.endswith(".mkv") or 
                new_game_path.endswith(".mp4")
                )

            # Create the directory if both conditions are true
            if path_does_not_exist or invalid_extension:
                os.makedirs(new_game_path, exist_ok=True)
            
            src_file = os.path.join(os.path.dirname(self.config.input_file), 'Labels-v2.json')
            dest_file = os.path.join(new_game_path, 'Labels-v2.json')
            if not os.path.exists(dest_file) or os.path.getmtime(src_file) > os.path.getmtime(dest_file):
                shutil.copy2(src_file, dest_file)
                
            return new_game_path
        else:
            print(f"Labels-v2.json does not exist in {self.config.input_file}")
            return None
     
    def process(cls):
        game_path = cls.create_and_copy_path()
        cls.chunk_video(do_sep=True)
        cls.post_process(game_path)
    
    def batch_process(self, worker: int = 20, ):
        with ThreadPoolExecutor(max_workers=worker) as executor:
            executor.map(self.process, self.config.input_file)
    def run(self, do_sep=True):
        # logger.info(f"Processing video {self.config.input_file}")
        self.chunk_video(do_sep=do_sep)
