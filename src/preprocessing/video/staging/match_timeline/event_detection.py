import easyocr
from pathlib import Path
from typing import List, Tuple, Optional
from src.logger.logger import Logging
from src.match_timeline.utils import is_timestamp

class ServeBall:
    def __init__(self,
                 fps: int = 2,
                 use_gpu: bool = True,
                 lang: str =  'en',
                 batch_size: int = 2,
                 low_text: float=  0.4,
                 model_dir: Optional[str] = None,
                 user_network_dir: Optional[str] = None):
        self.fps = fps
        self.lang = lang
        self.use_gpu  = use_gpu
        self.batch_size = batch_size
        self.low_text = low_text
        self.model_dir = model_dir or '/home/thiendc/projects/video_summarization/v2/model/easyocr'
        self.user_network_dir = user_network_dir or '/home/thiendc/projects/video_summarization/v2/model/easyocr'
        self.reader =  easyocr.Reader([lang], 
                                      gpu =  use_gpu, 
                                      model_storage_directory = self.model_dir,
                                      user_network_directory= self.user_network_dir,
                                      download_enabled= False)
        self.log = Logging()
        
    def scoreboard_reader(self, image_path: str | Path):
        return self.reader.readtext(image_path, 
                                    batch_size = self.batch_size, 
                                    low_text = self.low_text)
    
    @staticmethod
    def get_item(data: List):
        # bbox result
        bbox = [item[0] for item in data]
        y_values = [point[0][1] for point in bbox]
        # get text 
        text = [item[1] for item in data]
        # get confidence score
        conf_score = [item[2] for item in data]

        return y_values, text, conf_score

    def get_item_by_lines(self, image_path: str, y_threshold: int = 5):
        """
        Chia dữ liệu thành các dòng (lines) dựa trên các giá trị y của bounding boxes.
        Dữ liệu có giá trị y gần nhau sẽ được coi là cùng một dòng.
        """
        # Lấy thông tin y_values, confidence score và text
        data = self.scoreboard_reader(image_path)
        y_values, conf_scores, texts = self.get_item(data)

        # Sắp xếp các item theo giá trị y (từ thấp đến cao)
        sorted_data = list(zip(y_values, conf_scores, texts, data))
        
        # Khởi tạo danh sách các dòng (lines)
        lines = []
        current_line = []
        current_y = sorted_data[0][0]
        
        # Duyệt qua từng item trong dữ liệu đã sắp xếp và nhóm chúng theo dòng
        for y, _, _, item in sorted_data:
            # Nếu y_value của item này gần với y_value hiện tại, chúng cùng một dòng
            if abs(y - current_y) <= y_threshold:
                current_line.append(item)
            else: 
                # Nếu không, tạo một dòng mới
                lines.append(current_line)
                current_line = [item]
                current_y = y
        
        # Add last line
        if current_line:
            lines.append(current_line)
        return lines

    def timestamp2frames(self, timestamp: str):
        try:
            minutes, seconds = timestamp.split(':')[0], timestamp.split(':')[-1] 
            return self.fps * int(seconds) + self.fps * int(minutes) * 60
        except ValueError:
            self.log.error("Không có thông tin frame cho scoreboard, bỏ qua nội dung này")
            return None
        
        
    def at_timestamp(self, image_path: str, y_threshold: int = 5):
        lines = self.get_item_by_lines(image_path, y_threshold)
        for line in lines:
            text1, text2 =  line[0][1], line[-1][1]
            if is_timestamp(text1):
                current_time = is_timestamp(text1)
                return self.timestamp2frames(current_time)
            elif is_timestamp(text2):
                current_time = is_timestamp(text2)
                return self.timestamp2frames(current_time)
        return None


    def restart_timestamp(self, image_path: str, y_threshold: int = 5):
        value = self.at_timestamp(image_path, y_threshold)
        if value is not None:
            return (45*60 - value)*self.fps