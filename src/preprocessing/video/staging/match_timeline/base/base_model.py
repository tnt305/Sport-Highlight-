import os
from typing import Tuple, List, Dict
from dataclasses import dataclass
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field

class IsScoreBoardBase(ABC):
    def __init__(self, base_image: str):
        """
        Initializes the IsScoreBoardBase with a base image path.
    
        :param base_image: Path to the base image.
        :raises FileNotFoundError: If the base image path does not exist.
        """
        self.base_image = base_image
        if not os.path.exists(base_image):
            raise FileNotFoundError('Đường dẫn này hiện KHÔNG TỒN TẠI')
    @abstractmethod
    def get_roi(self, cordinates: Tuple[int, int, int, int]):
        pass
    @abstractmethod
    def extract_scoreboard(self):
        pass
    @abstractmethod
    def save_to(self):
        pass   
    
class ScoreBoardItem(ABC):
    home_team: str
    away_team: str
    score: str
    timestamp: str
    

@dataclass
class AudioSegment:
    """Class để lưu thông tin về một đoạn audio"""
    start_time: float
    end_time: float
    
    def __str__(self):
        return f"{self.start_time:.2f}s - {self.end_time:.2f}s"

class AudioEnergyConfig(BaseModel):
    """Configuration cho audio analysis"""
    frame_length: int = Field(default=2048, description="Độ dài của mỗi frame")
    hop_length: int = Field(default=512, description="Khoảng cách giữa các frame")
    energy_percentile: int = Field(default=50, description="Ngưỡng phần trăm cho energy min")
    energy_max: float = Field(default=10.0, description="Ngưỡng energy tối đa")
    max_gap: float = Field(default=2.0, description="Khoảng cách tối đa giữa các đoạn (giây)")
