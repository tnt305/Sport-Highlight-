import numpy as np
import librosa
from typing import List
from pydantic import BaseModel, Field
from typing import List, Tuple, Optional
from src.preprocessing.video.staging.match_timeline.base.base_model import AudioSegment, AudioEnergyConfig
from src.logger import Logging

class AudioEnergyDetector:
    """
    """
    def __init__(self, config: Optional[AudioEnergyConfig] = None):
        self.config = config or AudioEnergyConfig()
        self.log = Logging() 
    def compute_energy(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
        """Tính toán năng lượng của từng frame"""
        energy = librosa.feature.rms(y= audio, frame_length= self.config.frame_length, hop_length= self.config.hop_length)[0]
        # Tạo trục thời gian
        times = np.arange(len(energy)) * self.config.hop_length / sr
        
        return energy, times
    
    def find_segments(self, energy: np.ndarray, times: np.ndarray) -> List[AudioSegment]:
        """Tìm các đoạn có năng lượng nằm trong khoảng cho phép"""
        energy_threshold_min = np.percentile(energy, self.config.energy_percentile)
        energy_threshold_max = self.config.energy_max
        
        segments = []
        start = None
        
        for i, e in enumerate(energy):
            if energy_threshold_min < e < energy_threshold_max:
                if start is None:
                    start = times[i]
            else:
                if start is not None:
                    segments.append(AudioSegment(start, times[i]))
                    start = None
        
        # Xử lý đoạn cuối nếu có
        if start is not None:
            segments.append(AudioSegment(start, times[-1]))
            
        return segments
    
    def merge_segments(self, segments: List[AudioSegment]) -> AudioSegment:
        """Gộp các đoạn gần nhau và tìm đoạn dài nhất"""
        if not segments:
            return None
            
        # Khởi tạo các nhóm đoạn
        merged_groups = []
        current_group = [segments[0]]
        
        for segment in segments[1:]:
            if segment.start_time - current_group[-1].end_time <= self.config.max_gap:
                current_group.append(segment)
            else:
                merged_groups.append(current_group)
                current_group = [segment]
        
        merged_groups.append(current_group)
        
        # Tìm nhóm có tổng thời gian lớn nhất
        longest_group = max(merged_groups, 
                          key=lambda group: group[-1].end_time - group[0].start_time)
        
        return AudioSegment(
            start_time=longest_group[0].start_time,
            end_time=longest_group[-1].end_time
        )
    
    def analyze_file(self, file_path: str) -> AudioSegment:
        """Phân tích file audio và trả về đoạn dài nhất thỏa mãn điều kiện"""
        # Load audio
        try:
            audio, sr = librosa.load(file_path)
        except Exception as e:
            self.log.error(f'Lỗi khi load file {file_path} : {e}. Hãy kiểm tra lại đường dẫn')
            return None
        # Tính năng lượng
        energy, times = self.compute_energy(audio, sr)
        
        # Tìm các đoạn thỏa mãn điều kiện
        segments = self.find_segments(energy, times)
        
        # Gộp các đoạn và tìm đoạn dài nhất
        longest_segment = self.merge_segments(segments)
        if longest_segment is None:
            self.log.warning("Không tìm thấy đoạn audio nào thỏa mãn điều kiện, cân nhắc nới rộng threshold trong base_model.AudioEnergyConfig")
        
        return longest_segment