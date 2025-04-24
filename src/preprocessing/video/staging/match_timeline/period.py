import os
import glob
from pathlib import Path
from typing import List, Tuple, Union, Dict
from src.preprocessing.video.staging.match_timeline.utils import scoreboard_cropper
from src.preprocessing.video.staging.match_timeline.event_detection import ServeBall
from src.preprocessing.video.staging.color_domain.dominant_color import DominantColorDetector

class PeriodDetection:
    PERIODS = {"Kick-off", "Secondhalf", "Advertising"}
    FPS = 2
    HALF_DURATION = 45 * 60 * FPS
    
    def __init__(self,
                 image_dir: Union[str, Path],
                 ):
        self.image_dir = Path(image_dir) if isinstance(image_dir, str) else image_dir
        self.serve_ball = ServeBall()
        self.color_check = DominantColorDetector()
        self.period_timestamps: Dict[str, List[int]] = {
            "Kick-off": [],
            "Secondhalf": [],
            "Advertising": []
        }
    
    def base_detector(self, image: str| Path, frame_idx):
        image_path = scoreboard_cropper(self.image_dir, image)
        timing = self.serve_ball.at_timestamp(image_path)
        if timing is not None and int(timing):
            if frame_idx >= self.HALF_DURATION:
                return "Secondhalf"
            return "Kick-off"

        if self.is_advertising(image_path):
            return "Advertising"
        return None

    def is_advertising(self, image_path: Path) -> bool:
        """
        An image is detected as advertising if only its color close to red
        """
        color_name = self.color_check.get_color_main(image_path)
        if 'red' in color_name.lower():
            return True
        else:
            return False
        
    def detector(self) -> Dict[str, List[int]]:
        """
        Detect periods across all images in the directory.
        
        Returns:
            Dict[str, List[int]]: Dictionary mapping period types to lists of frame timestamps
        """
        current_period = None
        frame_idx = 0
        
        # Sort images to ensure chronological order
        sorted_images = sorted(self.image_dir.glob("*.jpg"))  # Adjust pattern as needed
        
        for image in sorted_images:
            detected_period = self.base_detector(image, frame_idx)
            
            if detected_period and detected_period != current_period:
                # Store timestamp when period changes
                timestamp = frame_idx // self.FPS  # Convert frames to seconds
                self.period_timestamps[detected_period].append(timestamp)
                current_period = detected_period
                
            frame_idx += 1
            
        return self.period_timestamps
    
    def get_period_timestamps(self) -> Dict[str, List[Tuple[int, int]]]:
        """
        Get start and end timestamps for each period.
        
        Returns:
            Dict[str, List[Tuple[int, int]]]: Dictionary mapping period types to lists of 
            (start_time, end_time) tuples
        """
        period_ranges = {period: [] for period in self.PERIODS}
        timestamps = self.detector()
        
        for period in self.PERIODS:
            period_times = timestamps[period]
            for i in range(0, len(period_times), 2):
                if i + 1 < len(period_times):
                    period_ranges[period].append((period_times[i], period_times[i + 1]))
                else:
                    # Handle case where period hasn't ended
                    period_ranges[period].append((period_times[i], None))
                    
        return period_ranges

    def analyze_match(self) -> Dict[str, Dict[str, any]]:
        """
        Analyze the full match and provide detailed period information.
        
        Returns:
            Dict containing period analysis including:
            - timestamps for each period
            - duration of each period
            - sequence of periods
        """
        period_ranges = self.get_period_timestamps()
        analysis = {}
        
        for period, ranges in period_ranges.items():
            period_info = {
                "occurrences": len(ranges),
                "timestamps": ranges,
                "total_duration": sum(
                    (end - start) if end else 0 
                    for start, end in ranges if start is not None
                )
            }
            analysis[period] = period_info
            
        return analysis 

## TEST USAGE
# detector = PeriodDetection("path/to/image/directory")

# # Lấy timestamps cho các period
# timestamps = detector.detector()

# # Lấy phân tích chi tiết
# analysis = detector.analyze_match()