import os
import math
import numpy
from PIL import Image

def sample_frames_indices(video: str, method: str, spacing: int):
        list_frames = [f"{video}/{i}" for i in os.listdir(video) if not i.endswith(".json")]
        if method == 'from_top':
            start_idx = 0
            end_idx = len(list_frames) - 1
            # Giới hạn số lượng indices tối đa theo x
            actual_spacing = min(spacing, 30, len(list_frames))
            
            indices = numpy.linspace(start_idx, end_idx, actual_spacing)
            indices = numpy.clip(indices, start_idx, end_idx - 1).astype(numpy.int64)
        elif method == 'from_middle':
            center_idx = math.ceil(len(list_frames) / 2) - 1
            half_spacing = spacing // 2
            
            # Xác định khoảng lấy mẫu
            start_idx = max(0, center_idx - half_spacing)
            end_idx = min(len(list_frames) - 1, center_idx + half_spacing)
            
            # Giới hạn số lượng indices tối đa theo x
            actual_spacing = min(spacing, 30, len(list_frames))
            
            indices = numpy.linspace(start_idx, end_idx, actual_spacing)
            indices = numpy.clip(indices, 0, len(list_frames) - 1).astype(numpy.int64)
            
        frames_list = []  
        for i, j in enumerate(list_frames):
            if i in indices:
                frames_list.append(j)
        print(frames_list)
        frames_list = [Image.open(img).convert('RGB') for img in frames_list]
        
        return frames_list