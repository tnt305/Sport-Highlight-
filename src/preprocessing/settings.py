import os
import yaml

def load_config(config_path="config/dataset_config.yaml", **kwargs):
    """Đọc cấu hình từ file YAML, khởi tạo hoặc ghi đè nếu có kwargs, kiểm tra tham số hợp lệ."""
    default_config = {
        "input_base_dir": "D:/projects/train",
        "video_extensions": [".mkv", ".mp4"],
        "video_names": ["1_224p.mkv", "2_224p.mkv"],
        "annotation_files": ["Labels-cameras.json", "Labels-v2.json"],
        "processing_params": {
            "frame_interval": 1,
            "min_duration": 10,
            "audio_pre": 10,
            "audio_post": 20,
            "seconds_per_segment": 60
        },
        "storage_params": {
            "output_base_dir": "F:/video_classification",
            "save_dir_prefix": "chunks"
        }
    }
    
    # Kiểm tra tham số hợp lệ
    valid_keys = set()
    def collect_keys(d, prefix=""):
        for key, value in d.items():
            if isinstance(value, dict):
                collect_keys(value, f"{prefix}{key}.")
            else:
                valid_keys.add(f"{prefix}{key}")
    collect_keys(default_config)
    
    # Xử lý kwargs dạng dictionary hoặc khóa lồng nhau
    config = default_config.copy()
    for key, value in kwargs.items():
        if key in ["processing_params", "storage_params"] and isinstance(value, dict):
            # Xử lý dictionary lồng nhau
            sub_config = config[key]
            for subkey, subvalue in value.items():
                full_key = f"{key}.{subkey}"
                if full_key not in valid_keys:
                    raise ValueError(f"Tham số '{full_key}' không hợp lệ. Các tham số hợp lệ: {sorted(valid_keys)}")
                sub_config[subkey] = subvalue
        elif key in valid_keys:
            # Xử lý khóa cấp cao (như input_base_dir)
            config[key] = value
        elif key.startswith("processing_params.") or key.startswith("storage_params."):
            # Xử lý khóa lồng nhau dạng chuỗi
            if key not in valid_keys:
                raise ValueError(f"Tham số '{key}' không hợp lệ. Các tham số hợp lệ: {sorted(valid_keys)}")
            section, subkey = key.split(".", 1)
            config[section][subkey] = value
        else:
            raise ValueError(f"Tham số '{key}' không hợp lệ. Các tham số hợp lệ: {sorted(valid_keys)}")
    
    # Ghi đè file config.yaml nếu có kwargs
    if kwargs:
        print(f"Đang ghi đè file {config_path} với các tham số: {kwargs}")
        try:
            with open(config_path, 'w') as f:
                yaml.safe_dump(config, f, default_flow_style=False)
            print(f"Đã ghi đè file {config_path}")
        except Exception as e:
            raise Exception(f"Lỗi khi ghi file {config_path}: {str(e)}")
        
        return config
    
    # Nếu không có kwargs, đọc từ file hoặc tạo mới
    if not os.path.exists(config_path):
        print(f"File {config_path} không tồn tại, đang khởi tạo với cấu hình mặc định...")
        try:
            with open(config_path, 'w') as f:
                yaml.safe_dump(default_config, f, default_flow_style=False)
            print(f"Đã tạo file {config_path}")
        except Exception as e:
            raise Exception(f"Lỗi khi tạo file {config_path}: {str(e)}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        raise Exception(f"Lỗi khi đọc file {config_path}: {str(e)}")
