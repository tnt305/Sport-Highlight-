import os
import json
import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix, save_npz, load_npz
from loguru import logger

def compute_cooccurrence(loader, num_classes, cache_dir="./cache"):
    # Tạo thư mục cache nếu chưa có
    os.makedirs(cache_dir, exist_ok=True)
    
    # Tạo tên file dựa trên cấu hình
    dataset_name = "my_dataset"  # Thay bằng tên dataset thực tế
    cache_file = os.path.join(cache_dir, f"cooccur_{dataset_name}_{num_classes}.npz")
    metadata_file = os.path.join(cache_dir, f"cooccur_{dataset_name}_{num_classes}.json")
    
    # Kiểm tra file cache
    if os.path.exists(cache_file) and os.path.exists(metadata_file):
        try:
            # Kiểm tra metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            # Giả sử bạn có cách tính hash của loader (hoặc số mẫu)
            if metadata['num_samples'] == len(loader.dataset):
                logger.info(f"Loading co-occurrence matrix from {cache_file}")
                return load_npz(cache_file)
            else:
                logger.warning("Dataset changed, recomputing co-occurrence matrix")
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
    
    # Tính toán ma trận đồng xuất hiện
    logger.info("Computing co-occurrence matrix...")
    cooccur = np.zeros((num_classes, num_classes))
    for _, labels in tqdm(loader):
        labels = labels.numpy()
        cooccur += labels.T @ labels
    cooccur = cooccur / np.maximum(1, np.diag(cooccur)[:, None])
    cooccur_matrix = csr_matrix(cooccur)
    
    # Lưu ma trận và metadata
    try:
        save_npz(cache_file, cooccur_matrix)
        metadata = {'num_samples': len(loader.dataset), 'num_classes': num_classes}
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
        logger.info(f"Saved co-occurrence matrix to {cache_file}")
    except Exception as e:
        logger.error(f"Error saving cache: {e}")
    
    return cooccur_matrix