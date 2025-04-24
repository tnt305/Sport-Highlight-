import os
import csv
import numpy
from tqdm import tqdm
from PIL import Image
import pandas
from torch.utils.data import Dataset
from itertools import compress
from logger import Logging
import pickle
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))  # Lên 3 cấp đến v5
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from short_sport.video_trainer.utils import normalize_path
from short_sport.video_trainer.datasets.indice_sampler import (
    random_sampling, 
    decremental_sampling,
    incremental_sampling,
    twopeak_sampling
)
logger = Logging()

class VideoAnnotations(object):
    def __init__(self, row):
        self._row = row
    
    @property
    def path(self):
        return self._row[0]
        
    @property
    def num_frames(self):
        return int(self._row[1])
    
    @property
    def label(self):
        return self._row[2]
    
    # @property
    # def audio(self):
    #     return self._row[3]
    
class ActionSpottingDataset(Dataset):
    def __init__(self, total_length):
        self.total_length = total_length
        self.video_lists = []
        self.random_shift = False
    
    def _sample_indices(self, num_frames):
        indices = random_sampling(num_frames, self.total_length, self.random_shift) ## 88.76
        # indices = incremental_sampling(self.total_length, num_frames, "gaussian+") ## 88.65
        #indices = twopeak_sampling(self.total_length, num_frames, "concentrated") # 88.56
        # indices = decremental_sampling(self.total_length, num_frames, "gaussian+")
        return indices
    
    @staticmethod
    def _load_image(dirs, img_name):
        return [Image.open(normalize_path(os.path.join(dirs, img_name))).convert('RGB')]
    
    def __getitem__(self, idx):
        record = self.video_list[idx]
        image_names = self.file_list[idx]
        indices = self._sample_indices(record.num_frames)
        return self._get(record, image_names, indices)
    
    def __len__(self):
        return len(self.video_list)
    
class TV360Dataset(ActionSpottingDataset):
    def __init__(self, 
                path,
                label2id,
                total_length,                 
                transform=None,
                random_shift = False,
                mode = 'train',
                ratio = (0.7, 0.2, 0.1)
            ):
        
        self.path = path
        self.total_length = total_length
        self.transform = transform
        self.random_shift = random_shift
        self.mode = mode
        self.ratio = ratio
        
        if self.mode not in ['train', 'val', 'test', 'predict']:
            raise ValueError(f"Invalid mode: {self.mode}")
                
        # if self.mode == 'train':
        #     self.set = 0.7
        # elif self.mode == 'val':
        #     self.set = 0.2
        # elif self.mode == 'test':
        #     self.set = 0.1
        
        self.annotations = normalize_path(os.path.join(self.path, 'timesformer_v2.csv'))
            
        #act_dict
        self.label2id = label2id
        #num_classes
        self.num_classes = len(label2id)
        
        if mode != 'predict':
            try:
                self.video_list, self.file_list = self._parse_annotations()
            except OSError:
                print('ERROR: Could not read annotation file "{}"'.format(self.annotations))
                raise
        
    def _parse_annotations(self):
        cache_filename = f'dataset_cache_{self.mode}_{"_".join(map(str, self.ratio))}.pkl'
        cache_file = os.path.join(self.path, cache_filename)
        
        # Kiểm tra và tải cache nếu tồn tại
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    # Kiểm tra tính toàn vẹn của cache
                    if len(cached_data) == 2 and isinstance(cached_data[0], list) and isinstance(cached_data[1], list):
                        logger.info(f"Loading cached dataset for mode: {self.mode}")
                        return cached_data
            except (pickle.UnpicklingError, EOFError):
                logger.warning("Cache file is corrupted. Regenerating dataset.")
        
        video_list = []
        file_list = []
        with open(self.annotations, 'r') as f:
            logger.info("Using csv reader without pandas instead")
            reader = csv.DictReader(f, delimiter=';')
            for row in tqdm(reader, desc='Reading annotations'):
                ovid = row['video_id']
                labels = row['labels']
                # audio = row['audio_id']
                path = os.path.join("/".join(ovid.split("/")[:-1]), 'frames')
                files = sorted(os.listdir(path))
                file_list += [files]
                count = len(files)
                labels = [int(l) for l in labels.split(',')]
                video_list += [VideoAnnotations([path, count, labels])]
        
        video_list, file_list = self._split_dataset(video_list, file_list)

        # Lưu cache
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump((video_list, file_list), f)
            logger.info(f"Saved dataset cache for mode: {self.mode}")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
            
        return video_list, file_list
    
    def _split_dataset(self, video_list, file_list):
        total = len(video_list)
        train_size = round(total * self.ratio[0])
        val_size = round(total * self.ratio[1])
        if self.mode == 'train':
            return video_list[:train_size], file_list[:train_size]
        elif self.mode == 'val':
            return video_list[train_size:train_size + val_size], file_list[train_size:train_size + val_size]
        elif self.mode == 'test':
            return video_list[train_size + val_size:], file_list[train_size + val_size:]
    
    def _get(self, record, img_names, indices):
        images = list()
        for idx in indices:
            try:
                img = self._load_image(record.path, img_names[idx])
            except:
                print('ERROR: Could not read image "{}"'.format(os.path.join(record.path, img_names[idx])))
                print('invalid indices: {}'.format(indices))
                raise
            images.extend(img)
            
        process_data = self.transform(images)
        # audio_path = record.audio
        process_data = process_data.view((self.total_length, -1) + process_data.size()[-2:])
        label = numpy.zeros(self.num_classes)  # need to fix this hard number
        label[record.label] = 1.0
        return process_data, label
    