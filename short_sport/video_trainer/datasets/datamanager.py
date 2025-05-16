import os
import sys
import random
import itertools
import torch
import numpy as np
from torchvision.transforms import Compose
from typing import (
    Iterable,
    List,
    Optional,
    TypeVar,
)

from torch.utils.data import (
    Dataset, 
    DataLoader,
    DistributedSampler, 
    Sampler,
    RandomSampler
)
# from catalyst.data.sampler import DistributedSamplerWrapper
from operator import itemgetter

# sys.path.append(os.path.abspath('/home/thiendc/projects/video_summarization/v5/src'))
from .transform import *
from .datasets import TV360Dataset

# from pytorchvideo.transforms import ApplyTransformToKey, create_video_transform



class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.
    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.
        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)

class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
            self,
            sampler,
            num_replicas: Optional[int] = None,
            rank: Optional[int] = None,
            shuffle: bool = True,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
              distributed training
            rank (int, optional): Rank of the current process
              within ``num_replicas``
            shuffle (bool, optional): If true (default),
              sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self):
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))

class LimitDataset(Dataset):
    """
    To ensure a constant number of samples are retrieved from the dataset we use this
    LimitDataset wrapper. This is necessary because several of the underlying videos
    may be corrupted while fetching or decoding, however, we always want the same
    number of steps per epoch.
    """

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.dataset_iter = itertools.chain.from_iterable(
            itertools.repeat(iter(dataset), 2)
        )

    def __getitem__(self, index):
        return next(self.dataset_iter)

    def __len__(self):
        return self.dataset.num_videos

class DataManager():
    def __init__(self, config, path):
        self.config = config
        self.path = path
        self.dataset = config['dataset']['name']
        self.total_length = config['model']['total_length']
        self.batch_size = config['dataset']['batch_size']
        self.num_workers = config['dataset']['num_workers']
        self.distributed = config['dataset']['distributed']
        random.seed(config['dataset']['seed'])
        np.random.seed(config['dataset']['seed'])
        torch.manual_seed(config['dataset']['seed'])
        torch.cuda.manual_seed(config['dataset']['seed'])
        torch.cuda.manual_seed_all(config['dataset']['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _check(self):
        datasets_list = ["tv360"]
        if self.dataset not in datasets_list:
            raise Exception("[ERROR] The dataset " + str(self.dataset) + " is not supported!")

    def get_num_classes(self):
        self._check()
        if self.dataset == "tv360": 
            return 18
        else: 
            raise NotImplementedError('No dataset is defined')

    def get_act_dict(self):
        self._check()
        tv360_dict = {
            "Penalty": 0,
            "Kick-off": 1,
            "Goal": 2,
            "Substitution": 3,
            "Offside": 4,
            "Shots on target": 5,
            "Shots off target": 6,
            "Clearance": 7,
            "Ball out of play": 8,
            "Throw-in": 9,
            "Foul": 10,
            "Indirect free-kick": 11,
            "Direct free-kick": 12,
            "Corner": 13,
            "Yellow card": 14,
            "Red card": 15,
            "Yellow->red card": 16,
            "Not Event": 17
        }
        if self.dataset == "tv360": 
            return tv360_dict
        else: 
            raise NotImplementedError('No dataset is defined')      

    def _build_transform(self, transform_config):
        """Build a Compose transform pipeline from config."""
        transforms = []
        for t in transform_config:
            transform_name, params = list(t.items())[0]
            if transform_name == 'GroupMultiScaleCrop':
                transforms.append(GroupMultiScaleCrop(**params))
            elif transform_name == 'GroupRandomHorizontalFlip':
                transforms.append(GroupRandomHorizontalFlip(**params))
            elif transform_name == 'GroupRandomColorJitter':
                transforms.append(GroupRandomColorJitter(**params))
            elif transform_name == 'GroupRandomGrayscale':
                transforms.append(GroupRandomGrayscale(**params))
            elif transform_name == 'GroupGaussianBlur':
                transforms.append(GroupGaussianBlur(**params))
            elif transform_name == 'GroupSolarization':
                transforms.append(GroupSolarization(**params))
            elif transform_name == 'GroupScale':
                transforms.append(GroupScale(**params))
            elif transform_name == 'GroupCenterCrop':
                transforms.append(GroupCenterCrop(**params))
            elif transform_name == 'Stack':
                transforms.append(Stack(**params))
            elif transform_name == 'ToTorchFormatTensor':
                transforms.append(ToTorchFormatTensor(**params))
            elif transform_name == 'GroupNormalize':
                transforms.append(GroupNormalize(**params))
            else:
                raise ValueError(f"Unsupported transform: {transform_name}")
        return Compose(transforms)

    def get_train_transforms(self):
        """Returns training transforms based on config."""
        self._check()
        if self.dataset == 'tv360':
            transforms = self._build_transform(self.config['dataset']['train_transforms'])
        else:
            raise NotImplementedError
        return transforms
    
    def get_test_transforms(self):
        """Returns evaluation transforms based on config."""
        self._check()
        if self.dataset == 'tv360':
            transforms = self._build_transform(self.config['dataset']['test_transforms'])
        else:
            raise NotImplementedError
        return transforms

    def get_train_loader(self, train_transform, drop_last=False):
        """Returns the training loader."""
        self._check()
        act_dict = self.get_act_dict()
        
        if self.dataset == 'tv360':
            train_data = TV360Dataset(self.path, act_dict, total_length=self.total_length, transform=train_transform, mode='train', ratio=(0.75, 0.15, 0.1))
            sampler = RandomSampler(train_data)
            if self.distributed:
                sampler = DistributedSamplerWrapper(sampler, shuffle=True)
            shuffle = False
        else:
            raise Exception("[ERROR] The dataset " + str(self.dataset) + " is not supported!")
        
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers, sampler=sampler, pin_memory=True, drop_last=drop_last)
        return train_loader
    
    def get_test_loader(self, test_transform, drop_last=False, mode='test'):
        """Returns the test loader."""
        self._check()
        act_dict = self.get_act_dict()
    
        if self.dataset == 'tv360':
            test_data = TV360Dataset(self.path, act_dict, total_length=self.total_length, transform=test_transform, mode=mode, ratio=(0.75, 0.15, 0.1))
        else:
            raise Exception("[ERROR] The dataset " + str(self.dataset) + " is not supported!")
        sampler = DistributedSampler(test_data, shuffle=False) if self.distributed else None
        shuffle = False
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers, sampler=sampler, pin_memory=False, drop_last=drop_last)
        return test_loader  

# class DataManager():
#     def __init__(self, args, path):
#         random.seed(args.seed)
#         np.random.seed(args.seed)
#         torch.manual_seed(args.seed)
#         torch.cuda.manual_seed(args.seed)
#         torch.cuda.manual_seed_all(args.seed)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False

#         self.path = path
#         self.dataset = args.dataset
#         self.total_length = args.total_length
#         # self.test_part = args.test_part
#         self.batch_size = args.batch_size
#         self.num_workers = args.num_workers
#         self.distributed = args.distributed
#         # self.animal = args.animal 
# #         Above, I store the subset animal name to set the training dataset

#     def _check(self,):
#         datasets_list = ["tv360"]
#         if(self.dataset not in datasets_list):
#             raise Exception("[ERROR] The dataset " + str(self.dataset) + " is not supported!")

#     def get_num_classes(self,):
#         self._check()
#         if(self.dataset == "tv360"): 
#             return 18
#         else: 
#             raise NotImplementedError('No dataset is defined')

#     def get_act_dict(self,):
#         self._check()
#         tv360_dict = {
#                     "Penalty":0,
#                     "Kick-off":1,
#                     "Goal":2,
#                     "Substitution":3,
#                     "Offside":4,
#                     "Shots on target":5,
#                     "Shots off target":6,
#                     "Clearance":7,
#                     "Ball out of play":8,
#                     "Throw-in":9,
#                     "Foul":10,
#                     "Indirect free-kick":11,
#                     "Direct free-kick":12,
#                     "Corner":13,
#                     "Yellow card":14
#                     ,"Red card":15,
#                     "Yellow->red card":16,
#                     "Event not recognized": 17
#                     }
    
#         if(self.dataset == "tv360"): 
#             return tv360_dict
#         else: 
#             raise NotImplementedError('No dataset is defined')      

#     def get_train_transforms(self,):
#         """Returns the training torchvision transformations for each dataset/method.
#            If a new method or dataset is added, this file should by modified
#            accordingly.
#         Args:
#           method: The name of the method.
#         Returns:
#           train_transform: An object of type torchvision.transforms.
#         """
#         self._check()
#         input_mean = [0.48145466, 0.4578275, 0.40821073]
#         input_std = [0.26862954, 0.26130258, 0.27577711]
#         input_size = 224

#         if self.dataset == 'tv360':
#             unique = Compose([GroupMultiScaleCrop(input_size, [1, .875, .75, .66]),
#                               GroupRandomHorizontalFlip(True),
#                               GroupRandomColorJitter(p=0.8, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
#                               GroupRandomGrayscale(p=0.2),
#                               GroupGaussianBlur(p=0.0),
#                               GroupSolarization(p=0.0)])
#             common = Compose([Stack(roll=False),
#                             ToTorchFormatTensor(div=True),
#                             GroupNormalize(input_mean, input_std)])
#             transforms = Compose([unique, common])
#         else:
#             raise NotImplementedError
#         return transforms
    
#     def get_test_transforms(self,):
#         """Returns the evaluation torchvision transformations for each dataset/method.
#            If a new method or dataset is added, this file should by modified
#            accordingly.
#         Args:
#           method: The name of the method.
#         Returns:
#           test_transform: An object of type torchvision.transforms.
#         """
#         self._check()
#         input_mean = [0.48145466, 0.4578275, 0.40821073]
#         input_std = [0.26862954, 0.26130258, 0.27577711]
#         input_size = 224
#         scale_size = 256

#         if self.dataset == 'tv360':
#             unique = Compose([GroupScale(scale_size),
#                               GroupCenterCrop(input_size)])
#             common = Compose([Stack(roll=False),
#                               ToTorchFormatTensor(div=True),
#                               GroupNormalize(input_mean, input_std)])
#             transforms = Compose([unique, common])
#         else:
#             raise NotImplementedError
#         return transforms

#     def get_train_loader(self, train_transform, drop_last=False):
#         """Returns the training loader for each dataset.
#            If a new method or dataset is added, this method should by modified
#            accordingly.
#         Args:
#           path: disk location of the dataset.
#           dataset: the name of the dataset.
#           total_length: the number of frames in a video clip
#           batch_size: the mini-batch size.
#           train_transform: the transformations used by the sampler, they
#             should be returned by the method get_train_transforms().
#           num_workers: the total number of parallel workers for the samples.
#           drop_last: it drops the last sample if the mini-batch cannot be
#              aggregated, necessary for methods like DeepInfomax.            
#         Returns:
#           train_loader: The loader that can be used a training time.
#         """
#         self._check()
#         act_dict = self.get_act_dict()
        
#         if(self.dataset == 'tv360'):
#             train_data = TV360Dataset(self.path, act_dict, total_length=self.total_length, transform=train_transform, mode='train',ratio=(0.75, 0.15, 0.1))
#             sampler = RandomSampler(train_data)
#             if self.distributed:
#                 sampler = DistributedSamplerWrapper(sampler, shuffle=True)
#             shuffle = False
#         else:
#             raise Exception("[ERROR] The dataset " + str(self.dataset) + " is not supported!")
        
#         train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers, sampler=sampler, pin_memory=False, drop_last=drop_last)
#         return train_loader
    
#     def get_test_loader(self, test_transform, drop_last=False, mode = 'test'):
#         """Returns the test loader for each dataset.
#            If a new method or dataset is added, this method should by modified
#            accordingly.
#         Args:
#           path: disk location of the dataset.
#           dataset: the name of the dataset.
#           total_length: the number of frames in a video clip
#           batch_size: the mini-batch size.
#           train_transform: the transformations used by the sampler, they
#             should be returned by the method get_train_transforms().
#           num_workers: the total number of parallel workers for the samples.
#           drop_last: it drops the last sample if the mini-batch cannot be
#              aggregated, necessary for methods like DeepInfomax.            
#         Returns:
#           train_loader: The loader that can be used a training time.
#         """
#         self._check()
#         act_dict = self.get_act_dict()
    
#         if(self.dataset == 'tv360'):
#             test_data = TV360Dataset(self.path, act_dict, total_length=self.total_length, transform=test_transform, mode= mode, ratio=(0.75, 0.15, 0.1))
#         else:
#             raise Exception("[ERROR] The dataset " + str(self.dataset) + " is not supported!")
#         sampler = DistributedSampler(test_data, shuffle=False) if self.distributed else None
#         shuffle = False
#         test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers, sampler=sampler, pin_memory=False, drop_last=drop_last)
#         return test_loader
