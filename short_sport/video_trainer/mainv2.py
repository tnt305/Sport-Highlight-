import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torchmetrics.classification import (
    MultilabelAveragePrecision,
    MultilabelAccuracy
)
import yaml
from loguru import logger
import mlflow
from scipy.sparse import csr_matrix
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))  # Lên 3 cấp đến v5
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from utils import read_config, AverageMeter
from datasets.datamanager import DataManager
from short_sport.video_trainer.architecture.sub_modules.matrix import compute_cooccurrence
from losses.loss import BinaryFocalLoss, TwoWayLoss, AsymmetricLossOptimized, CalibratedRankingLoss, CorrelationAwareLoss
# import logging
from logger import Logging
# Set the PYTORCH_CUDA_ALLOC_CONF environment variable
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:50'

def create_default_config(args):
    """Tạo cấu hình mặc định cho file YAML dựa trên args"""
    return {
        'model': {
            'name': args.model,
            'total_length': args.total_length,
            'num_classes': 18,  # Sẽ được cập nhật sau khi load dataset
            'pretrained': 'facebook/timesformer-hr-finetuned-k600',
            'gnn': {
                'in_channels': 768,
                'out_channels': 256,
                'heads': 2
            }
        },
        'dataset': {
            'name': args.dataset,
            'path': 'D:/projects/v2v/v5/data',  # Cần thay đổi thành đường dẫn thực tế
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'distributed': args.distributed,
            'seed': args.seed,
            'train_transforms': [
                {'GroupMultiScaleCrop': {'input_size': 224, 'scales': [1, 0.875, 0.75, 0.66]}},
                {'GroupRandomHorizontalFlip': {'is_sth': True}},  # True tương đương p=0.5
                {'GroupRandomColorJitter': {'p': 0.8, 'brightness': 0.4, 'contrast': 0.4, 'saturation': 0.2, 'hue': 0.1}},
                {'GroupRandomGrayscale': {'p': 0.2}},
                {'GroupGaussianBlur': {'p': 0.0}},
                {'GroupSolarization': {'p': 0.0}},
                {'Stack': {'roll': False}},
                {'ToTorchFormatTensor': {'div': True}},
                {'GroupNormalize': {'mean': [0.48145466, 0.4578275, 0.40821073], 'std': [0.26862954, 0.26130258, 0.27577711]}}
            ],
            'test_transforms': [
                {'GroupScale': {'scale_size': 256}},
                {'GroupCenterCrop': {'input_size': 224}},
                {'Stack': {'roll': False}},
                {'ToTorchFormatTensor': {'div': True}},
                {'GroupNormalize': {'mean': [0.48145466, 0.4578275, 0.40821073], 'std': [0.26862954, 0.26130258, 0.27577711]}}
            ]
        },
        'training': {
            'epochs': args.epochs,
            'max_steps': args.max_steps,
            'gradient_accumulation_steps': args.gradient_accumulation_steps,
            'seed': args.seed,
            'loss_function': args.loss_function,
            'optimizer': {
                'type': 'AdamW',
                'lr': 0.00001,
                'weight_decay': 0.01
            },
            'scheduler': {
                'type': 'cosine_with_warmup',
                'num_warmup_steps': int(args.max_steps * 0.1),
                'num_training_steps': args.max_steps
            },
            'distributed': args.distributed,
            'test_every': args.test_every,
            'gpu': args.gpu
        },
        'logging': {
            'log_file': './logs/visual_trainer_{model}_{dataset}_{loss_function}.log',
            'level': 'INFO'
        },
        'mlops': {
            'experiment_name': f'{args.model}_{args.dataset}',
            'run_id': None,
            'artifact_path': './artifacts',
            'tracking_uri': 'file:./mlruns'
        }
    }

def load_or_create_config(base_dir = "configs",config_path="config.yaml", args=None):
    """Đọc hoặc tạo file YAML cấu hình"""
    config_path = os.path.join(base_dir, config_path)
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    if os.path.exists(config_path):
        logger.info(f"Loading existing config from {config_path}")
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    else:
        logger.info(f"Config file {config_path} not found. Creating new config.")
        config = create_default_config(args)
        with open(config_path, 'w') as file:
            yaml.safe_dump(config, file, default_flow_style=False)
        logger.info(f"Created new config file at {config_path}")
        return config

def main(config, args):
    # Cấu hình logging
    log_file = config['logging']['log_file'].format(
        model=config['model']['name'],
        dataset=config['dataset']['name'],
        loss_function=config['training']['loss_function']
    )
    logger.remove()
    logger.add(log_file, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level=config['logging']['level'])

    # Cấu hình MLflow
    mlflow.set_tracking_uri(config['mlops']['tracking_uri'])
    mlflow.set_experiment(config['mlops']['experiment_name'])
    
    if(torch.cuda.is_available() == False): logger.warning("[WARNING] CUDA is not available.")

    with mlflow.start_run():
        params = {
            **{f"model_{k}": v for k, v in config['model'].items() if k != 'gnn'},  # Thêm tiền tố model_
            **{f"dataset_{k}": v for k, v in config['dataset'].items() if k not in ['train_transforms', 'test_transforms']},  # Thêm tiền tố dataset_
            **{f"training_{k}": v for k, v in config['training'].items()}  # Thêm tiền tố training_
        }
        mlflow.log_params(params)
        
        # Thiết lập seed
        seed = config['training']['seed']
        if seed >= 0:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logger.info(f"[INFO] Setting SEED: {seed}")
        else:
            logger.info("[INFO] Setting SEED: None")

        if not torch.cuda.is_available():
            logger.warning("[WARNING] CUDA is not available.")

        logger.info(f"[INFO] Found {torch.cuda.device_count()} GPU(s) available.")
        device = torch.device(f"cuda:{config['training']['gpu']}" if torch.cuda.is_available() else "cpu")
        logger.info(f"[INFO] Device type: {device}")

        # Tải dữ liệu
        manager = DataManager(config, config['dataset']['path'])
        class_list = list(manager.get_act_dict().keys())
        num_classes = len(class_list)
        config['model']['num_classes'] = num_classes  # Cập nhật số lớp
        
        train_transform = manager.get_train_transforms()
        train_loader = manager.get_train_loader(train_transform)
        logger.info(f"[INFO] Train size: {len(train_loader.dataset)}")

        val_transform = manager.get_test_transforms()
        val_loader = manager.get_test_loader(val_transform, mode='val')
        logger.info(f"[INFO] Val size: {len(val_loader.dataset)}")

        test_loader = manager.get_test_loader(val_transform, mode='test')
        logger.info(f"[INFO] Test size: {len(test_loader.dataset)}")


        # Cấu hình loss
        if config['dataset']['name'] in ['tv360']:
            if config['training']['loss_function'] == 'asl':
                criterion = AsymmetricLossOptimized()
            elif config['training']['loss_function'] == '2wl':
                criterion = TwoWayLoss()
            elif config['training']['loss_function'] == 'softmax_margin_w/o_weights':
                criterion = nn.MultiLabelMarginLoss()
            elif config['training']['loss_function'] == 'bce_with_logits':
                criterion = nn.BCEWithLogitsLoss()
            elif config['training']['loss_function'] == 'rank_bce':
                cooccur = compute_cooccurrence(train_loader, len(class_list))
                criterion = CorrelationAwareLoss(cooccur, base_loss=nn.BCEWithLogitsLoss(), alpha=0.3)
            else:
                raise ValueError(f"Unsupported loss function '{config['training']['loss_function']}' for dataset '{config['dataset']['name']}'")

            eval_metrics = {
                'mAP': MultilabelAveragePrecision(num_labels=num_classes, average='macro'),
                'Acc': MultilabelAccuracy(num_labels=num_classes, average='macro'),
                'weighted Accuracy': MultilabelAccuracy(num_labels=num_classes, average='micro'),
                'weighted v1 mAP': MultilabelAveragePrecision(num_labels=num_classes, average='micro')
            }
        # modeltest_transforms
        # Khởi tạo model
        model_args = (
            train_loader, val_loader, test_loader, criterion, eval_metrics, class_list,
            config['training']['test_every'], config['training']['distributed'], device,
            config['training']['max_steps'], config['training']['gradient_accumulation_steps'], logger
        )
        
        ## Model chose
        if config['model']['name'] == 'timesformer':
            from classifier import TimeSformerExecutor
            executor = TimeSformerExecutor(*model_args)
        elif config['model']['name'] == 'videomae':
            from classifier import VideoMaeExecutor
            executor = VideoMaeExecutor(*model_args)
        elif config['model']['name'] == 'timesformerclip':
            from classifier import TimeSformerCLIPInitExecutor
            executor = TimeSformerCLIPInitExecutor(*model_args) 
        # executor.model.to(device)
        logger.info('Start training')
        executor.run_training()
        try:
            eval = executor.test()
            mlflow.log_metrics({
                'mAP': float(eval['mAP']) * 100,
                'Acc': float(eval['Acc']) * 100,
                'weighted v1 mAP': float(eval['weighted v1 mAP']) * 100
            })
        except Exception as e:
            logger.error(f"Error during testing: {e}")   
        
        # Lưu model và log artifact
        model_path = f"ds_{config['dataset']['name']}_model_{config['model']['name']}_nepochs_{config['training']['epochs']}_nframes_{config['model']['total_length']}.pth"
        executor.save(model_path)
        mlflow.log_artifact(model_path)
        mlflow.log_artifact(log_file)
        mlflow.log_artifact("config.yaml")
    
        logger.info(f"MultiLabel Average Precision: {float(eval['mAP']) * 100:.3f}")
        logger.info(f"MultiLabel Accuracy: {float(eval['Acc']) * 100:.3f}")
        logger.info(f"Mean Average Precision: {eval['weighted v1 mAP'] * 100:.3f}")
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Training script for action recognition")
    parser.add_argument("--seed", default = 1, type=int, help="Seed for Numpy and PyTorch. Default: -1 (None)")
    parser.add_argument("--epoch_start", default=0, type=int, help="Epoch to start learning from, used when resuming")
    parser.add_argument("--epochs", default= 1, type=int, help="Total number of epochs")
    parser.add_argument("--dataset", default="tv360", help="Dataset: volleyball, hockey, charades, ava, animalkingdom")
    parser.add_argument("--model", default="timesformer")
    parser.add_argument("--total_length", default= 30,type=int, help="Number of frames in a video")
    parser.add_argument("--batch_size", default= 2,type=int, help="Size of the mini-batch")
    parser.add_argument("--max_steps", default= 6000,type=int, help="Number of frames in a video") 
    parser.add_argument("--gradient_accumulation_steps", default= 4,type=int, help="Number of frames in a video") 
    parser.add_argument("--id", default="", help="Additional string appended when saving the checkpoints")
    parser.add_argument("--checkpoint", default="", help="location of a checkpoint file, used to resume training")
    parser.add_argument("--num_workers", default = 15, type=int, help="Number of torchvision workers used to load data (default: 8)")
    parser.add_argument("--test_every", default=1, type=int, help="Test the model every this number of epochs")
    parser.add_argument("--gpu", default="0", type=str, help="GPU id in case of multiple GPUs")
    parser.add_argument("--distributed", default=False, type=bool, help="Distributed training flag")
    parser.add_argument("--train", default=True, type=bool, help="train or test")
    parser.add_argument("--loss_function", default = "rank_bce", help = "Los function")
    parser.add_argument("--config", default=None, help="Custom config file name (e.g., timesformer.yaml)")
    args = parser.parse_args()
    
    config_path = args.config if args.config else f"config_{args.model}.yaml"
    config = load_or_create_config(base_dir = 'config', config_path = config_path, args = args)
    main(config, args)
