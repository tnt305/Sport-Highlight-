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

def main(args):
    logger = Logging(log_name="TimeSformer", log_file = f"./logs/visual_trainer_{args.model}_{args.dataset}_{args.loss_function}.log")
    if(args.seed>=0):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        logger.info(f"[INFO] Setting SEED: {str(args.seed)}")   
    else:
        logger.info("[INFO] Setting SEED: None")

    if(torch.cuda.is_available() == False): print("[WARNING] CUDA is not available.")

    logger.info("[INFO] Found", str(torch.cuda.device_count()), "GPU(s) available.")
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")   
    logger.info(f"[INFO] Device type:{str(device)}")

    config = read_config()
    path_data = config['path_dataset']

    logger.info(f"[INFO] Dataset path:{path_data}")

    manager = DataManager(args, path_data)
    class_list = list(manager.get_act_dict().keys())
    num_classes = len(class_list)

    # training data
    train_transform = manager.get_train_transforms()
    print(train_transform, flush = True, )
    train_loader = manager.get_train_loader(train_transform)
    logger.info(f"[INFO] Train size:{str(len(train_loader.dataset))}")

    # val or test data
    val_transform = manager.get_test_transforms()
    val_loader = manager.get_test_loader(val_transform, mode = 'val')
    logger.info(f"[INFO] Test size: {str(len(val_loader.dataset))}")

    # test data
    val_transform = manager.get_test_transforms()
    test_loader = manager.get_test_loader(val_transform, mode = 'test')
    logger.info(f"[INFO] Test size: {str(len(test_loader.dataset))}")

    def compute_cooccurrence_matrix(dataloader, num_classes):
        cooccur = np.zeros((num_classes, num_classes))
        for _, labels in tqdm(dataloader):
            labels = labels.numpy()
            cooccur += labels.T @ labels  # Ma trận đồng xuất hiện
        # Chuẩn hóa
        cooccur = cooccur / np.maximum(1, np.diag(cooccur)[:, None])
        return csr_matrix(cooccur)
    # criterion or loss
    if args.dataset in ['tv360']:
        if args.loss_function == 'asl':
            criterion =  AsymmetricLossOptimized() #TwoWayLoss() #nn.MultiLabelSoftMarginLoss(weight = torch.tensor([3.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 3.0, 3.0, 1.0])) #BinaryFocalLoss(gamma = 1) #nn.BCEWithLogitsLoss()
        elif args.loss_function == '2wl':
            criterion = TwoWayLoss()
        elif args.loss_function == 'torch_softmax_soft_margin_w_custom_weights':
            criterion = nn.MultiLabelSoftMarginLoss(weight = torch.tensor([3.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 3.0, 1.0]))
        elif args.loss_function == 'softmax_margin_w/o_weights':
            criterion = nn.MultiLabelMarginLoss()
        elif args.loss_function == 'bce_with_logits':
            criterion = nn.BCEWithLogitsLoss()
        elif args.loss_function == 'rank_bce':
            cooccur = compute_cooccurrence(train_loader, len(class_list))
            criterion = CorrelationAwareLoss(cooccur, base_loss=nn.BCEWithLogitsLoss(), alpha=0.3)
        else:
            raise ValueError(f"Unsupported loss function '{args.loss_function}' for dataset '{args.dataset}'")

        eval_metrics ={
            'mAP': MultilabelAveragePrecision(num_labels=num_classes, average='macro'),
            'Acc': MultilabelAccuracy(num_labels=num_classes, average='macro'),
            'weighted Accuracy': MultilabelAccuracy(num_labels=num_classes, average='micro'),
            'weighted v1 mAP': MultilabelAveragePrecision(num_labels=num_classes, average='micro')
        }

    # modeltest_transforms
    model_args = (train_loader, val_loader, val_loader, criterion, eval_metrics, class_list, args.test_every, args.distributed, device, args.max_steps, args.gradient_accumulation_steps, logger)
    if args.model == 'timesformer':
        from classifier import TimeSformerExecutor
        executor = TimeSformerExecutor(*model_args)
    elif args.model == 'videomae':
        from classifier import VideoMaeExecutor
        executor = VideoMaeExecutor(*model_args)    
    elif args.model == 'timesformerclip':
        from classifier import TimeSformerCLIPInitExecutor
        executor = TimeSformerCLIPInitExecutor(*model_args)  
    # executor.model.to(device)
    logger.info('Start training')
    executor.run_training()
    try:
        eval = executor.test()
    except:
        pass    
    
    logger.info(f"MultiLabel Average Precision: {float(eval['mAP']) * 100:.3f}")
    logger.info(f"MultiLabel Accuracy: {float(eval['Acc']) * 100:.3f}")
    logger.info(f"Mean Average Precision: {eval['weighted v1 mAP'] * 100:.3f}")
    executor.save(f"ds_{str(args.dataset)}_model_{str(args.model)}_nepochs_{str(args.epochs)}_nframes_{str(args.total_length)}.pth")
    
    
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
    # parser.add_argument("--test_part", default=6, type=int, help="Test partition for Hockey dataset")
    # parser.add_argument("--zero_shot", default=False, type=bool, help="Zero-shot or Fully supervised")
    # parser.add_argument("--split", default=1, type=int, help="Split 1: 50:50, Split 2: 75:25")
    parser.add_argument("--train", default=True, type=bool, help="train or test")
    # parser.add_argument("--animal", default="", help="Animal subset of data to use.")
    parser.add_argument("--loss_function", default = "rank_bce", help = "Los function")
    args = parser.parse_args()
    
    main(args)