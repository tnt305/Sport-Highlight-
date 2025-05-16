import os
import time
import sys
from tqdm import tqdm
import torch
from torch import nn
from torch.optim import Adam, AdamW

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))  # Lên 3 cấp đến v5
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from short_sport.video_trainer.utils import AverageMeter
from transformers import TimesformerModel, logging
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import warnings
from short_sport.video_trainer.losses.loss import AsymmetricLossOptimized, CalibratedRankingLoss
from torch.amp import autocast , GradScaler
from loguru import logger
from typing import Any
warnings.filterwarnings("ignore")

logger.remove()
logger.add('timesformer_log.log', format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level='INFO')

class TimesformerBase(nn.Module):
    def __init__(self, num_frames, num_classes=18):
        super().__init__()
        self.num_classes = num_classes
        # "facebook/timesformer-base-finetuned-k600"
        self.backbone = TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k600",
                                                        num_frames=num_frames,
                                                        ignore_mismatched_sizes=True)
        self.classifier = (
            nn.Linear(768, num_classes)
        )
        # self.temperature  = nn.Parameter(torch.ones(1))
    def forward(self, images):
        x = self.backbone(images)[0]
        x = self.classifier(x[:, 0])
        # x = x/self.temperature 
        return x

class TimeSformerExecutor:
    def __init__(self, train_loader, 
                 test_loader, 
                 criterion, 
                 eval_metrics, 
                 class_list, 
                 test_every,
                 learning_rate,
                 distributed, 
                 gpu_id, 
                 logger) -> None:
        super().__init__()
        self.train_loader = train_loader
        # self.valid_loader = valid_loader  # Corrected reference
        self.test_loader = test_loader
        self.criterion = criterion.to(gpu_id)
        self.eval_metrics = {name: metric.to(gpu_id) for name, metric in eval_metrics.items()}
        self.class_list = class_list
        self.test_every = test_every
        self.distributed = distributed
        self.gpu_id = gpu_id
        num_frames = self.train_loader.dataset[0][0].shape[0]
        num_classes = len(class_list)
        logging.set_verbosity_error()
        model = TimesformerBase(num_frames=num_frames, num_classes=num_classes).to(gpu_id)
        if distributed: 
            self.model = DDP(model, device_ids=[gpu_id])
        else: 
            self.model = model
        for p in self.model.parameters():
            p.requires_grad = True
        self.lr = learning_rate
        self.optimizer = Adam([{"params": self.model.parameters(), "lr": self.lr}])
        self.scheduler = CosineAnnealingWarmRestarts(optimizer = self.optimizer, T_0=10)
        
        self.logger = logger
        
    def get_model(self):
            return self.model.module if self.distributed else self.model
    
    def _train_batch(self, data, label):
        self.optimizer.zero_grad()
        output = self.model(data)
        loss_this = self.criterion(output, label.long())
        loss_this.backward()
        self.optimizer.step()
        return loss_this.item()

    def _train_epoch(self, epoch):
        self.model.train()
        loss_meter = AverageMeter()
        start_time = time.time()
        for data, label in tqdm(self.train_loader):
            data, label = data.to(self.gpu_id), label.to(self.gpu_id)
            loss_this = self._train_batch(data, label)
            loss_meter.update(loss_this, data.shape[0])
        elapsed_time = time.time() - start_time
        if (self.distributed and self.gpu_id == 0) or not self.distributed:
            self.logger.info(
                f"Epoch [{epoch + 1}][{time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}] "
                f"loss: {loss_meter.avg:.4f}"
            )
    
    def train(self, start_epoch, end_epoch):
        
        self.logger.info("Unfrozen backbone")
        for param in self.get_model().backbone.parameters():
            param.requires_grad = True
        
        for epoch in range(start_epoch, end_epoch):
            self._train_epoch(epoch)
            if (epoch + 1) % self.test_every == 0:
                metrics_values = self.test()
                if (self.distributed and self.gpu_id == 0) or not self.distributed:
                    for metric_name, value in metrics_values.items():
                        self.logger.info(f"[INFO] {metric_name}: {value * 100:.2f}")
    
    def test(self):
        self.model.eval()
        metric_meters = {name: AverageMeter() for name in self.eval_metrics.keys()}
        
        for data, label in tqdm(self.test_loader):
            data, label = data.to(self.gpu_id), label.long().to(self.gpu_id)
            with torch.no_grad():
                output = self.model(data)
                
            # Update all metrics
            for name, metric in self.eval_metrics.items():
                metric_value = metric(output, label)
                metric_meters[name].update(metric_value.item(), data.shape[0])
        return {name: meter.avg for name, meter in metric_meters.items()}
    
    def save(self, file_path="./checkpoint.pth"):
        backbone_state_dict = self.model.backbone.state_dict()
        classifier_state_dict = self.model.classifier.state_dict()
        optimizer_state_dict = self.optimizer.state_dict()
        torch.save({"backbone": backbone_state_dict,
                    "classifier": classifier_state_dict,
                    "optimizer": optimizer_state_dict},
                    file_path)
        
    def load(self, file_path):
        checkpoint = torch.load(file_path)
        self.model.backbone.load_state_dict(checkpoint["backbone"])
        self.model.transformer.load_state_dict(checkpoint["classifier"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])