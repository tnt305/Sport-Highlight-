import os
import time
from tqdm import tqdm
import torch
from torch import nn
from torch.optim import Adam, AdamW
from transformers import VideoMAEForVideoClassification, logging
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from short_sport.video_trainer.utils import AverageMeter
from short_sport.video_trainer.labels import EVENT_DICTIONARY_V2, INVERSE_EVENT_DICTIONARY_V2

import warnings
warnings.filterwarnings("ignore")

from src.logger import Logging
logger = Logging()

class VideoMae(nn.Module):
    def __init__(self, num_frames, num_classes = 18):
        super().__init__()
        # self.num_classes = num_classes
        #"facebook/timesformer-base-finetuned-k600"
        # MCG-NJU/videomae-large-finetuned-kinetics
        label2id = EVENT_DICTIONARY_V2
        id2label = INVERSE_EVENT_DICTIONARY_V2
        self.backbone = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics", 
                                                        num_frames=num_frames,
                                                        label2id = label2id,
                                                        id2label = id2label, 
                                                        ignore_mismatched_sizes=True)
        # self.classifier = nn.Linear(in_features=self.backbone.config.hidden_size, out_features=num_classes)
        # self.classifier = nn.Sequential(
        #         nn.LayerNorm(self.backbone.config.hidden_size),
        #         nn.Dropout(p=0.5),
        #         nn.Linear(self.backbone.config.hidden_size, 512),
        #         nn.GELU(),
        #         nn.Dropout(p=0.3),
        #         nn.Linear(512, self.num_classes)
        # )         #Thiên Đặng sửa backbone [INFO] average precision: 46.31 accuracy: 80.69 mean average precision: 67.29
    def forward(self, images):
        x = self.backbone(images)[0]
        return x
    
class VideoMaeExecutor:
    def __init__(self, 
                train_loader, 
                test_loader, 
                criterion, 
                eval_metrics, 
                class_list, 
                test_every, 
                distributed, 
                gpu_id,
                logger) -> None:
        super().__init__()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion.to(gpu_id)
        self.eval_metrics = {name: metric.to(gpu_id) for name, metric in eval_metrics.items()} #eval_metric.to(gpu_id)
        self.class_list = class_list
        self.test_every = test_every
        self.distributed = distributed
        self.gpu_id = gpu_id
        num_frames = self.train_loader.dataset[0][0].shape[0]
        num_classes = len(class_list)
        self.logger = logger
        # logging.set_verbosity_error()
        model = VideoMae(num_frames=num_frames, num_classes=num_classes).to(gpu_id)
        if distributed: 
            self.model = DDP(model, device_ids=[gpu_id])
        else: 
            self.model = model
            
        for p in self.model.parameters():
            p.requires_grad = True
        
        if isinstance(criterion, nn.CrossEntropyLoss):
            # Phân loại rời rạc
            self.label_type = torch.long
        elif isinstance(criterion, nn.BCEWithLogitsLoss):
            # Phân loại đa nhãn
            self.label_type = torch.float
        
        self.optimizer = Adam([{"params": self.model.parameters(), "lr": 0.00005}])
        self.scheduler = CosineAnnealingWarmRestarts(optimizer = self.optimizer, T_0=10)
    
    def get_model(self):
        return self.model.module if self.distributed else self.model
    
    def _train_batch(self, data, label):
        self.optimizer.zero_grad()
        output = self.model(data)
        label  = label.float()
        loss_this = self.criterion(output, label)
        loss_this.backward()
        self.optimizer.step()
        return loss_this.item()

    def _train_epoch(self, epoch):
        self.model.train()
        loss_meter = AverageMeter()
        start_time = time.time()
        for data, label in tqdm(self.train_loader, desc=f"Epoch {epoch + 1}", unit="batch"):
            data, label = data.to(self.gpu_id), label.to(self.gpu_id) #long
            loss_this = self._train_batch(data, label) # float
            loss_meter.update(loss_this, data.shape[0])
        elapsed_time = time.time() - start_time
        
        self.scheduler.step()
        
        if (self.distributed and self.gpu_id == 0) or not self.distributed:
            self.logger.info(f"Epoch [{epoch + 1}][{time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}] loss: {loss_meter.avg:.4f}")
    
    def train(self, start_epoch, end_epoch):
        logger.info("Unfrozen backbone")
        for param in self.get_model().backbone.parameters():
            param.requires_grad = True
            
        for epoch in tqdm(range(start_epoch, end_epoch), desc = 'start training ...'):
            self.logger.info("Epoch at --> str(epoch + 1) + ")
            self._train_epoch(epoch)
            if (epoch + 1) % self.test_every == 0:
                metrics_values = self.test()
                if (self.distributed and self.gpu_id == 0) or not self.distributed:
                    for metric_name, value in metrics_values.items():
                        self.logger.info("[INFO] " + metric_name + ": " + "{:.2f}".format(value * 100))
    
    def test(self):
        self.model.eval()
        
        metric_meters = {name: AverageMeter() for name in self.eval_metrics.keys()}
        for data, label in tqdm(self.test_loader):
            data, label = data.to(self.gpu_id), label.to(self.gpu_id)
            label = label.long()
            with torch.no_grad():
                output = self.model(data)
                
            # Update all metrics
            for name, metric in self.eval_metrics.items():
                metric_value = metric(output, label)
                metric_meters[name].update(metric_value.item(), data.shape[0])
        
        return {name: meter.avg for name, meter in metric_meters.items()}
    
    def save(self, file_path="./checkpoint.pth"):
        
        if not os.path.exists("./models/"):
            os.makedirs("./models/")
        if self.distributed:
            backbone_state_dict = self.model.module.backbone.state_dict()
        else:
            backbone_state_dict = self.model.backbone.state_dict()
            
        if self.optimizer:
            optimizer_state_dict = self.optimizer.state_dict()
        else:
            optimizer_state_dict = None
            
        # Lưu checkpoint
        checkpoint = {
            "backbone": backbone_state_dict,
            "optimizer": optimizer_state_dict
        }
        torch.save(checkpoint, file_path)
        
    def load(self, file_path):
        checkpoint = torch.load(file_path)
        self.model.backbone.load_state_dict(checkpoint["backbone"])
        self.model.transformer.load_state_dict(checkpoint["classifier"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        
    # def inference(self, video_path: str):
    #     video_frames = os.listdir(video_path)
    #     self.model.eval()
        
    #     device = self.gpu_id
        
    #     if len(video_frames.shape) == 4:
    #         video_frames = video_frames.unsqueeze(0)
    #     video_frames = video_frames.to(device)

    #     with torch.no_grad():
    #         outputs=  self.model(video_frames)
            