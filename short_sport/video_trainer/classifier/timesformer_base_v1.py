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

# from src.logger.logger import Logging
# logger = Logging()
        # self.classifier = nn.Sequential(
        #         nn.LayerNorm(self.backbone.config.hidden_size),
        #         nn.Dropout(p=0.5),
        #         nn.Linear(self.backbone.config.hidden_size, 512),
        #         nn.GELU(),
        #         nn.Dropout(p=0.3),
        #         nn.Linear(512, self.num_classes)
        # )         #Thiên Đặng sửa backbone [INFO] average precision: 46.31 accuracy: 80.69 mean average precision: 67.29
class TimeSformer(nn.Module):
    def __init__(self, num_frames, num_classes=18):
        super().__init__()
        self.num_classes = num_classes
        # "facebook/timesformer-base-finetuned-k600"
        self.backbone = TimesformerModel.from_pretrained("facebook/timesformer-hr-finetuned-k600",
                                                        num_frames=num_frames,
                                                        ignore_mismatched_sizes=True)
        self.classifier = (
            nn.Linear(768, num_classes)
        )
        self.temperature  = nn.Parameter(torch.ones(1))
    def forward(self, images):
        x = self.backbone(images)[0]
        x = self.classifier(x[:, 0])
        x = x/self.temperature 
        return x
    

class TimeSformerExecutor:
    def __init__(self, train_loader, valid_loader, test_loader, criterion, eval_metrics, class_list, test_every, distributed, gpu_id, max_steps, gradient_accumulation_steps, logger) -> None:
        super().__init__()
        self.train_loader = train_loader
        self.valid_loader = valid_loader  # Corrected reference
        self.test_loader = test_loader
        self.criterion = criterion.to(gpu_id)
        self.eval_metrics = {name: metric.to(gpu_id) for name, metric in eval_metrics.items()}
        self.class_list = class_list
        self.test_every = test_every
        self.distributed = distributed
        self.gpu_id = gpu_id
        num_frames = self.train_loader.dataset[0][0].shape[0]
        num_classes = len(class_list)
        self.logger = logger or self._create_default_logger()  
        self.max_steps = max_steps
        self.test_every_steps = int(round(self.max_steps / self.test_every))
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.global_step = 0
        logging.set_verbosity_error()
        
        model = TimeSformer(num_frames=num_frames, num_classes=num_classes).to(gpu_id)
        if distributed:
            self.model = DDP(model, device_ids=[gpu_id])
        else:
            self.model = model
            
        for p in self.model.backbone.parameters():
            p.requires_grad = True
            
        self.optimizer = Adam([{"params": self.model.parameters(), "lr": 0.0001}])
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=int(self.max_steps * 0.1),
            num_training_steps=self.max_steps
        )
        self.scaler = GradScaler()
    
    def _create_default_logger(name: str = "TimeSformer"):
        """
        Tạo logger mặc định sử dụng loguru nếu chưa được cấu hình.

        Returns:
            logger: Instance của loguru logger với cấu hình mặc định.
        """
        # Xóa các sink (handlers) hiện tại để tránh trùng lặp
        logger.remove()

        # Định dạng log tương tự như logging
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<cyan>TimeSformer</cyan> | "
            "<level>{level: <8}</level> | "
            "<level>{message}</level>"
        )

        # Thêm sink mặc định (stdout) với định dạng và level INFO
        logger.add(
            sink=sys.stdout,
            format=log_format,
            level="INFO",
            colorize=True,
            backtrace=True,
            diagnose=True
        )

        return logger
    def get_model(self):
        return self.model.module if self.distributed else self.model

    def _train_epoch(self, epoch):
        self.model.train()
        loss_meter = AverageMeter()
        
        for data, label in tqdm(self.train_loader, desc=f"Epoch {epoch + 1}", unit="batch"):
            data, label = data.to(self.gpu_id), label.to(self.gpu_id)
            loss_this = self._train_batch(data, label)
            self.logger.info(f"loss: {loss_this}")
            loss_meter.update(loss_this, data.shape[0])
        
        self.scheduler.step()
        self.logger.info(f"Epoch {epoch + 1} - Average Loss: {loss_meter.avg}")
        
    def _train_step(self, data, label, batch_idx):
        with autocast(device_type='cuda', dtype=torch.float16):
            # Forward pass
            output = self.model(data)
            label = label.float()
            loss_this = self.criterion(output, label)
            
            # Scale the loss by the number of accumulation steps
            loss_this = loss_this / self.gradient_accumulation_steps
            
        # Backward pass with scaled gradients
        self.scaler.scale(loss_this).backward()
        
        # Only update weights after accumulating gradients for specified steps
        if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
            # Unscale gradients before optimizer step
            self.scaler.unscale_(self.optimizer)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights with scaled gradients
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.scheduler.step()
            self.global_step += 1
            
        return loss_this.item() * self.gradient_accumulation_steps  # Return the unscaled loss for logging
        
    def train(self):
        self.logger.info("Unfrozen backbone for Timesformer Training")
        for param in self.get_model().backbone.parameters():
            param.requires_grad = True
        
        self.logger.info(f"Starting training for {self.max_steps} steps with gradient accumulation={self.gradient_accumulation_steps}")
        self.model.train()
        loss_meter = AverageMeter()
        
        # Ensure temperature parameter exists
        if not hasattr(self.get_model(), 'temperature'):
            self.get_model().temperature = nn.Parameter(torch.ones(1).to(self.gpu_id))
        
        # Initialize optimizer gradients
        self.optimizer.zero_grad()
        
        # Create iterators for data loaders for infinite iteration
        train_iterator = iter(self.train_loader)
        batch_idx = 0
        
        with tqdm(total=self.max_steps, desc='Training') as pbar:
            while self.global_step < self.max_steps:
                try:
                    data, label = next(train_iterator)
                except StopIteration:
                    # Restart iterator when epoch ends
                    train_iterator = iter(self.train_loader)
                    data, label = next(train_iterator)
                    self.logger.info(f"Restarting data iterator at step {self.global_step}")
                
                data, label = data.to(self.gpu_id), label.to(self.gpu_id)
                loss_this = self._train_step(data, label, batch_idx)
                
                # Log information
                current_lr = self.optimizer.param_groups[0]['lr']
                loss_meter.update(loss_this, data.shape[0])
                
                # Update batch index
                batch_idx += 1
                
                # Only update progress bar when optimization step is completed
                if batch_idx % self.gradient_accumulation_steps == 0:
                    pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", lr=f"{current_lr:.6f}")
                    pbar.update(1)
                    
                    # Detailed logging
                    if self.global_step % 1000 == 0:
                        self.logger.info(f"Step {self.global_step}/{self.max_steps} - Loss: {loss_meter.avg:.4f} - LR: {current_lr:.6f}")
                
                    # Periodic model evaluation
                    if self.global_step % self.test_every_steps == 0:
                        avg_loss = loss_meter.avg
                        self.logger.info(f"Step {self.global_step} - Average Loss: {avg_loss:.4f}")
                        loss_meter.reset()
                        
                        # Switch to evaluation mode
                        self.model.eval()
                        
                        # 1. First evaluate without calibration
                        self.logger.info("Evaluating model without calibration...")
                        metrics_values_raw = self.test(apply_calibration=False)
                        
                        if (self.distributed and self.gpu_id == 0) or not self.distributed:
                            for metric_name, value in metrics_values_raw.items():
                                self.logger.info(f"[RAW] {metric_name}: {value * 100:.2f}")
                        
                        # 2. Perform temperature calibration
                        # self.logger.info("Performing temperature calibration...")
                        # optimal_temp = self.calibrate_temperature(self.valid_loader)
                        
                        # 3. Evaluate model after calibration
                        # self.logger.info(f"Evaluating model with calibration (temp={optimal_temp:.4f})...")
                        # metrics_values = self.test(apply_calibration=True)
                        
                        # if (self.distributed and self.gpu_id == 0) or not self.distributed:
                        #     for metric_name, value in metrics_values.items():
                        #         self.logger.info(f"[CALIBRATED] {metric_name}: {value * 100:.2f}")
                        
                        # 4. Save checkpoint
                        self.save(f"./models/timesformer_step_{self.global_step}.pth")
                        
                        # 5. Return to training mode
                        self.model.train()
                        
    def calibrate_temperature(self, valid_loader, max_iter=100):
        """
        Hiệu chỉnh nhiệt độ sử dụng thuật toán LBFGS
        """
        self.model.eval()
        model = self.get_model()
        
        # Đảm bảo có tham số nhiệt độ
        if not hasattr(model, 'temperature'):
            model.temperature = nn.Parameter(torch.ones(1).to(self.gpu_id))
        
        # Lưu giá trị nhiệt độ ban đầu
        original_temp = model.temperature.item()
        
        # Chuẩn bị dữ liệu validation
        all_logits = []
        all_labels = []
        
        with torch.no_grad(), autocast(device_type='cuda', dtype=torch.float16):
            for data, labels in tqdm(valid_loader):
                data, labels = data.to(self.gpu_id), labels.to(self.gpu_id)
                
                # Lấy logits gốc
                logits = model.backbone(data)[0]
                logits = model.classifier(logits[:, 0])
                
                all_logits.append(logits.cpu())
                all_labels.append(labels.float().cpu())
        
        # Ghép dữ liệu
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Định nghĩa hàm mất mát cho LBFGS
        def closure():
            optimizer.zero_grad()
            
            # Áp dụng temperature scaling
            scaled_logits = all_logits.to(self.gpu_id) / model.temperature
            
            # Sử dụng AsymmetricLossOptimized
            loss_fn =  CalibratedRankingLoss().to(self.gpu_id)
            
            loss = loss_fn(scaled_logits, all_labels.to(self.gpu_id))
            loss.backward()
            
            return loss
        
        # Khởi tạo LBFGS optimizer
        optimizer = torch.optim.LBFGS(
            [model.temperature], 
            max_iter=max_iter,
            line_search_fn='strong_wolfe'  # Chọn phương pháp tìm kiếm đường
        )
        
        # Thực hiện tối ưu hóa
        best_loss = float('inf')
        best_temp = original_temp
        
        try:
            # LBFGS yêu cầu gọi closure nhiều lần
            loss = optimizer.step(closure)
            
            # Kiểm tra và lưu kết quả tốt nhất
            current_temp = model.temperature.item()
            current_loss = loss.item()
            
            if current_loss < best_loss:
                best_loss = current_loss
                best_temp = current_temp
            
            self.logger.info(f"LBFGS Calibration: Original temp={original_temp:.4f}, New temp={best_temp:.4f}, Loss={best_loss:.6f}")
        
        except Exception as e:
            self.logger.warning(f"LBFGS Calibration failed: {str(e)}")
            # Quay về giá trị mặc định nếu thất bại
            best_temp = 1.0
        
        # Đặt lại nhiệt độ
        with torch.no_grad():
            model.temperature.copy_(torch.tensor([best_temp]).to(self.gpu_id))
        
        return best_temp
            
    def test(self, apply_calibration=False):
        self.model.eval()
        
        metric_meters = {name: AverageMeter() for name in self.eval_metrics.keys()}
        
        for data, label in tqdm(self.test_loader, desc='Evaluation'):
            label = label.long()  # Use float for multilabel
            data, label = data.to(self.gpu_id), label.to(self.gpu_id)
            
            with torch.no_grad(), autocast(device_type='cuda', dtype=torch.float16):
                output = self.model(data)
                
                # Apply temperature scaling if requested
                if not apply_calibration:
                    # Reset temperature to 1.0 temporarily for raw predictions
                    model = self.get_model()
                    original_temp = model.temperature.item()
                    with torch.no_grad():
                        model.temperature.copy_(torch.tensor([1.0]).to(self.gpu_id))
                    output = self.model(data)
                    # Restore original temperature
                    with torch.no_grad():
                        model.temperature.copy_(torch.tensor([original_temp]).to(self.gpu_id))
                
            # Update all metrics
            for name, metric in self.eval_metrics.items():
                metric_value = metric(output, label)
                metric_meters[name].update(metric_value.item(), data.shape[0])
        
        return {name: meter.avg for name, meter in metric_meters.items()}
    
    def save(self, file_path="./checkpoint.pth"):
        if not os.path.exists('./models'):
            os.makedirs('./models', exist_ok=True)
            
        model = self.get_model()
        
        checkpoint = {
            "backbone": model.backbone.state_dict(),
            "classifier": model.classifier.state_dict(),
            "temperature": model.temperature.item(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
        }
        
        torch.save(checkpoint, file_path)
        
    def load(self, file_path):
        checkpoint = torch.load(file_path, map_location=f"cuda:{self.gpu_id}")
        model = self.get_model()
        
        model.backbone.load_state_dict(checkpoint["backbone"])
        model.classifier.load_state_dict(checkpoint["classifier"])
        
        # Load temperature if it exists in checkpoint
        if "temperature" in checkpoint:
            with torch.no_grad():
                model.temperature.copy_(torch.tensor([checkpoint["temperature"]]).to(self.gpu_id))
                
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        
        # Load global step if available
        if "global_step" in checkpoint:
            self.global_step = checkpoint["global_step"]
            