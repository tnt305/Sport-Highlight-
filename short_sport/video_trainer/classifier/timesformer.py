import os
import time
import sys
import numpy as np
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
from short_sport.video_trainer.architecture.sub_modules.matrix import compute_cooccurrence
from scipy.sparse import csr_matrix
from torch_geometric.nn import GATConv  # Cần cài đặt torch-geometric

class SafeGATWrapper(nn.Module):
    def __init__(self, in_channels, out_channels, heads):
        super().__init__()
        self.gat = GATConv(in_channels, out_channels, heads=heads)
        
        # Tắt biên dịch động cho lớp này
        torch._dynamo.mark_dynamic(self.gat, 0)

    def forward(self, x, edge_index):
        # Thêm kiểm tra an toàn
        if edge_index.numel() > 0:  # Kiểm tra edge_index không rỗng
            assert edge_index.max() < x.size(0), "Edge index out of bounds"
            return self.gat(x, edge_index)
        else:
            # Trả về tensor zeros nếu không có edge
            return torch.zeros(x.size(0), self.gat.out_channels * self.gat.heads, device=x.device)


class TimeSformer(nn.Module):
    def __init__(self, num_frames, num_classes=18, train_loader=None):
        super().__init__()
        self.num_classes = num_classes
        # Khởi tạo backbone
        self.backbone = TimesformerModel.from_pretrained("facebook/timesformer-hr-finetuned-k600",
                                                        num_frames=num_frames,
                                                        ignore_mismatched_sizes=True)
        
        # Khởi tạo ma trận đồng xuất hiện và GNN
        if train_loader is not None:
            self.cooccur = self._compute_cooccurrence(train_loader, num_classes)
            self.edge_index = self._create_graph_edges(num_classes)
        else:
            # Fallback khi không có train_loader
            self.cooccur = torch.zeros((num_classes, num_classes))
            self.edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        hidden_size = self.backbone.config.hidden_size
        self.gnn = SafeGATWrapper(hidden_size, 256, heads=2)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size + 256),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size+ 256, hidden_size//2),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_size//2, num_classes)
        )
        
        # Thêm temperature scaling cho hiệu chỉnh
        self.temperature = nn.Parameter(torch.ones(1))
        
        # Thêm dynamic thresholds
        self.register_buffer('thresholds', torch.ones(num_classes) * 0.5)
        
        # Tắt biên dịch động cho forward
        torch._dynamo.mark_dynamic(self, 0)
    
    def _compute_cooccurrence(self, loader, num_classes):
        """Tính toán ma trận đồng xuất hiện từ dữ liệu"""
        try:
            return compute_cooccurrence(loader, num_classes)
        except Exception as e:
            print(f"Lỗi khi tính ma trận đồng xuất hiện: {e}")
            return torch.zeros((num_classes, num_classes))

    def _create_graph_edges(self, num_classes):
        """Tạo edge index với kiểm tra biên"""
        try:
            # Chuyển đổi sang tensor nếu là ma trận thưa
            if isinstance(self.cooccur, csr_matrix):
                rows, cols = self.cooccur.nonzero()
            else:
                rows, cols = self.cooccur.nonzero()
                rows, cols = rows.cpu().numpy(), cols.cpu().numpy()
            
            if len(rows) == 0:
                # Trả về tensor rỗng nếu không có cạnh
                return torch.zeros((2, 0), dtype=torch.long)
            
            edge_index = torch.tensor([rows, cols], dtype=torch.long)
            assert edge_index.max() < num_classes, "Edge index exceeds num_classes"
            return edge_index
        except Exception as e:
            print(f"Lỗi khi tạo edge index: {e}")
            return torch.zeros((2, 0), dtype=torch.long)

    def forward(self, x):
        # Feature extraction
        outputs = self.backbone(x)
        x = outputs[0][:, 0]  # [batch_size, hidden_size]
        
        # GNN processing với kiểm tra an toàn
        batch_size = x.size(0)
        
        # Tạo batch nodes cho GNN (mỗi batch là một graph riêng)
        batch_x = x.clone()  # [batch_size, hidden_size]
        
        # Sử dụng edge_index đã được tạo trước
        if self.edge_index.numel() > 0:
            edge_index = self.edge_index.to(x.device)
            
            # Kiểm tra tính hợp lệ của edge_index
            if edge_index.shape[1] > 0 and edge_index.max() < batch_size:
                gnn_feat = self.gnn(batch_x, edge_index)
                gnn_pooled = gnn_feat.mean(dim=0, keepdim=True).expand(batch_size, -1)
            else:
                # Fallback khi edge_index không hợp lệ
                gnn_pooled = torch.zeros(batch_size, 256, device=x.device)
        else:
            # Fallback khi không có edge
            gnn_pooled = torch.zeros(batch_size, 256, device=x.device)
        
        # Kết hợp features
        combined = torch.cat([x, gnn_pooled], dim=1)
        logits = self.classifier(combined)
        
        # Áp dụng temperature scaling trong quá trình inference
        if not self.training:
            logits = logits / self.temperature
        
        return logits
    
    def predict_with_threshold(self, x):
        """Dự đoán với threshold động"""
        with torch.no_grad():
            logits = self(x)
            probs = torch.sigmoid(logits)
            preds = (probs > self.thresholds).float()
            return preds, probs

class TimeSformerExecutor:
    def __init__(self, train_loader, valid_loader, test_loader, criterion, eval_metrics, 
                 class_list, test_every, distributed, gpu_id, max_steps, 
                 gradient_accumulation_steps, logger=None) -> None:
        
        # Tắt biên dịch động cho toàn bộ quá trình huấn luyện
        torch._dynamo.config.suppress_errors = True
        
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.gpu_id = gpu_id
        self.criterion = criterion.to(self.gpu_id)
        self.eval_metrics = {name: metric.to(self.gpu_id) for name, metric in eval_metrics.items()}
        self.class_list = class_list
        self.test_every = test_every
        self.distributed = distributed
        self.max_steps = max_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.global_step = 0
        self.current_step = 0
        self.logger = logger or self._create_default_logger()
        
        # Giảm verbosity của transformers
        logging.set_verbosity_error()
        
        # Khởi tạo model
        num_frames = self.train_loader.dataset[0][0].shape[0]
        num_classes = len(class_list)
        
        model = TimeSformer(num_frames=num_frames, num_classes=num_classes, train_loader=train_loader).to(self.gpu_id)
        
        if distributed:
            self.model = DDP(model, device_ids=[gpu_id])
        else:
            self.model = model
            
        # Fine-tuning toàn bộ backbone
        for p in self.model.backbone.parameters():
            p.requires_grad = True
            
        # Optimizer và scheduler
        self.optimizer = AdamW([
            {"params": self.model.parameters(), "lr": 0.00001, 'weight_decay': 0.01}
        ])
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=int(self.max_steps * 0.1),
            num_training_steps=self.max_steps
        )
        
        # Gradient scaler cho mixed precision
        self.scaler = GradScaler()
        
        # Iterator cho training
        self.train_iter = iter(train_loader)
        
        # Metrics
        self.best_map = 0.0
        
        # Flag để kiểm soát tần suất hiệu chỉnh temperature
        self.last_temp_calibration = 0
    
    def _create_default_logger(self):
        """Tạo logger mặc định nếu không được cung cấp"""
        log_format = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
        logger.configure(handlers=[{"sink": sys.stderr, "format": log_format}])
        return logger
    
    def get_model(self):
        """Trả về model cơ bản (không phải DDP wrapper)"""
        if self.distributed:
            return self.model.module
        else:
            return self.model
    
    def _get_next_batch(self):
        """Lấy batch tiếp theo từ iterator, reset nếu cần"""
        try:
            return next(self.train_iter)
        except StopIteration:
            self.train_iter = iter(self.train_loader)
            return next(self.train_iter)

    def train_step(self):
        """Thực hiện một bước huấn luyện"""
        self.model.train()
        total_loss = 0
        self.optimizer.zero_grad()
        
        for _ in range(self.gradient_accumulation_steps):
            data, labels = self._get_next_batch()
            data, labels = data.to(self.gpu_id), labels.to(self.gpu_id)
            
            # Forward với autocast
            with autocast(device_type="cuda", dtype=torch.float16):
                outputs = self.model(data)
                loss = self.criterion(outputs, labels.long()) / self.gradient_accumulation_steps
            
            # Backward accumulation với scaler
            self.scaler.scale(loss).backward()
            total_loss += loss.item()
        
        # Gradient clipping
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        # Optimizer step với scaler
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        self.scheduler.step()
        self.current_step += 1
        self.global_step += 1
        
        # Validation định kỳ
        if self.current_step % 500 == 0:
            # Luôn validate và cập nhật thresholds
            self._validate()
            
            # Chỉ hiệu chỉnh temperature theo định kỳ ít hơn
            if self.current_step % 2000 == 0 or self.current_step == 500:
                print("Calibrating temperature...")
                self._calibrate_temperature()
                self.last_temp_calibration = self.current_step
            
            self.save(f"./models/checkpoint_step{self.current_step}.pth")
            
        return total_loss
    
    def _validate(self):
        """Đánh giá model trên tập validation và cập nhật thresholds"""
        self.model.eval()
        metric_meters = {name: AverageMeter() for name in self.eval_metrics.keys()}
        
        print(f"\n=== Validation at Step {self.current_step} ===")
        
        # Cập nhật Dynamic Thresholds
        print("Updating thresholds...")
        self._update_thresholds()
        
        # Đánh giá metrics
        print("Evaluating metrics...")
        with torch.no_grad(), autocast(device_type="cuda", dtype=torch.float16):
            for data, labels in tqdm(self.valid_loader, desc="Validation"):
                labels = labels.to(self.gpu_id)
                data, labels = data.to(self.gpu_id), labels.to(self.gpu_id)
                outputs = self.model(data)
                
                # Áp dụng thresholds cho multi-label
                probs = torch.sigmoid(outputs)
                model = self.get_model()
                preds = (probs > model.thresholds).float()
                
                # Update all metrics
                for name, metric in self.eval_metrics.items():
                    if name == 'accuracy':  # Xử lý riêng cho accuracy
                        metric_value = metric(preds, labels)
                    else:
                        try:
                            metric_value = metric(outputs, labels)  # AP dùng logits
                        except:
                            print("AP trong _validate_and_calibrate dùng logits")
                            metric_value = metric(outputs, labels.long())  # AP dùng logits
                    metric_meters[name].update(metric_value.item(), data.size(0))
        
        # Log metrics
        current_metrics = {name: meter.avg for name, meter in metric_meters.items()}
        print(f"=== Validation Results ===")
        for name, value in current_metrics.items():
            print(f"{name}: {value:.4f}")
        
        # Lưu best model
        if current_metrics.get('average_precision', 0) > self.best_map:
            old_best = self.best_map
            self.best_map = current_metrics['average_precision']
            print(f"New best average_precision: {old_best:.4f} -> {self.best_map:.4f}")
            self.save(f'./models/best_model_step{self.current_step}.pth')
            print(f"Saved best model at step {self.current_step}")
        else:
            print(f"No improvement in average_precision. Current best: {self.best_map:.4f}")

    def _update_thresholds(self, percentile=85):
        """Cập nhật dynamic thresholds dựa trên validation set"""
        probs, truths = [], []
        with torch.no_grad(), autocast(device_type="cuda", dtype=torch.float16):
            for data, labels in tqdm(self.valid_loader, desc="Collecting threshold data"):
                outputs = torch.sigmoid(self.model(data.to(self.gpu_id)))
                probs.append(outputs.cpu())
                truths.append(labels.cpu())
        
        probs = torch.cat(probs)
        truths = torch.cat(truths)
        model = self.get_model()
        
        new_thresholds = []
        for i in range(truths.shape[1]):
            pos_probs = probs[truths[:, i] == 1, i]
            if len(pos_probs) > 0:
                new_thresholds.append(torch.quantile(pos_probs, percentile/100))
            else:
                new_thresholds.append(torch.tensor(0.5))
        
        # Chuyển danh sách thành tensor
        threshold_tensor = torch.stack(new_thresholds).to(self.gpu_id)
        model.thresholds.copy_(threshold_tensor)
        print(f"Thresholds updated. Min: {threshold_tensor.min().item():.4f}, Max: {threshold_tensor.max().item():.4f}")

    def _calibrate_temperature(self):
        """Hiệu chỉnh temperature parameter để cải thiện calibration"""
        model = self.get_model()
        nll_criterion = nn.BCEWithLogitsLoss().to(self.gpu_id)
        
        # Thu thập dữ liệu validation trước khi hiệu chỉnh
        print("Collecting validation data for temperature calibration...")
        logits_list = []
        labels_list = []
        with torch.no_grad(), autocast(device_type="cuda", dtype=torch.float16):
            for data, labels in tqdm(self.valid_loader, desc="Collecting calibration data"):
                data, labels = data.to(self.gpu_id), labels.to(self.gpu_id)
                logits = model(data)
                logits_list.append(logits)
                labels_list.append(labels)
        
        all_logits = torch.cat(logits_list)
        all_labels = torch.cat(labels_list)
        
        # Tối ưu hiệu quả hơn với dữ liệu đã thu thập
        temp_optimizer = torch.optim.LBFGS([model.temperature], lr=0.01, max_iter=20, tolerance_change = 1e-5, tolerance_grad = 1e-6)
        
        def closure():
            temp_optimizer.zero_grad()
            scaled_logits = all_logits / model.temperature
            loss = nll_criterion(scaled_logits, all_labels.float())
            loss.backward()
            return loss
        
        # Hiển thị giá trị temperature trước khi hiệu chỉnh
        print(f"Temperature trước khi hiệu chỉnh: {model.temperature.item():.4f}")
        
        # Thực hiện tối ưu
        temp_optimizer.step(closure)
        
        # Giới hạn temperature trong khoảng hợp lý
        model.temperature.data.clamp_(0.1, 10.0)
        print(f"Temperature sau khi hiệu chỉnh: {model.temperature.item():.4f}")

    def run_training(self):
        """Chạy quá trình huấn luyện và thực hiện final test"""
        progress_bar = tqdm(range(self.max_steps), desc="Training")
        for _ in progress_bar:
            loss = self.train_step()
            progress_bar.set_postfix(
                loss=f"{loss:.4f}", 
                lr=f"{self.scheduler.get_last_lr()[0]:.2e}",
                step=f"{self.current_step}/{self.max_steps}"
            )
        
        # Thực hiện final test sau khi huấn luyện xong
        print("\n=== Running Final Test ===")
        # Đầu tiên sử dụng model hiện tại
        print("Testing current model:")
        current_results = self.test()
        
        # Sau đó nạp best model (nếu có) và test lại
        best_model_path = f'./models/best_model_step{self.current_step}.pth'
        if os.path.exists(best_model_path):
            print(f"\nTesting best model (from step {self.current_step}):")
            self.load(best_model_path)
            best_results = self.test()
            
            # So sánh kết quả
            print("\n=== Results Comparison ===")
            for metric in current_results.keys():
                print(f"{metric}: Current={current_results[metric]:.4f}, Best={best_results[metric]:.4f}")
        
        return current_results
    
    def save(self, file_path="./models/checkpoint.pth"):
        """Lưu checkpoint"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
        model = self.get_model()
        
        checkpoint = {
            "backbone": model.backbone.state_dict(),
            "classifier": model.classifier.state_dict(),
            "gnn": model.gnn.state_dict(),
            "temperature": model.temperature.item(),
            "thresholds": model.thresholds.cpu(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "last_temp_calibration": self.last_temp_calibration
        }
        
        torch.save(checkpoint, file_path)
        print(f"Model saved to {file_path}")
        
    def load(self, file_path):
        """Nạp checkpoint"""
        if not os.path.exists(file_path):
            print(f"Checkpoint {file_path} không tồn tại!")
            return False
            
        checkpoint = torch.load(file_path, map_location=f"cuda:{self.gpu_id}")
        model = self.get_model()
        
        # Nạp các thành phần của model
        model.backbone.load_state_dict(checkpoint["backbone"])
        model.classifier.load_state_dict(checkpoint["classifier"])
        
        # Nạp GNN nếu có
        if "gnn" in checkpoint:
            model.gnn.load_state_dict(checkpoint["gnn"])
        
        # Nạp temperature nếu có
        if "temperature" in checkpoint:
            with torch.no_grad():
                model.temperature.copy_(torch.tensor([checkpoint["temperature"]]).to(self.gpu_id))
        
        # Nạp thresholds nếu có
        if "thresholds" in checkpoint:
            model.thresholds.copy_(checkpoint["thresholds"].to(self.gpu_id))
                
        # Nạp optimizer và scheduler
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        
        # Nạp global step
        if "global_step" in checkpoint:
            self.global_step = checkpoint["global_step"]
            self.current_step = self.global_step
            
        # Nạp last_temp_calibration nếu có
        if "last_temp_calibration" in checkpoint:
            self.last_temp_calibration = checkpoint["last_temp_calibration"]
            
        print(f"Loaded checkpoint from {file_path} (step {self.global_step})")
        return True
    
    def test(self):
        """Đánh giá model trên tập test"""
        self.model.eval()
        metric_meters = {name: AverageMeter() for name in self.eval_metrics.keys()}
        
        all_probs = []
        all_preds = []
        all_labels = []
        
        with torch.no_grad(), autocast(device_type="cuda", dtype=torch.float16):
            for data, labels in tqdm(self.test_loader, desc="Testing"):
                data, labels = data.to(self.gpu_id), labels.to(self.gpu_id)
                outputs = self.model(data)
                
                # Áp dụng temperature scaling và thresholds
                model = self.get_model()
                probs = torch.sigmoid(outputs / model.temperature)
                preds = (probs > model.thresholds).float()
                
                # Lưu kết quả
                all_probs.append(probs.cpu())
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
                
                # Update all metrics
                for name, metric in self.eval_metrics.items():
                    if name == 'accuracy':
                        metric_value = metric(preds, labels)
                    else:
                        metric_value = metric(outputs, labels)
                    metric_meters[name].update(metric_value.item(), data.size(0))
        
        # Tổng hợp kết quả
        all_probs = torch.cat(all_probs)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        
        # In kết quả chi tiết
        print("\n=== Test Results ===")
        for name, meter in metric_meters.items():
            print(f"{name}: {meter.avg:.4f}")
        
        # Phân tích chi tiết theo từng lớp
        print("\n=== Per-Class Analysis ===")
        for i, class_name in enumerate(self.class_list):
            class_precision = (all_preds[:, i] * all_labels[:, i]).sum() / (all_preds[:, i].sum() + 1e-8)
            class_recall = (all_preds[:, i] * all_labels[:, i]).sum() / (all_labels[:, i].sum() + 1e-8)
            class_f1 = 2 * class_precision * class_recall / (class_precision + class_recall + 1e-8)
            
            print(f"{class_name}: P={class_precision:.4f}, R={class_recall:.4f}, F1={class_f1:.4f}, "
                  f"Threshold={model.thresholds[i].item():.4f}")
        
        return {name: meter.avg for name, meter in metric_meters.items()}

# class TimeSformerExecutor:
#     def __init__(self, train_loader, valid_loader, test_loader, criterion, eval_metrics, 
#                  class_list, test_every, distributed, gpu_id, max_steps, 
#                  gradient_accumulation_steps, logger=None) -> None:
        
#         # Tắt biên dịch động cho toàn bộ quá trình huấn luyện
#         torch._dynamo.config.suppress_errors = True
        
#         self.train_loader = train_loader
#         self.valid_loader = valid_loader
#         self.test_loader = test_loader
#         self.gpu_id = gpu_id
#         self.criterion = criterion.to(self.gpu_id)
#         self.eval_metrics = {name: metric.to(self.gpu_id) for name, metric in eval_metrics.items()}
#         self.class_list = class_list
#         self.test_every = test_every
#         self.distributed = distributed
#         self.max_steps = max_steps
#         self.gradient_accumulation_steps = gradient_accumulation_steps
#         self.global_step = 0
#         self.current_step = 0
#         self.logger = logger or self._create_default_logger()
        
#         # Giảm verbosity của transformers
#         logging.set_verbosity_error()
        
#         # Khởi tạo model
#         num_frames = self.train_loader.dataset[0][0].shape[0]
#         num_classes = len(class_list)
        
#         model = TimeSformer(num_frames=num_frames, num_classes=num_classes, train_loader=train_loader).to(self.gpu_id)
        
#         if distributed:
#             self.model = DDP(model, device_ids=[gpu_id])
#         else:
#             self.model = model
            
#         # Fine-tuning toàn bộ backbone
#         for p in self.model.backbone.parameters():
#             p.requires_grad = True
            
#         # Optimizer và scheduler
#         self.optimizer = AdamW([
#             {"params": self.model.parameters(), "lr": 0.00001, 'weight_decay': 0.01}
#         ])
        
#         self.scheduler = get_cosine_schedule_with_warmup(
#             self.optimizer, 
#             num_warmup_steps=int(self.max_steps * 0.1),
#             num_training_steps=self.max_steps
#         )
        
#         # Gradient scaler cho mixed precision
#         self.scaler = GradScaler()
        
#         # Iterator cho training
#         self.train_iter = iter(train_loader)
        
#         # Metrics
#         self.best_map = 0.0
    
#     def _create_default_logger(self):
#         """Tạo logger mặc định nếu không được cung cấp"""
#         log_format = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
#         logger.configure(handlers=[{"sink": sys.stderr, "format": log_format}])
#         return logger
    
#     def get_model(self):
#         """Trả về model cơ bản (không phải DDP wrapper)"""
#         if self.distributed:
#             return self.model.module
#         else:
#             return self.model
    
#     def _get_next_batch(self):
#         """Lấy batch tiếp theo từ iterator, reset nếu cần"""
#         try:
#             return next(self.train_iter)
#         except StopIteration:
#             self.train_iter = iter(self.train_loader)
#             return next(self.train_iter)

#     def train_step(self):
#         """Thực hiện một bước huấn luyện"""
#         self.model.train()
#         total_loss = 0
#         self.optimizer.zero_grad()
        
#         for _ in range(self.gradient_accumulation_steps):
#             data, labels = self._get_next_batch()
#             data, labels = data.to(self.gpu_id), labels.to(self.gpu_id)
            
#             # Forward với autocast
#             with autocast(device_type="cuda", dtype=torch.float16):
#                 outputs = self.model(data)
#                 loss = self.criterion(outputs, labels.long()) / self.gradient_accumulation_steps
            
#             # Backward accumulation với scaler
#             self.scaler.scale(loss).backward()
#             total_loss += loss.item()
        
#         # Gradient clipping
#         self.scaler.unscale_(self.optimizer)
#         torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
#         # Optimizer step với scaler
#         self.scaler.step(self.optimizer)
#         self.scaler.update()
        
#         self.scheduler.step()
#         self.current_step += 1
#         self.global_step += 1
        
#         # Validation định kỳ
#         if self.current_step % 500 == 0:
#             self._validate_and_calibrate()
#             self.save(f"./models/checkpoint_step{self.current_step}.pth")
            
#         return total_loss
    
#     def _validate_and_calibrate(self):
#         """Đánh giá và hiệu chỉnh model trên tập validation"""
#         self.model.eval()
#         metric_meters = {name: AverageMeter() for name in self.eval_metrics.keys()}
        
#         print(f"\n=== Validation at Step {self.current_step} ===")
        
#         # 1. Cập nhật Dynamic Thresholds
#         print("Updating thresholds...")
#         self._update_thresholds()
        
#         # 2. Hiệu chỉnh Temperature
#         print("Calibrating temperature...")
#         self._calibrate_temperature()
        
#         # 3. Đánh giá metrics
#         print("Evaluating metrics...")
#         with torch.no_grad(), autocast(device_type="cuda", dtype=torch.float16):
#             for data, labels in tqdm(self.valid_loader, desc="Validation"):
#                 labels = labels.to(self.gpu_id)
#                 data, labels = data.to(self.gpu_id), labels.to(self.gpu_id)
#                 outputs = self.model(data)
                
#                 # Áp dụng thresholds cho multi-label
#                 probs = torch.sigmoid(outputs)
#                 model = self.get_model()
#                 preds = (probs > model.thresholds).float()
                
#                 # Update all metrics
#                 for name, metric in self.eval_metrics.items():
#                     if name == 'accuracy':  # Xử lý riêng cho accuracy
#                         metric_value = metric(preds, labels)
#                     else:
#                         try:
#                             metric_value = metric(outputs, labels)  # AP dùng logits
#                         except:
#                             print("AP trong _validate_and_calibrate dùng logits")
#                             metric_value = metric(outputs, labels.long())  # AP dùng logits
#                     metric_meters[name].update(metric_value.item(), data.size(0))
        
#         # Log metrics
#         current_metrics = {name: meter.avg for name, meter in metric_meters.items()}
#         print(f"=== Validation Results ===")
#         for name, value in current_metrics.items():
#             print(f"{name}: {value:.4f}")
        
#         # Lưu best model
#         if current_metrics.get('average_precision', 0) > self.best_map:
#             old_best = self.best_map
#             self.best_map = current_metrics['average_precision']
#             print(f"New best average_precision: {old_best:.4f} -> {self.best_map:.4f}")
#             self.save(f'./models/best_model_step{self.current_step}.pth')
#             print(f"Saved best model at step {self.current_step}")
#         else:
#             print(f"No improvement in average_precision. Current best: {self.best_map:.4f}")

#     def _update_thresholds(self, percentile=85):
#         """Cập nhật dynamic thresholds dựa trên validation set"""
#         probs, truths = [], []
#         with torch.no_grad(), autocast(device_type="cuda", dtype=torch.float16):
#             for data, labels in tqdm(self.valid_loader, desc="Updating thresholds"):
#                 outputs = torch.sigmoid(self.model(data.to(self.gpu_id)))
#                 probs.append(outputs.cpu())
#                 truths.append(labels.cpu())
        
#         probs = torch.cat(probs)
#         truths = torch.cat(truths)
#         model = self.get_model()
        
#         new_thresholds = []
#         for i in tqdm(tqdm(range(truths.shape[1]))):
#             pos_probs = probs[truths[:, i] == 1, i]
#             if len(pos_probs) > 0:
#                 new_thresholds.append(torch.quantile(pos_probs, percentile/100))
#             else:
#                 new_thresholds.append(torch.tensor(0.5))
        
#         # Chuyển danh sách thành tensor
#         threshold_tensor = torch.stack(new_thresholds).to(self.gpu_id)
#         model.thresholds.copy_(threshold_tensor)

#     def _calibrate_temperature(self):
#         """Hiệu chỉnh temperature parameter để cải thiện calibration"""
#         model = self.get_model()
#         nll_criterion = nn.BCEWithLogitsLoss().to(self.gpu_id)
        
#         # Tạo optimizer cho temperature
#         temp_optimizer = torch.optim.LBFGS([model.temperature], lr=0.01, max_iter=50)
        
#         def closure():
#             temp_optimizer.zero_grad()
#             loss = 0
#             for data, labels in tqdm(self.valid_loader):
#                 data, labels = data.to(self.gpu_id), labels.to(self.gpu_id)
#                 with torch.no_grad():
#                     logits = model(data)
#                 # Áp dụng temperature
#                 scaled_logits = logits / model.temperature
#                 loss += nll_criterion(scaled_logits, labels.float())
#             loss.backward()
#             return loss
        
#         temp_optimizer.step(closure)
#         # Giới hạn temperature trong khoảng hợp lý
#         model.temperature.data.clamp_(0.1, 10.0)
#         print(f"Calibrated temperature: {model.temperature.item():.4f}")

#     def run_training(self):
#         """Chạy quá trình huấn luyện và thực hiện final test"""
#         progress_bar = tqdm(range(self.max_steps), desc="Training")
#         for _ in progress_bar:
#             loss = self.train_step()
#             progress_bar.set_postfix(
#                 loss=f"{loss:.4f}", 
#                 lr=f"{self.scheduler.get_last_lr()[0]:.2e}",
#                 step=f"{self.current_step}/{self.max_steps}"
#             )
        
#         # Thực hiện final test sau khi huấn luyện xong
#         print("\n=== Running Final Test ===")
#         # Đầu tiên sử dụng model hiện tại
#         print("Testing current model:")
#         current_results = self.test()
        
#         # Sau đó nạp best model (nếu có) và test lại
#         best_model_path = f'./models/best_model_step{self.current_step}.pth'
#         if os.path.exists(best_model_path):
#             print(f"\nTesting best model (from step {self.current_step}):")
#             self.load(best_model_path)
#             best_results = self.test()
            
#             # So sánh kết quả
#             print("\n=== Results Comparison ===")
#             for metric in current_results.keys():
#                 print(f"{metric}: Current={current_results[metric]:.4f}, Best={best_results[metric]:.4f}")
        
#         return current_results
    
#     def save(self, file_path="./models/checkpoint.pth"):
#         """Lưu checkpoint"""
#         os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
#         model = self.get_model()
        
#         checkpoint = {
#             "backbone": model.backbone.state_dict(),
#             "classifier": model.classifier.state_dict(),
#             "gnn": model.gnn.state_dict(),
#             "temperature": model.temperature.item(),
#             "thresholds": model.thresholds.cpu(),
#             "optimizer": self.optimizer.state_dict(),
#             "scheduler": self.scheduler.state_dict(),
#             "global_step": self.global_step,
#         }
        
#         torch.save(checkpoint, file_path)
#         print(f"Model saved to {file_path}")
        
#     def load(self, file_path):
#         """Nạp checkpoint"""
#         if not os.path.exists(file_path):
#             print(f"Checkpoint {file_path} không tồn tại!")
#             return False
            
#         checkpoint = torch.load(file_path, map_location=f"cuda:{self.gpu_id}")
#         model = self.get_model()
        
#         # Nạp các thành phần của model
#         model.backbone.load_state_dict(checkpoint["backbone"])
#         model.classifier.load_state_dict(checkpoint["classifier"])
        
#         # Nạp GNN nếu có
#         if "gnn" in checkpoint:
#             model.gnn.load_state_dict(checkpoint["gnn"])
        
#         # Nạp temperature nếu có
#         if "temperature" in checkpoint:
#             with torch.no_grad():
#                 model.temperature.copy_(torch.tensor([checkpoint["temperature"]]).to(self.gpu_id))
        
#         # Nạp thresholds nếu có
#         if "thresholds" in checkpoint:
#             model.thresholds.copy_(checkpoint["thresholds"].to(self.gpu_id))
                
#         # Nạp optimizer và scheduler
#         self.optimizer.load_state_dict(checkpoint["optimizer"])
#         if "scheduler" in checkpoint:
#             self.scheduler.load_state_dict(checkpoint["scheduler"])
        
#         # Nạp global step
#         if "global_step" in checkpoint:
#             self.global_step = checkpoint["global_step"]
#             self.current_step = self.global_step
            
#         print(f"Loaded checkpoint from {file_path} (step {self.global_step})")
#         return True
    
#     def test(self):
#         """Đánh giá model trên tập test"""
#         self.model.eval()
#         metric_meters = {name: AverageMeter() for name in self.eval_metrics.keys()}
        
#         all_probs = []
#         all_preds = []
#         all_labels = []
        
#         with torch.no_grad(), autocast(device_type="cuda", dtype=torch.float16):
#             for data, labels in tqdm(self.test_loader, desc="Testing"):
#                 data, labels = data.to(self.gpu_id), labels.to(self.gpu_id)
#                 outputs = self.model(data)
                
#                 # Áp dụng temperature scaling và thresholds
#                 model = self.get_model()
#                 probs = torch.sigmoid(outputs / model.temperature)
#                 preds = (probs > model.thresholds).float()
                
#                 # Lưu kết quả
#                 all_probs.append(probs.cpu())
#                 all_preds.append(preds.cpu())
#                 all_labels.append(labels.cpu())
                
#                 # Update all metrics
#                 for name, metric in self.eval_metrics.items():
#                     if name == 'accuracy':
#                         metric_value = metric(preds, labels)
#                     else:
#                         metric_value = metric(outputs, labels)
#                     metric_meters[name].update(metric_value.item(), data.size(0))
        
#         # Tổng hợp kết quả
#         all_probs = torch.cat(all_probs)
#         all_preds = torch.cat(all_preds)
#         all_labels = torch.cat(all_labels)
        
#         # In kết quả chi tiết
#         print("\n=== Test Results ===")
#         for name, meter in metric_meters.items():
#             print(f"{name}: {meter.avg:.4f}")
        
#         # Phân tích chi tiết theo từng lớp
#         print("\n=== Per-Class Analysis ===")
#         for i, class_name in enumerate(self.class_list):
#             class_precision = (all_preds[:, i] * all_labels[:, i]).sum() / (all_preds[:, i].sum() + 1e-8)
#             class_recall = (all_preds[:, i] * all_labels[:, i]).sum() / (all_labels[:, i].sum() + 1e-8)
#             class_f1 = 2 * class_precision * class_recall / (class_precision + class_recall + 1e-8)
            
#             print(f"{class_name}: P={class_precision:.4f}, R={class_recall:.4f}, F1={class_f1:.4f}, "
#                   f"Threshold={model.thresholds[i].item():.4f}")
        
#         return {name: meter.avg for name, meter in metric_meters.items()}