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

import torch
import torch.nn as nn
from transformers import TimesformerModel
from torch_geometric.nn import GATv2Conv
from torch_cluster import knn_graph
import math

class FeatureFusionWithCrossAttention(nn.Module):
    def __init__(self,query_dim, context_dim, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.context_proj = nn.Linear(context_dim, hidden_dim)
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()
    def _reset_parameters(self):
        """Khởi tạo tham số với kỹ thuật tốt hơn để ổn định quá trình huấn luyện"""
        nn.init.xavier_uniform_(self.query_proj.weight, gain=1/math.sqrt(2))
        nn.init.xavier_uniform_(self.context_proj.weight, gain=1/math.sqrt(2))
        nn.init.zeros_(self.query_proj.bias)
        nn.init.zeros_(self.context_proj.bias)
    def forward(self, query_features, context_features):
        """
        Args:
            query_features: [batch_size, hidden_dim] (instance features)
            context_features: [batch_size, context_dim] (GNN features)
        """
        q = self.query_proj(query_features).unsqueeze(1)  # [B, 1, D]
        k = v = self.context_proj(context_features).unsqueeze(1)  # [B, 1, D]
        
        attn_output, _ = self.cross_attn(q, k, v)
        return self.norm(query_features + self.dropout(attn_output.squeeze(1)))

class EMAMeter:
    """
    Exponential Moving Average để làm mượt giá trị loss trong quá trình huấn luyện.
    Giúp theo dõi xu hướng thực sự của loss thay vì bị ảnh hưởng bởi dao động ngẫu nhiên.
    """
    def __init__(self, alpha=0.9):
        self.alpha = alpha
        self.value = None
        
    def update(self, value):
        if self.value is None:
            self.value = value
        else:
            self.value = self.alpha * self.value + (1 - self.alpha) * value
        return self.value


class TimeSformerGNN(nn.Module):
    def __init__(self, num_frames, num_classes=18, train_loader=None, 
                 hidden_size=768, gnn_dim=256, fusion_type='cross_attn'):
        super().__init__()
        self.num_classes = num_classes
        self.fusion_type = fusion_type
        
        # Backbone initialization
        self.backbone = TimesformerModel.from_pretrained(
            "facebook/timesformer-hr-finetuned-k600",
            num_frames=num_frames,
            ignore_mismatched_sizes=True
        )
        
        # Graph components
        self.class_emb = nn.Parameter(torch.randn(num_classes, hidden_size) * 0.02)
        self.register_buffer('cooccur_matrix', self._init_cooccur_matrix(train_loader))
        self.class_edge_index = self._build_class_graph()
        
        # GNN Networks
        self.gnn_class = GATv2Conv(hidden_size, gnn_dim, heads=2, concat=True, dropout = 0.5)
        self.gnn_instance = GATv2Conv(hidden_size, gnn_dim, heads=2, concat=True, dropout = 0.5)
        
        # Feature fusion
        if fusion_type == 'cross_attn':
            self.fusion = FeatureFusionWithCrossAttention(
                query_dim=hidden_size,
                context_dim=gnn_dim*2,  # *2 due to GAT concat
                hidden_dim=hidden_size,
                dropout = 0.5
            )
        else:
            self.fusion = nn.Sequential(
                nn.Linear(hidden_size + gnn_dim, hidden_size),
                nn.GELU()
            )
        
        # Classifier
        self.pre_classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.classifier = GroupWiseLinear(num_classes, hidden_size)
        
        # Dynamic parameters
        self.temperature = nn.Parameter(torch.ones(1))
        self.register_buffer('thresholds', torch.ones(num_classes) * 0.5)
        
        # Configuration
        self.knn_k = 5  # For instance-level graph
        self.loss_ema = EMAMeter(alpha=0.9)
        
    def _init_cooccur_matrix(self, loader):
        if loader is None:
            return torch.zeros((self.num_classes, self.num_classes))
        try:
            cooccur = compute_cooccurrence(loader, self.num_classes)
            return torch.from_numpy(cooccur.toarray()) if isinstance(cooccur, csr_matrix) else cooccur
        except Exception as e:
            print(f"Co-occurrence matrix error: {e}")
            return torch.zeros((self.num_classes, self.num_classes))

    def _build_class_graph(self):
        rows, cols = torch.where(self.cooccur_matrix > 0)
        if len(rows) == 0:
            return torch.zeros((2, 0), dtype=torch.long)
        return torch.stack([rows, cols], dim=0)

    def forward(self, x):
        # Feature extraction
        backbone_out = self.backbone(x)
        instance_features = backbone_out.last_hidden_state[:, 0]  # [B, D]
        
        # Class-level GNN
        class_features = self.gnn_class(
            self.class_emb.to(instance_features.device),
            self.class_edge_index.to(instance_features.device)
        )  # [num_classes, gnn_dim]
        
        # Instance-level GNN
        instance_edge_index = knn_graph(instance_features, k=self.knn_k, loop=True)
        gnn_instance_features = self.gnn_instance(instance_features, instance_edge_index)
        
        # Feature fusion
        if self.fusion_type == 'cross_attn':
            # Cross-attention between instance and class features
            class_context = class_features.mean(0, keepdim=True).expand(instance_features.size(0), -1)
            fused_features = self.fusion(instance_features, class_context + gnn_instance_features)
        else:
            # Concatenation fusion
            class_context = class_features.mean(0, keepdim=True).expand(instance_features.size(0), -1)
            fused_features = self.fusion(
                torch.cat([instance_features, class_context + gnn_instance_features], dim=-1)
            )
        # Classification
        fused_features = self.pre_classifier(fused_features)
        logits = self.classifier(fused_features.unsqueeze(1))
        
        # Temperature scaling
        return logits / self.temperature if not self.training else logits

class GroupWiseLinear(nn.Module):
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        self.W.data.uniform_(-stdv, stdv)
        if self.bias:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: [B, 1, D]
        return (self.W * x).sum(-1) + (self.b if self.bias else 0)

class TimeSformerGNNExecutor:
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
        
        model = TimeSformerGNN(num_frames=num_frames, num_classes=num_classes, train_loader=train_loader).to(self.gpu_id)
        
        if distributed:
            self.model = DDP(model, device_ids=[gpu_id])
        else:
            self.model = model
            
        # Fine-tuning toàn bộ backbone
        for p in self.model.backbone.parameters():
            p.requires_grad = True
            
        # Optimizer và scheduler
        self.optimizer = AdamW([{"params": self.model.parameters(), "lr": 0.00001}])
        
        self.scheduler = OneCycleLR(
            self.optimizer, 
            max_lr= 0.00005,
            total_steps=self.max_steps,
            pct_start=0.1,
            div_factor=25,
            final_div_factor=1000
        )
        
        # Gradient scaler cho mixed precision
        self.scaler = GradScaler()
        
        # Iterator cho training
        self.train_iter = iter(train_loader)
        # Metrics
        self.best_map = 0.0
        # Flag để kiểm soát tần suất hiệu chỉnh temperature
        self.last_temp_calibration = 0
        self.loss_ema = EMAMeter()
    
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
        smoothed_loss = self.loss_ema.update(total_loss)
        # Validation định kỳ
        if self.current_step % 300 == 0:
            # Luôn validate và cập nhật thresholds
            self._validate()
            
            # Chỉ hiệu chỉnh temperature theo định kỳ ít hơn
            if self.current_step % 2400 == 0 or self.current_step == 600:
                print("Calibrating temperature...")
                self._calibrate_temperature()
                self.last_temp_calibration = self.current_step
            
            self.save(f"./models/checkpoint_step{self.current_step}.pth")
            
        return smoothed_loss
    
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
                    if name == 'weighted Accuracy':  # Xử lý riêng cho accuracy
                        metric_value = metric(preds, labels)
                    else:
                        try:
                            metric_value = metric(outputs, labels)  # AP dùng logits
                        except:
                            # print("AP trong _validate_and_calibrate dùng logits")
                            metric_value = metric(outputs, labels.long())  # AP dùng logits
                    metric_meters[name].update(metric_value.item(), data.size(0))
        
        # Log metrics
        current_metrics = {name: meter.avg for name, meter in metric_meters.items()}
        print(f"=== Validation Results ===")
        for name, value in current_metrics.items():
            print(f"{name}: {value:.4f}")
        
        # Lưu best model
        if current_metrics.get('weighted v1 mAP', 0) > self.best_map:
            old_best = self.best_map
            self.best_map = current_metrics['weighted v1 mAP']
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
        """Lưu checkpoint hoàn chỉnh"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        model = self.get_model()
        
        checkpoint = {
            # Model states
            "backbone": model.backbone.state_dict(),
            "gnn_class": model.gnn_class.state_dict(),
            "gnn_instance": model.gnn_instance.state_dict(),
            "classifier": model.classifier.state_dict(),
            "fusion": model.fusion.state_dict(),
            "class_emb": model.class_emb.data,
            "temperature": model.temperature.data,
            "thresholds": model.thresholds.data,
            
            # Training states
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "scaler": self.scaler.state_dict(),
            
            # Training progress
            "current_step": self.current_step,
            "global_step": self.global_step,
            "best_metric": self.best_map,
            "last_temp_calibration": self.last_temp_calibration,
        }
        
        torch.save(checkpoint, file_path)
        print(f"Checkpoint saved to {file_path} (step {self.current_step})")

    def load(self, file_path):
        """Nạp checkpoint hoàn chỉnh với xử lý tương thích ngược"""
        if not os.path.exists(file_path):
            print(f"Warning: Checkpoint {file_path} not found!")
            return False
        
        checkpoint = torch.load(file_path, map_location=f"cuda:{self.gpu_id}")
        model = self.get_model()
        
        # 1. Load model parameters với xử lý tương thích
        model.backbone.load_state_dict(checkpoint["backbone"])
        
        # Xử lý cho GNN (tương thích cả version cũ và mới)
        if "gnn" in checkpoint:  # Version cũ dùng chung 1 GNN
            model.gnn_class.load_state_dict(checkpoint["gnn"])
            model.gnn_instance.load_state_dict(checkpoint["gnn"])
        else:  # Version mới tách riêng
            model.gnn_class.load_state_dict(checkpoint["gnn_class"])
            model.gnn_instance.load_state_dict(checkpoint["gnn_instance"])
        
        model.classifier.load_state_dict(checkpoint["classifier"])
        model.fusion.load_state_dict(checkpoint["fusion"])
        
        # Load các parameter đặc biệt
        with torch.no_grad():
            model.class_emb.copy_(checkpoint["class_emb"])
            if "temperature" in checkpoint:
                model.temperature.copy_(checkpoint["temperature"])
            else:
                model.temperature.copy_(torch.tensor([1.0]).to(self.gpu_id))
            
            if "thresholds" in checkpoint:
                model.thresholds.copy_(checkpoint["thresholds"])
            else:
                model.thresholds.copy_(torch.ones(model.num_classes, device=self.gpu_id) * 0.5)
        
        # 2. Load training states
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        
        if "scheduler" in checkpoint and checkpoint["scheduler"] is not None:
            if self.scheduler:
                self.scheduler.load_state_dict(checkpoint["scheduler"])
        
        if "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])
        
        # 3. Load training progress
        self.current_step = checkpoint.get("current_step", 0)
        self.global_step = checkpoint.get("global_step", self.current_step)
        self.best_map = checkpoint.get("best_metric", 0.0)
        self.last_temp_calibration = checkpoint.get("last_temp_calibration", 0)
        
        print(f"Loaded checkpoint from {file_path} (step {self.current_step})")
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
                    if name == 'weighted Accuracy':
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
