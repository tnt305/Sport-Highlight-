import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from transformers import TimesformerModel, ASTModel, ASTFeatureExtractor
import librosa
import os
import time
import sys
from tqdm import tqdm
import numpy as np
from torch.optim import Adam, AdamW

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))  # Lên 3 cấp đến v5
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from transformers import TimesformerModel, logging
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import warnings
from torch.amp import autocast , GradScaler
from loguru import logger
from typing import Any
from sklearn.metrics import average_precision_score
from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperModel

from short_sport.video_trainer.losses.loss import AsymmetricLossOptimized, CalibratedRankingLoss
from short_sport.video_trainer.utils import AverageMeter
from short_sport.video_trainer.datasets.augmentations import MixUpAugmentation, MultimodalMixupAugmentation
from short_sport.video_trainer.architecture import VideoRoPE, VideoRoPESelfAttention
warnings.filterwarnings("ignore")

logger.info("MultimodalMixupAugmentation is initialized")
multimodal_mixup_fn = MultimodalMixupAugmentation(alpha=0.2)
class AudioVideoFusionClassifier(nn.Module):
    def __init__(self, num_frames, num_classes=18, hidden_size=768, dropout_prob=0.1):
        super(AudioVideoFusionClassifier, self).__init__()
        # Pre-trained models
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.backbone = TimesformerModel.from_pretrained(
            "facebook/timesformer-hr-finetuned-k600", 
            num_frames=num_frames,
            ignore_mismatched_sizes=True
        )
        self._add_video_rope(self.backbone.config.hidden_size)
        
        self.ast = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593", ignore_mismatched_sizes=True)
        self.audio_feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593", ignore_mismatched_sizes=True)
        self.backbone.gradient_checkpointing_enable()
        self.ast.gradient_checkpointing_enable()
        
        # Feature extractors for better representation
        # Streamlined projectors
        self.video_projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout_prob)
        )
        
        self.audio_projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout_prob)
        )
        
        # Simplified cross-attention - single layer instead of two
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size, 
            num_heads=8,
            dropout=dropout_prob
        )
        
        self.modality_balancer = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.Sigmoid()
        )
        
        # Simplified fusion with reduced hidden dimension
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size*2),  # Reduced from 4x to 2x
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size*2, hidden_size)
        )
        
        # Streamlined classifier
        self.classifier = nn.Linear(hidden_size, num_classes)
    def _add_video_rope(self, dim_size):
        """Replace original attention with VideoRoPE-enhanced attention"""
        # Debug: Print the structure of the first layer's attention
        first_layer = self.backbone.encoder.layer[0]
        print(f"Attention module structure: {dir(first_layer.attention)}")
        print(f"Attention module type: {type(first_layer.attention)}")
        
        for layer in self.backbone.encoder.layer:
            original_attn = layer.attention
            
            # Use values from config instead
            new_attn = VideoRoPESelfAttention(
                embed_dim=dim_size,
                num_heads=self.backbone.config.num_attention_heads,
                dropout=0.1,
                dim_size=dim_size
            )
            layer.attention = new_attn
    def _cross_modal_attention(self, query, key, value):
        """Multi-layer cross attention with residuals"""
        attn_out = query
        for attn_layer in self.cross_attention:
            attn_output, _ = attn_layer(
                query=attn_out,
                key=key,
                value=value
            )
            attn_out = attn_out + attn_output  # Residual
        return attn_out
    def preprocess_audio(self, audio_paths, sample_rate=16000):
        waveforms = []
        valid_indices = []
        min_duration = 0.025  # 25ms minimum (400 samples at 16kHz)
        
        for idx, path in enumerate(audio_paths):
            try:
                y, sr = librosa.load(path, sr=sample_rate)
                # Kiểm tra độ dài audio
                if len(y) / sr >= min_duration:
                    waveforms.append(y)
                    valid_indices.append(idx)
                else:
                    print(f"Bỏ qua audio tại {path}: quá ngắn ({len(y)/sr:.3f}s < {min_duration}s)")
            except Exception as e:
                print(f"Lỗi khi tải audio tại {path}: {str(e)}")
                continue
                
        return waveforms, valid_indices

    def gated_fusion(self, video_features, audio_features):
        """Implement improved gated fusion mechanism"""
        # Project features first
        video_proj = self.video_projector(video_features)
        audio_proj = self.audio_projector(audio_features)
        
        # Concatenate features
        combined = torch.cat((video_proj, audio_proj), dim=-1)
        
        # Compute gates
        video_gate = self.video_gate(combined)
        audio_gate = self.audio_gate(combined)
        
        # Gated features with residual connections
        gated_video = video_features * video_gate + video_features * 0.1
        gated_audio = audio_features * audio_gate + audio_features * 0.1
        
        return gated_video + gated_audio

    def forward(self, video, audio_paths):
        # Video feature extraction with VideoRoPE
        video_features = self.backbone(video).last_hidden_state[:, 0, :]  # CLS token
        print(video_features.shape)
        video_features = self.video_projector(video_features)
        
        # Audio feature extraction with safety checks
        waveforms, valid_indices = self.preprocess_audio(audio_paths)
        
        if not valid_indices:
            # If no valid audio, use zero features but keep video processing
            audio_features = torch.zeros_like(video_features)
        else:
            # Filter video based on valid audio indices if needed
            if len(valid_indices) != video.size(0):
                video = video[valid_indices]
                video_features = self.backbone(video).last_hidden_state[:, 0, :]
                video_features = self.video_projector(video_features)
            
            audio_inputs = self.audio_feature_extractor(
                waveforms,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            ).to(video.device)
            
            # Use encoder outputs for feature extraction
            audio_features = self.ast(**audio_inputs).last_hidden_state[:, 0, :]  # Average pooling
            audio_features = self.audio_projector(audio_features)

        # Single cross-attention operation
        attn_output, _ = self.cross_attention(
            query=video_features.unsqueeze(1),
            key=audio_features.unsqueeze(1),
            value=audio_features.unsqueeze(1)
        )
        attended_features = attn_output.squeeze(1)
        
        # Modality balancing
        modality_weights = self.modality_balancer(
            torch.cat([video_features, audio_features], dim=-1)
        )
        
        # Apply modality weights to balance video and audio contributions
        weighted_features = modality_weights * video_features + (1-modality_weights) * audio_features
        
        # Combine features using both weighted features and cross-attention results
        combined_features = torch.cat([video_features, audio_features], dim=-1)
        fusion_features = self.fusion(combined_features)
        
        # Final feature representation combines attended features, weighted features, and fusion
        final_features = attended_features + weighted_features + fusion_features
        
        return self.classifier(final_features)

class VAHarderFusionExecuter:
    def __init__(self, train_loader, 
                 test_loader, 
                 criterion, 
                 eval_metrics, 
                 class_list, 
                 test_every,
                 learning_rate,
                 n_epochs,
                 distributed, 
                 gpu_id, 
                 logger) -> None:
        super().__init__()
        self.train_loader = train_loader
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
        self.n_epochs = n_epochs
        # Khởi tạo mô hình với dropout cao hơn để tránh overfitting
        model = AudioVideoFusionClassifier(num_frames=num_frames, num_classes=num_classes, dropout_prob=0.1).to(gpu_id)
        
        if distributed: 
            self.model = DDP(model, device_ids=[gpu_id])
        else: 
            self.model = model
        
        for p in self.model.parameters():
            p.requires_grad = True
        self.lr = learning_rate
        self.optimizer = Adam([{"params": self.model.parameters(), "lr": self.lr}]) # Thêm weight decay để giảm overfitting
        self.scaler = GradScaler()
        # Khởi tạo scheduler với warm-up
        num_training_steps = len(train_loader) * self.n_epochs # Giả sử 50 epochs
        num_warmup_steps = int(0.05 * num_training_steps)  # 5% warm-up
        
        self.scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Thêm biến để theo dõi best performance
        self.best_map = 0.0
        self.best_epoch = 0
        self.patience = 0
        self.max_patience = 2  # Early stopping sau 2 epochs không cải thiện
        
        self.logger = logger
        self.logger.info(f"[INFO] Initialized VAFusionExecuter on GPU {gpu_id}")

    def get_model(self):
        return self.model.module if self.distributed else self.model
    
    def _train_batch(self, data, audio, label):
        self.optimizer.zero_grad()
        
        # Forward pass với mixup cho cả video và audio
        with autocast(device_type="cuda", dtype=torch.float16):
            mixed_video, mixed_audio, mixed_labels = multimodal_mixup_fn(data, audio, label)
            output = self.model(mixed_video, mixed_audio)
            loss_this = self.criterion(output, mixed_labels)
        # Backward và optimize
        self.scaler.scale(loss_this).backward()
        
        # Gradient clipping để ổn định học tập
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()  # Cập nhật learning rate
        
        return loss_this.item()

    def _train_epoch(self, epoch):
        self.model.train()
        loss_meter = AverageMeter()
        
        # Thêm metric tracking trong training
        metric_meters = {name: AverageMeter() for name in self.eval_metrics.keys()}
        
        start_time = time.time()
        for data, audio, label in tqdm(self.train_loader):
            data, label = data.to(self.gpu_id), label.to(self.gpu_id)
            
            # Train batch
            loss_this = self._train_batch(data, audio, label)
            loss_meter.update(loss_this, data.shape[0])
            
            # Tính metrics trong training (không tính gradient)
            with torch.no_grad(), autocast(device_type="cuda", dtype=torch.float16):
                output = self.model(data, audio)
                for name, metric in self.eval_metrics.items():
                    metric_value = metric(output, label.long())
                    metric_meters[name].update(metric_value.item(), data.shape[0])
        
        elapsed_time = time.time() - start_time
        if (self.distributed and self.gpu_id == 0) or not self.distributed:
            self.logger.info(
                f"Epoch [{epoch + 1}][{time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}] "
                f"loss: {loss_meter.avg:.4f}"
            )
            
            # Log training metrics
            for name, meter in metric_meters.items():
                self.logger.info(f"[INFO] Train {name}: {meter.avg * 100:.2f}")
    
    def train(self, start_epoch, end_epoch):
        # Phương pháp huấn luyện theo giai đoạn
        
        # # Giai đoạn 1: Freeze backbone, chỉ huấn luyện fusion và classifier
        # if start_epoch == 0:
        #     self.logger.info("Phase 1: Freezing backbone and AST, training only fusion and classifier")
        #     for param in self.get_model().backbone.parameters():
        #         param.requires_grad = False
        #     for param in self.get_model().ast.parameters():
        #         param.requires_grad = False
                
        #     # Huấn luyện 5 epochs đầu tiên
        #     for epoch in range(start_epoch, min(5, end_epoch)):
        #         self._train_epoch(epoch)
        #         if (epoch + 1) % self.test_every == 0:
        #             metrics_values = self.test()
        #             self._save_if_best(metrics_values, epoch)
            
        #     start_epoch = 5  # Cập nhật start_epoch cho giai đoạn tiếp theo
        
        # # Giai đoạn 2: Unfreeze một phần backbone và AST
        # if start_epoch <= 5 and end_epoch > 5:
        #     self.logger.info("Phase 2: Partially unfreezing backbone and AST")
            
        #     # Unfreeze một số lớp cuối của backbone
        #     for i, (name, param) in enumerate(self.get_model().backbone.named_parameters()):
        #         if 'layer.11' in name or 'layer.10' in name:  # Chỉ unfreeze 2 lớp cuối
        #             param.requires_grad = True
            
        #     # Unfreeze một số lớp cuối của AST
        #     for i, (name, param) in enumerate(self.get_model().ast.named_parameters()):
        #         if 'layer.11' in name or 'layer.10' in name:  # Chỉ unfreeze 2 lớp cuối
        #             param.requires_grad = True
            
        #     # Huấn luyện từ epoch 5 đến 15
        #     for epoch in range(max(start_epoch, 5), min(15, end_epoch)):
        #         self._train_epoch(epoch)
        #         if (epoch + 1) % self.test_every == 0:
        #             metrics_values = self.test()
        #             self._save_if_best(metrics_values, epoch)
            
        #     start_epoch = 15  # Cập nhật start_epoch cho giai đoạn tiếp theo
        
        # # Giai đoạn 3: Unfreeze toàn bộ mô hình
        # if start_epoch <= 15 and end_epoch > 15:
        self.logger.info("Phase 3: Unfreezing all layers")
        for param in self.get_model().parameters():
            param.requires_grad = True
        
        # Huấn luyện các epochs còn lại
        for epoch in range(start_epoch, end_epoch):
            self._train_epoch(epoch)
            if (epoch + 1) % self.test_every == 0:
                metrics_values = self.test()
                self._save_if_best(metrics_values, epoch)
                
            # Kiểm tra early stopping
            if self.patience >= self.max_patience:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    def _save_if_best(self, metrics_values, epoch):
        # Lưu mô hình nếu mAP tốt nhất
        if 'mAP' in metrics_values and metrics_values['mAP'] > self.best_map:
            self.best_map = metrics_values['mAP']
            self.best_epoch = epoch + 1
            self.patience = 0
            
            if (self.distributed and self.gpu_id == 0) or not self.distributed:
                self.logger.info(f"[INFO] New best mAP: {self.best_map * 100:.2f}% at epoch {self.best_epoch}")
                self.save(file_path=f"./best_model_epoch{self.best_epoch}.pth")
        else:
            self.patience += 1
            if (self.distributed and self.gpu_id == 0) or not self.distributed:
                self.logger.info(f"[INFO] No improvement for {self.patience} epochs. Best mAP: {self.best_map * 100:.2f}% at epoch {self.best_epoch}")
    
    def test(self):
        self.model.eval()
        metric_meters = {name: AverageMeter() for name in self.eval_metrics.keys()}
        
        # Thêm tracking cho logits và labels để tính mAP chính xác hơn
        all_outputs = []
        all_labels = []
        
        for data, audio, label in tqdm(self.test_loader):
            data, label = data.to(self.gpu_id), label.to(self.gpu_id)
            with torch.no_grad(), autocast(device_type="cuda", dtype=torch.float16):
                # Thêm test-time augmentation (TTA)
                # if hasattr(self, 'use_tta') and self.use_tta:
                #     # Original
                #     output1 = self.model(data, audio)
                    
                #     # Horizontal flip
                #     flipped_data = torch.flip(data, dims=[-1])  # Flip horizontal
                #     output2 = self.model(flipped_data, audio)
                    
                #     # Average predictions
                #     output = (output1 + output2) / 2.0
                # else:
                output = self.model(data, audio)
                
            # Lưu outputs và labels
            # all_outputs.append(output.detach().cpu())
            # all_labels.append(label.detach().cpu())
            
            # Update all metrics
            for name, metric in self.eval_metrics.items():
                metric_value = metric(output, label.long())
                metric_meters[name].update(metric_value.item(), data.shape[0])
        
        # Tính mAP trên toàn bộ tập dữ liệu (không phải trung bình batch)
   
        metrics = {name: meter.avg for name, meter in metric_meters.items()}
        for name, value in metrics.items():
            self.logger.info(f"[INFO] Test {name}: {value * 100:.2f}")
        
        return metrics
    
    def save(self, file_path="./checkpoint.pth"):
        # Lưu toàn bộ mô hình thay vì chỉ lưu một số phần
        if self.distributed:
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
            
        torch.save({
            "model": model_state_dict,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "best_map": self.best_map,
            "best_epoch": self.best_epoch,
            "patience": self.patience
        }, file_path)
        
        self.logger.info(f"[INFO] Model saved to {file_path}")
        
    def load(self, file_path):
        self.logger.info(f"[INFO] Loading model from {file_path}")
        checkpoint = torch.load(file_path, map_location=f"cuda:{self.gpu_id}")
        
        if self.distributed:
            self.model.module.load_state_dict(checkpoint["model"])
        else:
            self.model.load_state_dict(checkpoint["model"])
            
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scheduler and "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            
        if "best_map" in checkpoint:
            self.best_map = checkpoint["best_map"]
        if "best_epoch" in checkpoint:
            self.best_epoch = checkpoint["best_epoch"]
        if "patience" in checkpoint:
            self.patience = checkpoint["patience"]
            
        self.logger.info(f"[INFO] Loaded model with best mAP: {self.best_map * 100:.2f}% at epoch {self.best_epoch}")
        
    # Thêm phương thức để bật/tắt TTA
    def enable_tta(self, enabled=True):
        self.use_tta = enabled
        self.logger.info(f"[INFO] Test-time augmentation: {'enabled' if enabled else 'disabled'}")