import copy
import torch
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
from short_sport.video_trainer.losses.loss import AsymmetricLossOptimized, CalibratedRankingLoss
from torch.amp import autocast , GradScaler
from loguru import logger
from typing import Any

from short_sport.video_trainer.architecture.uniformer_v2_root import uniformerv2_l14, uniformerv2_b16
from short_sport.video_trainer.utils import AverageMeter
from short_sport.video_trainer.datasets.augmentations import MixUpAugmentation
from short_sport.video_trainer.architecture import VideoRoPE
warnings.filterwarnings("ignore")

mixup_fn = MixUpAugmentation(alpha=0.2)

class Uniformerv2Backbone(nn.Module):
    def __init__(self, original_model):
        super(Uniformerv2Backbone, self).__init__()
        # Giữ lại các thành phần cần thiết từ mô hình gốc
        self.original_model = original_model
        if hasattr(self.original_model.transformer, 'sigmoid'):
            delattr(self.original_model.transformer, 'sigmoid')
        if hasattr(self.original_model.transformer, 'proj'):
            delattr(self.original_model.transformer, 'proj')
    def forward(self, x):
        return self.original_model(x)

class AudioVideoFusionClassifier(nn.Module):
    def __init__(self, num_frames, num_classes=18, hidden_size=768, dropout_prob=0.1):
        super(AudioVideoFusionClassifier, self).__init__()
        # Pre-trained models
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.backbone = Uniformerv2Backbone(uniformerv2_b16())
        state_dict = torch.load("./hub/uniformerv2-base-p16-res224_clip-kinetics710-pre_8xb32-u8_kinetics400.pth", map_location='cpu')
        self.backbone.load_state_dict(state_dict, strict  = False)
        
        self.ast = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593", ignore_mismatched_sizes = True)
        self.audio_feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593", ignore_mismatched_sizes = True)
        
        # Advanced Fusion Mechanisms
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size, 
            num_heads=4, 
            dropout=dropout_prob
        )
        self.learnable_temperature = nn.Parameter(torch.ones(1))
        # Gated Fusion Layer
        self.video_gate = nn.Linear(hidden_size * 2, hidden_size)
        self.audio_gate = nn.Linear(hidden_size * 2, hidden_size)
        
        # Residual Fusion Layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size // 2, self.num_classes)
        )

    def preprocess_audio(self, audio_paths, sample_rate=16000):
        waveforms = []
        min_duration = 0.025  # 25ms minimum (400 samples at 16kHz)
        for path in audio_paths:
            y, sr = librosa.load(path, sr=sample_rate)
            
            # If too short, repeat the audio
            if len(y)/sr < min_duration:
                repeat_factor = int(np.ceil(min_duration*sr/len(y)))
                y = np.tile(y, repeat_factor)[:int(min_duration*sr)]
                
            waveforms.append(y)
        return waveforms

    def gated_fusion(self, video_features, audio_features):
        """Implement gated fusion mechanism"""
        # Concatenate features
        combined = torch.cat((video_features, audio_features), dim=-1)
        
        # Compute gates
        video_gate = torch.sigmoid(self.video_gate(combined))
        audio_gate = torch.sigmoid(self.audio_gate(combined))
        
        # Gated features
        gated_video = video_features * video_gate
        gated_audio = audio_features * audio_gate
        
        return gated_video + gated_audio

    def cross_modal_attention(self, video_features, audio_features):
        """Apply cross-modal attention"""
        # Reshape for multi-head attention
        video_features_attn = video_features.unsqueeze(0)
        audio_features_attn = audio_features.unsqueeze(0)
        
        # Cross-modal attention
        attn_output, _ = self.cross_attention(
            video_features_attn, 
            audio_features_attn, 
            audio_features_attn
        )
        
        return attn_output.squeeze(0)


    def forward(self, video, audio_paths):
        # Extract video features
        video = video.permute(0, 2, 1, 3, 4)
        video_outputs = self.backbone(video)
        # video_features = video_outputs.last_hidden_state[:, 0, :]
        video_features = video_outputs
        print(video_features.shape)
        # Extract audio features
        waveforms = self.preprocess_audio(audio_paths)
        audio_inputs = self.audio_feature_extractor(
            waveforms, 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding=True, 
            return_attention_mask=True
        ).to(video.device)
        
        audio_outputs = self.ast(**audio_inputs)
        audio_features = audio_outputs.last_hidden_state[:, 0, :]

        # Advanced Fusion Techniques
        # 1. Gated Fusion
        gated_features = self.gated_fusion(video_features, audio_features)
        
        # 2. Cross-Modal Attention
        cross_modal_features = self.cross_modal_attention(video_features, audio_features)
        
        # 3. Residual Fusion
        fused_features = torch.cat((gated_features, cross_modal_features), dim=-1)
        fused_features = self.fusion_layer(fused_features)
        
        # Classification
        logits = self.classifier(fused_features)
        
        return logits/self.learnable_temperature

class UniformerVAFusionExecuter:
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
        model = AudioVideoFusionClassifier(num_frames=num_frames, num_classes=num_classes).to(gpu_id)
        if distributed: 
            self.model = DDP(model, device_ids=[gpu_id])
        else: 
            self.model = model
        for p in self.model.parameters():
            p.requires_grad = True
        self.lr = learning_rate
        self.optimizer = AdamW([{"params": self.model.parameters(), "lr": self.lr}])
        self.scheduler = CosineAnnealingWarmRestarts(optimizer = self.optimizer, T_0=10)
        
        self.logger = logger
        self.logger.info(f"[INFO] Initialized TimeSformerEpochBasedWithAugmentationExecutor on GPU {gpu_id}")

    def get_model(self):
            return self.model.module if self.distributed else self.model
    
    def _train_batch(self, data, audio, label):
        self.optimizer.zero_grad()
        output = self.model(data, audio)
        loss_this = self.criterion(output, label) # from label.long() 
        loss_this.backward()
        self.optimizer.step()
        return loss_this.item()

    def _train_epoch(self, epoch):
        self.model.train()
        loss_meter = AverageMeter()
        start_time = time.time()
        for data,audio, label in tqdm(self.train_loader):

            # New augmentation
            data, label = mixup_fn(data, label)
            
            data, label = data.to(self.gpu_id),  label.to(self.gpu_id)
            loss_this = self._train_batch(data, audio, label)
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
        
        for data, audio, label in tqdm(self.test_loader):
            data, label = data.to(self.gpu_id), label.long().to(self.gpu_id)
            with torch.no_grad():
                output = self.model(data, audio)
                
            # Update all metrics
            for name, metric in self.eval_metrics.items():
                metric_value = metric(output, label)
                metric_meters[name].update(metric_value.item(), data.shape[0])
        
        metrics = {name: meter.avg for name, meter in metric_meters.items()}
        for name, value in metrics.items():
            self.logger.info(f"[INFO] Test {name}: {value * 100:.2f}")
        return metrics

    
def save(self, file_path="./checkpoint.pth"):
    # Lưu toàn bộ state_dict của mô hình thay vì từng thành phần riêng lẻ
    model_state_dict = self.model.state_dict()
    
    # Lưu state_dict của optimizer
    optimizer_state_dict = self.optimizer.state_dict()
    
    # Lưu thêm các thành phần khác của mô hình fusion
    checkpoint = {
        "model": model_state_dict,  # Toàn bộ state_dict của mô hình
        "optimizer": optimizer_state_dict,
        "ast_config": self.model.ast.config,  # Cấu hình của AST model
        "backbone_config": self.model.backbone.config,  # Cấu hình của TimesformerModel
        "num_frames": self.model.num_frames,
        "num_classes": self.model.num_classes,
        "hidden_size": self.model.backbone.config.hidden_size,
        "audio_feature_extractor_config": self.model.audio_feature_extractor.config
    }
    
    # Lưu file checkpoint
    torch.save(checkpoint, file_path)
    print(f"Model saved to {file_path}")
        
    def load(self, file_path, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
        checkpoint = torch.load(file_path, map_location=device)
        
        # Khởi tạo lại model nếu cần
        if not hasattr(self, 'model') or self.model is None:
            self.model = AudioVideoFusionClassifier(
                num_frames=checkpoint['num_frames'],
                num_classes=checkpoint['num_classes'],
                hidden_size=checkpoint['hidden_size']
            )
        
        # Load state dict
        self.model.load_state_dict(checkpoint['model'])
        
        # Load optimizer state nếu có
        if hasattr(self, 'optimizer') and 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        # Load các thông tin training
        if 'epoch' in checkpoint:
            self.current_epoch = checkpoint['epoch']
        
        if 'best_metric' in checkpoint:
            self.best_metric = checkpoint['best_metric']
        
        self.model.to(device)
        print(f"Model loaded from {file_path}")
        return self.model
