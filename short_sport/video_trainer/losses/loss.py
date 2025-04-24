import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, size_average=True):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


nINF = -100

class TwoWayLoss(nn.Module):
    '''
    Result of two-way loss with Tp =4 and Tn = 1 is 47.25
    '''
    def __init__(self, Tp=4., Tn=1.):
        super(TwoWayLoss, self).__init__()
        self.Tp = Tp
        self.Tn = Tn

    def forward(self, x, y):
        class_mask = (y > 0).any(dim=0)
        sample_mask = (y > 0).any(dim=1)

        # Calculate hard positive/negative logits
        pmask = y.masked_fill(y <= 0, nINF).masked_fill(y > 0, float(0.0))
        plogit_class = torch.logsumexp(-x/self.Tp + pmask, dim=0).mul(self.Tp)[class_mask]
        plogit_sample = torch.logsumexp(-x/self.Tp + pmask, dim=1).mul(self.Tp)[sample_mask]
    
        nmask = y.masked_fill(y != 0, nINF).masked_fill(y == 0, float(0.0))
        nlogit_class = torch.logsumexp(x/self.Tn + nmask, dim=0).mul(self.Tn)[class_mask]
        nlogit_sample = torch.logsumexp(x/self.Tn + nmask, dim=1).mul(self.Tn)[sample_mask]

        return F.softplus(nlogit_class + plogit_class).mean() + \
                F.softplus(nlogit_sample + plogit_sample).mean()
                

class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations
    
    Result of AsymetricLoss with gamma_neg=4, gamma_pos=1 = 46.71
    '''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()
    
class CalibratedRankingLoss(nn.Module):
    def __init__(self, temperature=1.0, margin=0.5, alpha=0.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * temperature)
        self.margin = margin
        self.alpha = alpha  # Trọng số cân bằng giữa classification và ranking
        self.ce_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, logits, labels):
        # Temperature scaling cho calibration
        scaled_logits = logits / self.temperature
        
        # Classification loss
        cls_loss = self.ce_loss(scaled_logits, labels)
        
        # Ranking loss (Triplet-like)
        batch_size = logits.size(0)
        if batch_size <= 1:
            return cls_loss
            
        # Tạo ma trận khoảng cách giữa các embedding
        similarity = torch.matmul(scaled_logits, scaled_logits.t())
        
        # Tạo mask cho positive pairs (cùng class) và negative pairs (khác class)
        label_matrix = torch.matmul(labels, labels.t())
        positive_mask = label_matrix > 0
        negative_mask = label_matrix == 0
        
        # Triplet loss
        positive_pairs = similarity[positive_mask]
        negative_pairs = similarity[negative_mask]
        
        if positive_pairs.numel() > 0 and negative_pairs.numel() > 0:
            # Đảm bảo positive pairs có similarity cao hơn negative pairs một margin
            ranking_loss = torch.clamp(self.margin - positive_pairs.mean() + negative_pairs.mean(), min=0.0)
        else:
            ranking_loss = torch.tensor(0.0, device=logits.device)
        
        # Kết hợp loss
        total_loss = self.alpha * cls_loss + (1 - self.alpha) * ranking_loss
        
        return total_loss

class CorrelationAwareLoss(nn.Module):
    def __init__(self, cooccur_matrix, base_loss=nn.BCEWithLogitsLoss(), alpha=0.3):
        super().__init__()
        self.base_loss = base_loss
        self.alpha = alpha
        self.register_buffer('cooccur', torch.from_numpy(cooccur_matrix.toarray()).float())

    def forward(self, logits, labels):
        # Base loss
        loss = self.base_loss(logits, labels.float())
        
        # Correlation penalty
        pred_probs = torch.sigmoid(logits)
        expected_corr = pred_probs @ self.cooccur  # Dự đoán quan hệ nhãn
        correlation_loss = torch.norm(expected_corr - labels.float(), p=2)
        
        return loss + self.alpha * correlation_loss

