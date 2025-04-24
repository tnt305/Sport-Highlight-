import torch
import torch.nn as nn


class TwoWayLoss(nn.Module):
    '''
    Result of two-way loss with Tp =4 and Tn = 1 is 47.25
    '''
    def __init__(self, Tp=4.0, Tn=1.0, nINF = -100):
        super(TwoWayLoss, self).__init__()
        self.Tp = Tp
        self.Tn = Tn
        self.nINF = nINF
    def forward(self, x, y):
        class_mask = (y > 0).any(dim=0)
        sample_mask = (y > 0).any(dim=1)

        # Calculate hard positive/negative logits
        pmask = y.masked_fill(y <= 0, self.nINF).masked_fill(y > 0, float(0.0))
        plogit_class = torch.logsumexp(-x/self.Tp + pmask, dim=0).mul(self.Tp)[class_mask]
        plogit_sample = torch.logsumexp(-x/self.Tp + pmask, dim=1).mul(self.Tp)[sample_mask]
    
        nmask = y.masked_fill(y != 0, self.nINF).masked_fill(y == 0, float(0.0))
        nlogit_class = torch.logsumexp(x/self.Tn + nmask, dim=0).mul(self.Tn)[class_mask]
        nlogit_sample = torch.logsumexp(x/self.Tn + nmask, dim=1).mul(self.Tn)[sample_mask]

        return F.softplus(nlogit_class + plogit_class).mean() + \
                F.softplus(nlogit_sample + plogit_sample).mean()