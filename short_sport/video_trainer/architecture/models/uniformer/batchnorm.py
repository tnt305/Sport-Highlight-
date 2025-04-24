import torch.nn as nn

def bn_3d(dim):
    return nn.BatchNorm3d(dim)
