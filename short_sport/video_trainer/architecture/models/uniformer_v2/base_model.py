import torch 
import torch.nn as nn
from short_sport.video_trainer.architecture.models.uniformer_v2.layer_norm import LayerNorm
from short_sport.video_trainer.architecture.models.uniformer_v2.attention import Transformer

class VisionTransformer(nn.Module):
    def __init__(
        self, 
        # backbone
        input_resolution, patch_size, width, layers, heads, output_dim, backbone_drop_path_rate=0.,
        use_checkpoint=False, checkpoint_num=[0], t_size=8, kernel_size=3, dw_reduction=1.5,
        temporal_downsample=True,
        no_lmhra=-False, double_lmhra=True,
        # global block
        return_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        n_layers=12, n_dim=768, n_head=12, mlp_factor=4.0, drop_path_rate=0.,
        mlp_dropout=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], 
        cls_dropout=0.5, num_classes=400,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        padding = (kernel_size - 1) // 2
        if temporal_downsample:
            self.conv1 = nn.Conv3d(3, width, (kernel_size, patch_size, patch_size), (2, patch_size, patch_size), (padding, 0, 0), bias=False)
            t_size = t_size // 2
        else:
            self.conv1 = nn.Conv3d(3, width, (1, patch_size, patch_size), (1, patch_size, patch_size), (0, 0, 0), bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        
        self.transformer = Transformer(
            width, layers, heads, dw_reduction=dw_reduction, 
            backbone_drop_path_rate=backbone_drop_path_rate, 
            use_checkpoint=use_checkpoint, checkpoint_num=checkpoint_num, t_size=t_size,
            no_lmhra=no_lmhra, double_lmhra=double_lmhra,
            return_list=return_list, n_layers=n_layers, n_dim=n_dim, n_head=n_head, 
            mlp_factor=mlp_factor, drop_path_rate=drop_path_rate, mlp_dropout=mlp_dropout, 
            cls_dropout=cls_dropout, num_classes=num_classes,
        )

    def forward(self, x):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        N, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(N * T, H * W, C)
        
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        out = self.transformer(x)
        return out
