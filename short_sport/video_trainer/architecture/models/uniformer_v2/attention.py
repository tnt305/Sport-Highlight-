import torch
from torch import nn
from short_sport.video_trainer.architecture.models.uniformer_v2.block import ResidualAttentionBlock
from short_sport.video_trainer.architecture.models.uniformer_v2.modules import Extractor

class Transformer(nn.Module):
    def __init__(
            self, width, layers, heads, attn_mask=None, backbone_drop_path_rate=0., 
            use_checkpoint=False, checkpoint_num=[0], t_size=8, dw_reduction=2,
            no_lmhra=False, double_lmhra=True,
            return_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            n_layers=12, n_dim=768, n_head=12, mlp_factor=4.0, drop_path_rate=0.,
            mlp_dropout=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], 
            cls_dropout=0.5, num_classes=400,
        ):
        super().__init__()
        self.T = t_size
        self.return_list = return_list
        # backbone
        b_dpr = [x.item() for x in torch.linspace(0, backbone_drop_path_rate, layers)]
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                width, heads, attn_mask, 
                drop_path=b_dpr[i],
                dw_reduction=dw_reduction,
                no_lmhra=no_lmhra,
                double_lmhra=double_lmhra,
            ) for i in range(layers)
        ])
        # checkpoint
        self.use_checkpoint = use_checkpoint
        self.checkpoint_num = checkpoint_num
        self.n_layers = n_layers
        print(f'Use checkpoint: {self.use_checkpoint}')
        print(f'Checkpoint number: {self.checkpoint_num}')

        # global block
        assert n_layers == len(return_list)
        if n_layers > 0:
            self.temporal_cls_token = nn.Parameter(torch.zeros(1, 1, n_dim))
            self.dpe = nn.ModuleList([
                nn.Conv3d(n_dim, n_dim, kernel_size=3, stride=1, padding=1, bias=True, groups=n_dim)
                for i in range(n_layers)
            ])
            for m in self.dpe:
                nn.init.constant_(m.bias, 0.)
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
            self.dec = nn.ModuleList([
                Extractor(
                    n_dim, n_head, mlp_factor=mlp_factor, 
                    dropout=mlp_dropout[i], drop_path=dpr[i],
                ) for i in range(n_layers)
            ])
            self.balance = nn.Parameter(torch.zeros((n_dim)))
            self.sigmoid = nn.Sigmoid()
        # projection
        self.proj = nn.Sequential(
            nn.LayerNorm(n_dim),
            nn.Dropout(cls_dropout),
            nn.Linear(n_dim, num_classes),
        )

    def forward(self, x):
        T_down = self.T
        L, NT, C = x.shape
        N = NT // T_down
        H = W = int((L - 1) ** 0.5)

        if self.n_layers > 0:
            cls_token = self.temporal_cls_token.repeat(1, N, 1)

        j = -1
        for i, resblock in enumerate(self.resblocks):
            if self.use_checkpoint and i < self.checkpoint_num[0]:
                x = resblock(x, self.T, use_checkpoint=True)
            else:
                x = resblock(x, T_down)
            if i in self.return_list:
                j += 1
                tmp_x = x.clone()
                tmp_x = tmp_x.view(L, N, T_down, C)
                # dpe
                _, tmp_feats = tmp_x[:1], tmp_x[1:]
                tmp_feats = tmp_feats.permute(1, 3, 2, 0).reshape(N, C, T_down, H, W)
                tmp_feats = self.dpe[j](tmp_feats).view(N, C, T_down, L - 1).permute(3, 0, 2, 1).contiguous()
                tmp_x[1:] = tmp_x[1:] + tmp_feats
                # global block
                tmp_x = tmp_x.permute(2, 0, 1, 3).flatten(0, 1)  # T * L, N, C
                cls_token = self.dec[j](cls_token, tmp_x)

        if self.n_layers > 0:
            weight = self.sigmoid(self.balance)
            residual = x.view(L, N, T_down, C)[0].mean(1)  # L, N, T, C
            return self.proj((1 - weight) * cls_token[0, :, :] + weight * residual)
        else:
            residual = x.view(L, N, T_down, C)[0].mean(1)  # L, N, T, C
            return self.proj(residual)

