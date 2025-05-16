import torch
import torch.nn as nn
from collections import OrderedDict
from timm.models.layers import DropPath
from short_sport.video_trainer.architecture.models.uniformer_v2.activation import QuickGELU

class Local_MHRA(nn.Module):
    def __init__(self, d_model, dw_reduction=1.5, pos_kernel_size=3):
        super().__init__() 

        padding = pos_kernel_size // 2
        re_d_model = int(d_model // dw_reduction)
        self.pos_embed = nn.Sequential(
            nn.BatchNorm3d(d_model),
            nn.Conv3d(d_model, re_d_model, kernel_size=1, stride=1, padding=0),
            nn.Conv3d(re_d_model, re_d_model, kernel_size=(pos_kernel_size, 1, 1), stride=(1, 1, 1), padding=(padding, 0, 0), groups=re_d_model),
            nn.Conv3d(re_d_model, d_model, kernel_size=1, stride=1, padding=0),
        )

        # init zero
        print('Init zero for Conv in pos_emb')
        nn.init.constant_(self.pos_embed[3].weight, 0)
        nn.init.constant_(self.pos_embed[3].bias, 0)

    def forward(self, x):
        return self.pos_embed(x)
    

class Extractor(nn.Module):
    def __init__(
            self, d_model, n_head, attn_mask=None,
            mlp_factor=4.0, dropout=0.0, drop_path=0.0,
        ):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        print(f'Drop path rate: {drop_path}')
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        d_mlp = round(mlp_factor * d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_mlp)),
            ("gelu", QuickGELU()),
            ("dropout", nn.Dropout(dropout)),
            ("c_proj", nn.Linear(d_mlp, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.ln_3 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

        # zero init
        nn.init.xavier_uniform_(self.attn.in_proj_weight)
        nn.init.constant_(self.attn.out_proj.weight, 0.)
        nn.init.constant_(self.attn.out_proj.bias, 0.)
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.constant_(self.mlp[-1].weight, 0.)
        nn.init.constant_(self.mlp[-1].bias, 0.)

    def attention(self, x, y):
        d_model = self.ln_1.weight.size(0)
        q = (x @ self.attn.in_proj_weight[:d_model].T) + self.attn.in_proj_bias[:d_model]

        k = (y @ self.attn.in_proj_weight[d_model:-d_model].T) + self.attn.in_proj_bias[d_model:-d_model]
        v = (y @ self.attn.in_proj_weight[-d_model:].T) + self.attn.in_proj_bias[-d_model:]
        Tx, Ty, N = q.size(0), k.size(0), q.size(1)
        q = q.view(Tx, N, self.attn.num_heads, self.attn.head_dim).permute(1, 2, 0, 3)
        k = k.view(Ty, N, self.attn.num_heads, self.attn.head_dim).permute(1, 2, 0, 3)
        v = v.view(Ty, N, self.attn.num_heads, self.attn.head_dim).permute(1, 2, 0, 3)
        aff = (q @ k.transpose(-2, -1) / (self.attn.head_dim ** 0.5))

        aff = aff.softmax(dim=-1)
        out = aff @ v
        out = out.permute(2, 0, 1, 3).flatten(2)
        out = self.attn.out_proj(out)
        return out

    def forward(self, x, y):
        x = x + self.drop_path(self.attention(self.ln_1(x), self.ln_3(y)))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x
