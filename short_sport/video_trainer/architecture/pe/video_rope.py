import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding

class VideoRoPE(nn.Module):
    def __init__(self, dim_size, patch_size=(16, 16), image_size=224):
        super().__init__()
        self.dim_size = dim_size
        self.patch_size = patch_size[0] if isinstance(patch_size, tuple) else patch_size
        self.image_size = image_size
        self.num_patches_per_frame = (image_size // self.patch_size) ** 2
        # Không cố định max_frames, sẽ tính động trong forward

        # Tần số quay
        self.theta = torch.tensor([10000 ** (-2 * i / dim_size) for i in range(dim_size // 2)])

    def get_rotary_matrix(self, positions, device):
        theta = self.theta.to(device)
        angles = positions.unsqueeze(-1) * theta
        cosines = torch.cos(angles)
        sines = torch.sin(angles)
        return cosines, sines

    def apply_rotary_embedding(self, x, cosines, sines):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        x_rotated = torch.cat([
            x1 * cosines - x2 * sines,
            x1 * sines + x2 * cosines
        ], dim=-1)
        return x_rotated

    def forward(self, x, num_frames):
        batch_size, num_patches_total, dim = x.shape
        device = x.device

        # Tính số patch thực tế (trừ token [CLS] nếu có)
        num_patches_data = num_patches_total - 1  # Giả sử có token [CLS]
        num_patches_per_frame = self.num_patches_per_frame

        # Tính vị trí thời gian
        temporal_positions = torch.arange(num_frames, device=device).repeat_interleave(num_patches_per_frame)
        temporal_positions = temporal_positions[:num_patches_data]

        # Tính vị trí không gian
        h_patches = w_patches = int(math.sqrt(num_patches_per_frame))
        spatial_x = torch.arange(w_patches, device=device).repeat(h_patches)
        spatial_y = torch.arange(h_patches, device=device).repeat_interleave(w_patches)
        spatial_positions = torch.stack([spatial_x, spatial_y], dim=-1)
        spatial_positions = spatial_positions.repeat(num_frames, 1)[:num_patches_data]

        # Tạo ma trận quay
        temporal_cos, temporal_sin = self.get_rotary_matrix(temporal_positions, device)
        spatial_x_cos, spatial_x_sin = self.get_rotary_matrix(spatial_positions[:, 0], device)
        spatial_y_cos, spatial_y_sin = self.get_rotary_matrix(spatial_positions[:, 1], device)

        # Áp dụng RoPE
        x_data = x[:, 1:, :]  # Bỏ token [CLS] để áp dụng RoPE
        x_data = self.apply_rotary_embedding(x_data, temporal_cos, temporal_sin)
        x_data = self.apply_rotary_embedding(x_data, spatial_x_cos, spatial_x_sin)
        x_data = self.apply_rotary_embedding(x_data, spatial_y_cos, spatial_y_sin)

        # Gộp lại token [CLS] (không áp dụng RoPE cho [CLS])
        x = torch.cat([x[:, :1, :], x_data], dim=1)
        return x


class VideoRoPETimesformer(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.rotary_emb = RotaryEmbedding(dim=64)  # dimension should match head dimension
        
    def forward(self, x):
        # Get attention layers from Timesformer
        for layer in self.backbone.encoder.layer:
            # Apply rotary embedding to attention queries and keys
            attn = layer.attention.self
            q, k = attn.query(x), attn.key(x)
            q, k = self.apply_rotary_emb(q, k)
            # Rest of attention computation...
        return self.backbone(x)
    
    def apply_rotary_emb(self, q, k):
        # Rearrange for rotary embedding
        q = rearrange(q, 'b n (h d) -> b h n d', h=8)  # 8 heads
        k = rearrange(k, 'b n (h d) -> b h n d', h=8)
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)
        return rearrange(q, 'b h n d -> b n (h d)'), rearrange(k, 'b h n d -> b n (h d)')

class VideoRoPESelfAttention(nn.Module):
    """Self-attention layer with VideoRoPE positional embeddings"""
    def __init__(self, embed_dim, num_heads, dropout, dim_size):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Projection layers
        self.qkv_proj = nn.Linear(embed_dim, embed_dim*3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # VideoRoPE component
        self.rope = ImprovedVideoRoPE(dim_size)
        
    def forward(self, hidden_states, output_attentions=False, **kwargs):
        """
        Forward pass with compatibility for TimeSformer's expected parameters
        
        Args:
            hidden_states: Input tensor of shape [B, N, D]
            output_attentions: Whether to return attention weights
            **kwargs: Additional keyword arguments for compatibility
        """
        B, N, _ = hidden_states.shape
        
        # TimeSformer uses 197 tokens per frame (1 CLS + 196 patch tokens)
        num_frames = N // 197
        
        # Apply VideoRoPE
        x = self.rope(hidden_states, num_frames)
        
        # Project queries, keys, values
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply attention mask if provided (handle TimeSformer's expected parameters)
        attention_mask = kwargs.get('attention_mask', None)
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
            
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Combine values
        context_layer = (attn_probs @ v).transpose(1, 2).reshape(B, N, -1)
        attention_output = self.out_proj(context_layer)
        
        # Return attention weights if requested (match TimeSformer's return format)
        outputs = (attention_output,)
        if output_attentions:
            outputs = outputs + (attn_probs,)
            
        return outputs

class ImprovedVideoRoPE(nn.Module):
    """Improved Rotary Position Embedding for video understanding"""
    def __init__(self, dim_size):
        super().__init__()
        self.dim_per_component = dim_size // 3  # Allocate 1/3 of dims for each component
        
        # Initialize frequency for rotary embeddings
        self.freqs = self._precompute_freqs(dim_size)
        self.max_seq_len = 1000  # Maximum sequence length to cache
        self._init_cached_rotary_emb()
        
    def _precompute_freqs(self, dim):
        """Precompute frequencies for rotary embeddings"""
        # Implementation details...
        freqs = 1.0 / (10000 ** (torch.arange(0, self.dim_per_component, 2).float() / self.dim_per_component))
        return freqs
        
    def _init_cached_rotary_emb(self):
        """Initialize cached sin/cos values"""
        t = torch.arange(self.max_seq_len).type_as(self.freqs)
        freqs = torch.einsum('i,j->ij', t, self.freqs)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
        
    def forward(self, x, num_frames):
        """
        Apply rotary embeddings to input tensor
        
        Args:
            x: Input tensor of shape [B, N, D]
            num_frames: Number of frames in the video
        """
        B, N, D = x.shape
        
        # Calculate tokens per frame (should be 197 for TimeSformer)
        tokens_per_frame = N // num_frames
        
        # Debug print to verify our understanding
        # print(f"Tokens per frame: {tokens_per_frame}, Total tokens: {N}, Frames: {num_frames}")
        
        # Reshape to separate frames and tokens
        x_reshaped = x.reshape(B, num_frames, tokens_per_frame, D)
        
        # Generate position IDs for temporal dimension
        # First token in each frame is CLS token - we'll skip rotary embeddings for it
        position_ids = torch.arange(tokens_per_frame, device=x.device)
        
        # Create a mask for CLS tokens (assuming first token in each frame is CLS)
        cls_mask = position_ids == 0
        
        # Apply rotary embeddings separately for each frame
        x_rotated = x_reshaped.clone()
        
        # Only apply rotations to non-CLS tokens
        non_cls_indices = ~cls_mask
        if non_cls_indices.any():
            # Extract non-CLS tokens
            x_non_cls = x_reshaped[:, :, non_cls_indices, :]
            
            # Apply rotary embeddings to the non-CLS tokens
            dim_per_component = self.dim_per_component
            
            # Split into components
            x1 = x_non_cls[..., :dim_per_component]
            x2 = x_non_cls[..., dim_per_component:2*dim_per_component]
            
            # Get position IDs for non-CLS tokens only
            non_cls_positions = position_ids[non_cls_indices] - 1  # Adjust positions
            
            # Get sin and cos values
            cos = self.cos_cached[non_cls_positions]
            sin = self.sin_cached[non_cls_positions]
            
            # Apply rotary embeddings
            x_rotated[:, :, non_cls_indices, :dim_per_component] = (
                x1 * cos.unsqueeze(0).unsqueeze(0) - 
                x2 * sin.unsqueeze(0).unsqueeze(0)
            )
            
            x_rotated[:, :, non_cls_indices, dim_per_component:2*dim_per_component] = (
                x2 * cos.unsqueeze(0).unsqueeze(0) + 
                x1 * sin.unsqueeze(0).unsqueeze(0)
            )
            
        # Reshape back to original shape
        return x_rotated.reshape(B, N, D)
