"""
MOT-style action expert: fuse action and WAN hidden tokens by concatenating
sequence tokens and running joint QKV attention.

Design:
- Keep DiT-style AdaLN timestep conditioning.
- Self-attend on action tokens.
- Project action tokens to WAN hidden size, concatenate with video tokens,
  and run joint attention (Q/K/V from concatenated tokens).
- Slice back action tokens and continue FFN update.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Timesteps(nn.Module):
    """Sinusoidal timestep projection."""

    def __init__(self, num_channels: int, flip_sin_to_cos: bool = True, downscale_freq_shift: float = 0.0):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(half, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / (half - self.downscale_freq_shift)
        emb = timesteps.float()[:, None] * exponent[None, :].exp()
        if self.flip_sin_to_cos:
            emb = torch.cat([emb.cos(), emb.sin()], dim=-1)
        else:
            emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.num_channels % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class TimestepEmbedding(nn.Module):
    """Linear -> SiLU -> Linear MLP."""

    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(x)))


class FP32LayerNorm(nn.LayerNorm):
    """LayerNorm compatible with bfloat16 flow."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)


class RMSNorm(nn.Module):
    """RMSNorm used for Q/K normalization."""

    def __init__(self, dim: int, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        if self.weight is not None:
            x = x * self.weight.to(x.dtype)
        return x.to(x.dtype)


def apply_rope_1d(
    q: torch.Tensor,
    k: torch.Tensor,
    theta: float = 10_000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply 1-D rotary position embedding to action Q/K."""
    _, seq_len, _, head_dim = q.shape
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE requires an even head_dim, got {head_dim}.")

    q_dtype = q.dtype
    k_dtype = k.dtype
    pos = torch.arange(seq_len, device=q.device, dtype=torch.float32)
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=q.device, dtype=torch.float32) / head_dim))
    emb = torch.outer(pos, freqs)
    cos = emb.cos()[None, :, None, :]
    sin = emb.sin()[None, :, None, :]

    def _rotate(x: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        x1, x2 = x[..., : head_dim // 2], x[..., head_dim // 2 :]
        rotated = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        return rotated.to(out_dtype)

    return _rotate(q.float(), q_dtype), _rotate(k.float(), k_dtype)


class DitTimeEmbedding(nn.Module):
    """Sinusoidal -> MLP -> 6 * inner_dim AdaLN parameters."""

    def __init__(self, inner_dim: int, freq_dim: int = 256):
        super().__init__()
        self.timesteps_proj = Timesteps(num_channels=freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=freq_dim, time_embed_dim=inner_dim)
        self.act = nn.SiLU()
        self.time_proj = nn.Linear(inner_dim, inner_dim * 6)

    def forward(self, timestep: torch.Tensor, dtype=None) -> torch.Tensor:
        t_freq = self.timesteps_proj(timestep * 1000.0)
        if dtype is not None and t_freq.dtype != dtype:
            t_freq = t_freq.to(dtype)
        temb = self.time_embedder(t_freq)
        timestep_proj = self.time_proj(self.act(temb))
        return timestep_proj.unflatten(1, (6, -1))


class MotJointBlock(nn.Module):
    """Action block that exposes WAN-space QKV for layer-internal joint attention."""

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        joint_num_heads: int = 24,
        video_hidden_size: int = 3072,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)
        self.video_hidden_size = video_hidden_size
        self.joint_num_heads = joint_num_heads
        if video_hidden_size % joint_num_heads != 0:
            raise ValueError(
                f"video_hidden_size ({video_hidden_size}) must be divisible by joint_num_heads ({joint_num_heads})."
            )
        self.joint_dim_head = video_hidden_size // joint_num_heads

        self.norm1 = (
            FP32LayerNorm(dim, eps=eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )
        self.wan_action_qkv = nn.Parameter(
            torch.randn(3, dim, joint_num_heads, self.joint_dim_head) / dim ** 0.5
        )
        self.wan_action_norm_q = RMSNorm(video_hidden_size, eps=eps)
        self.wan_action_norm_k = RMSNorm(video_hidden_size, eps=eps)
        self.wan_action_o = nn.Linear(video_hidden_size, dim)

        self.norm2 = FP32LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )

    def compute_adaln(
        self,
        timestep_proj: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, ...]:
        adaln = self.scale_shift_table.to(dtype) + timestep_proj.to(dtype)
        shift_msa, scale_msa, gate_msa, c_shift, c_scale, c_gate = adaln.unbind(dim=1)
        return (
            shift_msa[:, None, :],
            scale_msa[:, None, :],
            gate_msa[:, None, :],
            c_shift[:, None, :],
            c_scale[:, None, :],
            c_gate[:, None, :],
        )

    def prepare_wan_qkv(
        self,
        x: torch.Tensor,
        adaln: tuple[torch.Tensor, ...],
        num_heads: int,
        head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shift_msa, scale_msa, _, _, _, _ = adaln
        h_action = self.norm1(x).to(x.dtype)
        h_action = h_action * (1.0 + scale_msa) + shift_msa

        if num_heads != self.joint_num_heads or head_dim != self.joint_dim_head:
            raise ValueError(
                f"WAN attention shape mismatch: expected ({self.joint_num_heads}, {self.joint_dim_head}), "
                f"got ({num_heads}, {head_dim})."
            )

        qkv_weight = self.wan_action_qkv.to(x.dtype)
        a_qkv = torch.einsum("BLD,KDNE->KBLNE", h_action, qkv_weight)
        a_q_h, a_k_h, a_v_h = a_qkv[0], a_qkv[1], a_qkv[2]
        batch_size, action_len = x.shape[:2]
        a_q = self.wan_action_norm_q(a_q_h.flatten(-2)).view(batch_size, action_len, num_heads, head_dim)
        a_k = self.wan_action_norm_k(a_k_h.flatten(-2)).view(batch_size, action_len, num_heads, head_dim)
        a_q, a_k = apply_rope_1d(a_q, a_k)
        a_v = a_v_h.view(batch_size, action_len, num_heads, head_dim)
        return a_q, a_k, a_v

    def apply_joint_attention(
        self,
        x: torch.Tensor,
        attn_out_h: torch.Tensor,
        adaln: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        _, _, gate_msa, _, _, _ = adaln
        action_out = self.wan_action_o(attn_out_h.flatten(2).to(x.dtype))
        return x + action_out * gate_msa

    def apply_ffn(
        self,
        x: torch.Tensor,
        adaln: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        _, _, _, c_shift, c_scale, c_gate = adaln
        h = self.norm2(x) * (1.0 + c_scale) + c_shift
        return x + self.ffn(h) * c_gate


class DitActionExpert(nn.Module):
    """MOT-style DiT action expert with joint QKV over concatenated action+video tokens."""

    def __init__(
        self,
        hidden_size: int = 1024,
        joint_num_heads: int = 24,
        ffn_dim: int = 4096,
        num_layers: int = 30,
        video_hidden_size: int = 3072,
        freq_dim: int = 256,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.time_embed = DitTimeEmbedding(inner_dim=hidden_size, freq_dim=freq_dim)
        self.blocks = nn.ModuleList([
            MotJointBlock(
                dim=hidden_size,
                ffn_dim=ffn_dim,
                joint_num_heads=joint_num_heads,
                video_hidden_size=video_hidden_size,
                cross_attn_norm=cross_attn_norm,
                eps=eps,
            )
            for _ in range(num_layers)
        ])

        self.norm_out = FP32LayerNorm(hidden_size, eps=eps, elementwise_affine=False)
        self.final_scale_shift = nn.Parameter(torch.randn(1, 2, hidden_size) / hidden_size ** 0.5)
        self.final_proj = nn.Linear(hidden_size, hidden_size * 2)

    def compute_timestep_proj(self, timestep: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return self.time_embed(timestep, dtype=dtype)

    def prepare_wan_joint_qkv(
        self,
        x: torch.Tensor,
        timestep_proj: torch.Tensor,
        layer_idx: int,
        num_heads: int,
        head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        block = self.blocks[layer_idx]
        adaln = block.compute_adaln(timestep_proj, dtype=x.dtype)
        a_q, a_k, a_v = block.prepare_wan_qkv(x, adaln, num_heads=num_heads, head_dim=head_dim)
        return a_q, a_k, a_v, adaln

    def apply_wan_joint_output(
        self,
        x: torch.Tensor,
        attn_out_h: torch.Tensor,
        adaln: tuple[torch.Tensor, ...],
        layer_idx: int,
    ) -> torch.Tensor:
        return self.blocks[layer_idx].apply_joint_attention(x, attn_out_h, adaln)

    def apply_ffn(
        self,
        x: torch.Tensor,
        adaln: tuple[torch.Tensor, ...],
        layer_idx: int,
    ) -> torch.Tensor:
        return self.blocks[layer_idx].apply_ffn(x, adaln)

    def finalize(self, x: torch.Tensor, timestep_proj: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        temb_proxy = timestep_proj[:, 0, :]
        proj_dtype = self.final_proj.weight.dtype
        final_adaln = self.final_proj(temb_proxy.to(proj_dtype)).unflatten(1, (2, -1)).to(dtype)
        final_adaln = self.final_scale_shift.to(dtype) + final_adaln
        shift_out, scale_out = final_adaln.unbind(dim=1)
        shift_out = shift_out[:, None, :]
        scale_out = scale_out[:, None, :]
        return self.norm_out(x) * (1.0 + scale_out) + shift_out

    def forward(self, action_embs: torch.Tensor, video_features: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(
            "DitActionExpert.forward is not used in wan_mot. "
            "Run the interleaved WAN/action path from DitVideoFlowMatching instead."
        )
