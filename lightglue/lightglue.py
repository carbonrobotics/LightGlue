import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

try:
    from flash_attn.modules.mha import FlashCrossAttention
except ModuleNotFoundError:
    FlashCrossAttention = None

if FlashCrossAttention or hasattr(F, "scaled_dot_product_attention"):
    FLASH_AVAILABLE = True
else:
    FLASH_AVAILABLE = False

torch.backends.cudnn.deterministic = True


AMP_CUSTOM_FWD_F32 = (
    torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "custom_fwd")
    else torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
)


@AMP_CUSTOM_FWD_F32
def normalize_keypoints(
    kpts: torch.Tensor, size: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if size is None:
        size = 1 + kpts.max(-2).values - kpts.min(-2).values
    elif not isinstance(size, torch.Tensor):
        size = torch.tensor(size, device=kpts.device, dtype=kpts.dtype)
    size = size.to(kpts)
    shift = size / 2
    scale = size.max(-1).values / 2
    kpts = (kpts - shift[..., None, :]) / scale[..., None, None]
    return kpts


def pad_to_length(x: torch.Tensor, length: int) -> Tuple[torch.Tensor]:
    if length <= x.shape[-2]:
        return x, torch.ones_like(x[..., :1], dtype=torch.bool)
    pad = torch.ones(
        *x.shape[:-2], length - x.shape[-2], x.shape[-1], device=x.device, dtype=x.dtype
    )
    y = torch.cat([x, pad], dim=-2)
    mask = torch.zeros(*y.shape[:-1], 1, dtype=torch.bool, device=x.device)
    mask[..., : x.shape[-2], :] = True
    return y, mask


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def apply_cached_rotary_emb(freqs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return (t * freqs[0]) + (rotate_half(t) * freqs[1])


# TensorRT-compatible versions of the above functions
def rotate_half_trt(x: torch.Tensor) -> torch.Tensor:
    """TensorRT-compatible version matching original rotate_half behavior.

    Original rotate_half treats the last dimension as pairs: [a0, b0, a1, b1, ...]
    and rotates each pair to [-b0, a0, -b1, a1, ...].

    This TRT version uses reshape instead of unflatten/unbind/stack.
    """
    # Reshape to pairs: [..., D] -> [..., D//2, 2]
    shape = list(x.shape)
    new_shape = shape[:-1] + [shape[-1] // 2, 2]
    x = x.reshape(*new_shape)
    # Extract pairs: x[..., 0] = a values, x[..., 1] = b values
    x1 = x[..., 0]  # [..., D//2]
    x2 = x[..., 1]  # [..., D//2]
    # Stack rotated pairs: [-b, a] for each pair, then flatten
    # Result shape: [..., D//2, 2] -> [..., D]
    rotated = torch.stack((-x2, x1), dim=-1)
    return rotated.reshape(*shape)


def apply_cached_rotary_emb_trt(cos: torch.Tensor, sin: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """TensorRT-compatible version that takes cos and sin separately."""
    return (t * cos) + (rotate_half_trt(t) * sin)


def repeat_interleave_trt(x: torch.Tensor, repeats: int, dim: int) -> torch.Tensor:
    """TensorRT-compatible version of repeat_interleave.

    For example, if x has shape [2, N, D] and repeats=2, dim=-1:
    - Unsqueeze to [2, N, D, 1]
    - Expand to [2, N, D, 2]
    - Reshape to [2, N, D*2]
    """
    # Normalize negative dimension
    ndim = x.dim()
    if dim < 0:
        dim = ndim + dim

    # Add a dimension after dim for repeating: [2, N, D] -> [2, N, D, 1]
    x = x.unsqueeze(dim + 1)

    # Get the expanded shape and expand
    expand_shape = list(x.shape)
    expand_shape[dim + 1] = repeats
    x = x.expand(*expand_shape)

    # Flatten the repeated dimension with the original dimension
    # [2, N, D, repeats] -> [2, N, D*repeats]
    shape = list(x.shape)
    new_shape = shape[:dim] + [shape[dim] * repeats] + shape[dim + 2:]
    return x.reshape(*new_shape)


class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, M: int, dim: int, F_dim: int = None, gamma: float = 1.0,
                 repeat_fn: callable = None, return_separated_encoding: bool = False) -> None:
        super().__init__()
        F_dim = F_dim if F_dim is not None else dim
        self.gamma = gamma
        self.Wr = nn.Linear(M, F_dim // 2, bias=False)
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma**-2)
        self.repeat_fn = repeat_fn if repeat_fn is not None else torch.Tensor.repeat_interleave
        self.return_separated_encoding = return_separated_encoding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """encode position vector"""
        projected = self.Wr(x)
        cosines, sines = torch.cos(projected), torch.sin(projected)
        if self.return_separated_encoding:
            # Return cos and sin separately for TensorRT-compatible rotary embedding
            cos_emb = self.repeat_fn(cosines.unsqueeze(-3), 2, dim=-1)
            sin_emb = self.repeat_fn(sines.unsqueeze(-3), 2, dim=-1)
            return cos_emb, sin_emb
        emb = torch.stack([cosines, sines], 0).unsqueeze(-3)
        return self.repeat_fn(emb, 2, dim=-1)


class TokenConfidence(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.token = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        """get confidence tokens"""
        return (
            self.token(desc0.detach()).squeeze(-1),
            self.token(desc1.detach()).squeeze(-1),
        )


class Attention(nn.Module):
    def __init__(self, allow_flash: bool) -> None:
        super().__init__()
        if allow_flash and not FLASH_AVAILABLE:
            warnings.warn(
                "FlashAttention is not available. For optimal speed, "
                "consider installing torch >= 2.0 or flash-attn.",
                stacklevel=2,
            )
        self.enable_flash = allow_flash and FLASH_AVAILABLE
        self.has_sdp = hasattr(F, "scaled_dot_product_attention")
        if allow_flash and FlashCrossAttention:
            self.flash_ = FlashCrossAttention()
        if self.has_sdp:
            torch.backends.cuda.enable_flash_sdp(allow_flash)

    def forward(self, q, k, v, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if q.shape[-2] == 0 or k.shape[-2] == 0:
            return q.new_zeros((*q.shape[:-1], v.shape[-1]))
        if self.enable_flash and q.device.type == "cuda":
            # use torch 2.0 scaled_dot_product_attention with flash
            if self.has_sdp:
                args = [x.half().contiguous() for x in [q, k, v]]
                v = F.scaled_dot_product_attention(*args, attn_mask=mask).to(q.dtype)
                return v if mask is None else v.nan_to_num()
            else:
                assert mask is None
                q, k, v = [x.transpose(-2, -3).contiguous() for x in [q, k, v]]
                m = self.flash_(q.half(), torch.stack([k, v], 2).half())
                return m.transpose(-2, -3).to(q.dtype).clone()
        elif self.has_sdp:
            args = [x.contiguous() for x in [q, k, v]]
            v = F.scaled_dot_product_attention(*args, attn_mask=mask)
            return v if mask is None else v.nan_to_num()
        else:
            s = q.shape[-1] ** -0.5
            sim = torch.einsum("...id,...jd->...ij", q, k) * s
            if mask is not None:
                sim.masked_fill(~mask, -float("inf"))
            attn = F.softmax(sim, -1)
            return torch.einsum("...ij,...jd->...id", attn, v)


class SelfBlock(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, flash: bool = False, bias: bool = True
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0
        self.head_dim = self.embed_dim // num_heads
        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.inner_attn = Attention(flash)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        encoding: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qkv = self.Wqkv(x)
        qkv = qkv.unflatten(-1, (self.num_heads, -1, 3)).transpose(1, 2)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        q = apply_cached_rotary_emb(encoding, q)
        k = apply_cached_rotary_emb(encoding, k)
        context = self.inner_attn(q, k, v, mask=mask)
        message = self.out_proj(context.transpose(1, 2).flatten(start_dim=-2))
        return x + self.ffn(torch.cat([x, message], -1))


class CrossBlock(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, flash: bool = False, bias: bool = True
    ) -> None:
        super().__init__()
        self.heads = num_heads
        dim_head = embed_dim // num_heads
        self.scale = dim_head**-0.5
        inner_dim = dim_head * num_heads
        self.to_qk = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_out = nn.Linear(inner_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )
        if flash and FLASH_AVAILABLE:
            self.flash = Attention(True)
        else:
            self.flash = None

    def map_(self, func: Callable, x0: torch.Tensor, x1: torch.Tensor):
        return func(x0), func(x1)

    def forward(
        self, x0: torch.Tensor, x1: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        qk0, qk1 = self.map_(self.to_qk, x0, x1)
        v0, v1 = self.map_(self.to_v, x0, x1)
        qk0, qk1, v0, v1 = map(
            lambda t: t.unflatten(-1, (self.heads, -1)).transpose(1, 2),
            (qk0, qk1, v0, v1),
        )
        if self.flash is not None and qk0.device.type == "cuda":
            m0 = self.flash(qk0, qk1, v1, mask)
            m1 = self.flash(
                qk1, qk0, v0, mask.transpose(-1, -2) if mask is not None else None
            )
        else:
            qk0, qk1 = qk0 * self.scale**0.5, qk1 * self.scale**0.5
            sim = torch.einsum("bhid, bhjd -> bhij", qk0, qk1)
            if mask is not None:
                sim = sim.masked_fill(~mask, -float("inf"))
            attn01 = F.softmax(sim, dim=-1)
            attn10 = F.softmax(sim.transpose(-2, -1).contiguous(), dim=-1)
            m0 = torch.einsum("bhij, bhjd -> bhid", attn01, v1)
            m1 = torch.einsum("bhji, bhjd -> bhid", attn10.transpose(-2, -1), v0)
            if mask is not None:
                m0, m1 = m0.nan_to_num(), m1.nan_to_num()
        m0, m1 = self.map_(lambda t: t.transpose(1, 2).flatten(start_dim=-2), m0, m1)
        m0, m1 = self.map_(self.to_out, m0, m1)
        x0 = x0 + self.ffn(torch.cat([x0, m0], -1))
        x1 = x1 + self.ffn(torch.cat([x1, m1], -1))
        return x0, x1


class TransformerLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.self_attn = SelfBlock(*args, **kwargs)
        self.cross_attn = CrossBlock(*args, **kwargs)

    def forward(
        self,
        desc0,
        desc1,
        encoding0,
        encoding1,
        mask0: Optional[torch.Tensor] = None,
        mask1: Optional[torch.Tensor] = None,
    ):
        if mask0 is not None and mask1 is not None:
            return self.masked_forward(desc0, desc1, encoding0, encoding1, mask0, mask1)
        else:
            desc0 = self.self_attn(desc0, encoding0)
            desc1 = self.self_attn(desc1, encoding1)
            return self.cross_attn(desc0, desc1)

    # This part is compiled and allows padding inputs
    def masked_forward(self, desc0, desc1, encoding0, encoding1, mask0, mask1):
        mask = mask0 & mask1.transpose(-1, -2)
        mask0 = mask0 & mask0.transpose(-1, -2)
        mask1 = mask1 & mask1.transpose(-1, -2)
        desc0 = self.self_attn(desc0, encoding0, mask0)
        desc1 = self.self_attn(desc1, encoding1, mask1)
        return self.cross_attn(desc0, desc1, mask)


# ==================== TensorRT-Compatible Classes ====================

def unflatten_trt(x: torch.Tensor, dim: int, sizes: Tuple[int, ...]) -> torch.Tensor:
    """TensorRT-compatible unflatten using reshape.

    Args:
        x: Input tensor
        dim: Dimension to unflatten
        sizes: Tuple of sizes for the new dimensions. -1 means infer.

    Note: This function uses x.shape which will be traced with concrete values
    during torch2trt conversion. For truly dynamic shapes, consider using ONNX
    export with dynamic axes.
    """
    shape = list(x.shape)
    ndim = len(shape)

    # Normalize negative dimension
    if dim < 0:
        dim = ndim + dim

    # Compute -1 in sizes if present
    sizes = list(sizes)
    if -1 in sizes:
        idx = sizes.index(-1)
        known_product = 1
        for s in sizes:
            if s != -1:
                known_product *= s
        sizes[idx] = shape[dim] // known_product

    # Build new shape
    new_shape = shape[:dim] + sizes + shape[dim + 1:]
    return x.reshape(*new_shape)


class AttentionTRT(nn.Module):
    """TensorRT-compatible attention using matmul instead of einsum."""
    def __init__(self) -> None:
        super().__init__()

    def forward(self, q, k, v, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # q, k, v: [B, H, N, D]
        s = q.shape[-1] ** -0.5
        # Use matmul: [B, H, N, D] @ [B, H, D, M] -> [B, H, N, M]
        sim = torch.matmul(q, k.transpose(-2, -1)) * s
        if mask is not None:
            sim = sim.masked_fill(~mask, -float("inf"))
        attn = F.softmax(sim, -1)
        # [B, H, N, M] @ [B, H, M, D] -> [B, H, N, D]
        return torch.matmul(attn, v)


class SelfBlockTRT(nn.Module):
    """TensorRT-compatible SelfBlock."""
    def __init__(
        self, embed_dim: int, num_heads: int, flash: bool = False, bias: bool = True
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0
        self.head_dim = self.embed_dim // num_heads
        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.inner_attn = AttentionTRT()
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        encoding_cos: torch.Tensor,
        encoding_sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qkv = self.Wqkv(x)
        # Match original memory layout: unflatten to [B, N, H, head_dim, 3], transpose, extract
        # Original: qkv.unflatten(-1, (num_heads, -1, 3)).transpose(1, 2) -> [B, H, N, head_dim, 3]
        # The weights are interleaved as [q0, k0, v0, q1, k1, v1, ...] for each head position
        qkv = unflatten_trt(qkv, -1, (self.num_heads, self.head_dim, 3))  # [B, N, H, head_dim, 3]
        qkv = qkv.transpose(1, 2)  # [B, H, N, head_dim, 3]
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]  # Each [B, H, N, head_dim]
        q = apply_cached_rotary_emb_trt(encoding_cos, encoding_sin, q)
        k = apply_cached_rotary_emb_trt(encoding_cos, encoding_sin, k)
        context = self.inner_attn(q, k, v, mask=mask)
        message = self.out_proj(context.transpose(1, 2).flatten(start_dim=-2))
        return x + self.ffn(torch.cat([x, message], -1))


class CrossBlockTRT(nn.Module):
    """TensorRT-compatible CrossBlock using matmul instead of einsum."""
    def __init__(
        self, embed_dim: int, num_heads: int, flash: bool = False, bias: bool = True
    ) -> None:
        super().__init__()
        self.heads = num_heads
        dim_head = embed_dim // num_heads
        self.scale = dim_head**-0.5
        inner_dim = dim_head * num_heads
        self.to_qk = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_out = nn.Linear(inner_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def _unflatten_trt(self, t: torch.Tensor) -> torch.Tensor:
        """Reshape [B, N, D] -> [B, H, N, D//H]"""
        # Use unflatten_trt instead of extracting shape values for TensorRT compatibility
        return unflatten_trt(t, -1, (self.heads, -1)).transpose(1, 2)

    def forward(
        self, x0: torch.Tensor, x1: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        qk0, qk1 = self.to_qk(x0), self.to_qk(x1)
        v0, v1 = self.to_v(x0), self.to_v(x1)
        qk0, qk1, v0, v1 = [self._unflatten_trt(t) for t in (qk0, qk1, v0, v1)]

        qk0, qk1 = qk0 * self.scale**0.5, qk1 * self.scale**0.5
        # qk0: [B, H, M, D], qk1: [B, H, N, D]
        # sim = qk0 @ qk1.T: [B, H, M, D] @ [B, H, D, N] -> [B, H, M, N]
        sim = torch.matmul(qk0, qk1.transpose(-2, -1))
        if mask is not None:
            sim = sim.masked_fill(~mask, -float("inf"))
        attn01 = F.softmax(sim, dim=-1)  # [B, H, M, N]
        attn10 = F.softmax(sim.transpose(-2, -1).contiguous(), dim=-1)  # [B, H, N, M]
        # m0: attn01 @ v1 = [B, H, M, N] @ [B, H, N, D] -> [B, H, M, D]
        m0 = torch.matmul(attn01, v1)
        # m1: attn10 @ v0 = [B, H, N, M] @ [B, H, M, D] -> [B, H, N, D]
        m1 = torch.matmul(attn10, v0)
        if mask is not None:
            m0, m1 = m0.nan_to_num(), m1.nan_to_num()

        # Flatten back: [B, H, N, D] -> [B, N, H*D]
        m0 = m0.transpose(1, 2).flatten(start_dim=-2)
        m1 = m1.transpose(1, 2).flatten(start_dim=-2)
        m0, m1 = self.to_out(m0), self.to_out(m1)
        x0 = x0 + self.ffn(torch.cat([x0, m0], -1))
        x1 = x1 + self.ffn(torch.cat([x1, m1], -1))
        return x0, x1


class TransformerLayerTRT(nn.Module):
    """TensorRT-compatible TransformerLayer."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.self_attn = SelfBlockTRT(*args, **kwargs)
        self.cross_attn = CrossBlockTRT(*args, **kwargs)

    def forward(
        self,
        desc0,
        desc1,
        encoding0_cos,
        encoding0_sin,
        encoding1_cos,
        encoding1_sin,
        mask0: Optional[torch.Tensor] = None,
        mask1: Optional[torch.Tensor] = None,
    ):
        # No masked forward for TRT - just use the simple path
        desc0 = self.self_attn(desc0, encoding0_cos, encoding0_sin)
        desc1 = self.self_attn(desc1, encoding1_cos, encoding1_sin)
        return self.cross_attn(desc0, desc1)


def logsigmoid_trt(x: torch.Tensor) -> torch.Tensor:
    """TensorRT-compatible logsigmoid using softplus."""
    # logsigmoid(x) = -softplus(-x)
    return -F.softplus(-x)


def log_softmax_trt(x: torch.Tensor, dim: int) -> torch.Tensor:
    """TensorRT-compatible log_softmax.

    F.log_softmax doesn't work correctly with torch2trt, so we implement it
    using operations that are supported: max, exp, sum, log.
    """
    x_max = x.max(dim=dim, keepdim=True).values
    x_shifted = x - x_max
    log_sum_exp = torch.log(torch.exp(x_shifted).sum(dim=dim, keepdim=True))
    return x_shifted - log_sum_exp


def sigmoid_log_double_softmax_trt(
    sim: torch.Tensor, z0: torch.Tensor, z1: torch.Tensor
) -> torch.Tensor:
    """TensorRT-compatible version - create the log assignment matrix from logits and similarity."""
    certainties = logsigmoid_trt(z0) + logsigmoid_trt(z1).transpose(1, 2)
    scores0 = log_softmax_trt(sim, 2)
    scores1 = log_softmax_trt(sim.transpose(-1, -2).contiguous(), 2).transpose(-1, -2)

    # Build scores matrix without new_full
    # Main scores block
    main_scores = scores0 + scores1 + certainties  # [B, M, N]

    # Unmatched scores for dim 0: logsigmoid(-z0) -> [B, M, 1]
    unmatch0 = logsigmoid_trt(-z0)  # [B, M, 1]
    # Unmatched scores for dim 1: logsigmoid(-z1) -> [B, N, 1] -> [B, 1, N]
    unmatch1 = logsigmoid_trt(-z1).transpose(1, 2)  # [B, 1, N]

    # Build the full [B, M+1, N+1] matrix
    # Row for unmatched in dim1: [unmatch1, 0] -> [B, 1, N+1]
    corner = main_scores[:, :1, :1] * 0  # [B, 1, 1] of zeros
    unmatch1_row = torch.cat([unmatch1, corner], dim=2)  # [B, 1, N+1]

    # Main block with unmatch0 column: [main_scores, unmatch0] -> [B, M, N+1]
    main_with_unmatch = torch.cat([main_scores, unmatch0], dim=2)  # [B, M, N+1]

    # Full matrix: stack main_with_unmatch and unmatch1_row
    scores = torch.cat([main_with_unmatch, unmatch1_row], dim=1)  # [B, M+1, N+1]

    return scores


class MatchAssignmentTRT(nn.Module):
    """TensorRT-compatible MatchAssignment using matmul instead of einsum."""
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.matchability = nn.Linear(dim, 1, bias=True)
        self.final_proj = nn.Linear(dim, dim, bias=True)

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        """build assignment matrix from descriptors"""
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        # Use self.dim (fixed at construction) instead of extracting from shape
        mdesc0, mdesc1 = mdesc0 / self.dim**0.25, mdesc1 / self.dim**0.25
        # Use matmul: [B, M, D] @ [B, D, N] -> [B, M, N]
        sim = torch.matmul(mdesc0, mdesc1.transpose(-2, -1))
        z0 = self.matchability(desc0)
        z1 = self.matchability(desc1)
        scores = sigmoid_log_double_softmax_trt(sim, z0, z1)
        return scores, sim


def sigmoid_log_double_softmax(
    sim: torch.Tensor, z0: torch.Tensor, z1: torch.Tensor
) -> torch.Tensor:
    """create the log assignment matrix from logits and similarity"""
    b, m, n = sim.shape
    certainties = F.logsigmoid(z0) + F.logsigmoid(z1).transpose(1, 2)
    scores0 = F.log_softmax(sim, 2)
    scores1 = F.log_softmax(sim.transpose(-1, -2).contiguous(), 2).transpose(-1, -2)
    scores = sim.new_full((b, m + 1, n + 1), 0)
    scores[:, :m, :n] = scores0 + scores1 + certainties
    scores[:, :-1, -1] = F.logsigmoid(-z0.squeeze(-1))
    scores[:, -1, :-1] = F.logsigmoid(-z1.squeeze(-1))
    return scores


class MatchAssignment(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.matchability = nn.Linear(dim, 1, bias=True)
        self.final_proj = nn.Linear(dim, dim, bias=True)

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        """build assignment matrix from descriptors"""
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        _, _, d = mdesc0.shape
        mdesc0, mdesc1 = mdesc0 / d**0.25, mdesc1 / d**0.25
        sim = torch.einsum("bmd,bnd->bmn", mdesc0, mdesc1)
        z0 = self.matchability(desc0)
        z1 = self.matchability(desc1)
        scores = sigmoid_log_double_softmax(sim, z0, z1)
        return scores, sim

    def get_matchability(self, desc: torch.Tensor):
        return torch.sigmoid(self.matchability(desc)).squeeze(-1)


def filter_matches(scores: torch.Tensor, th: float):
    """obtain matches from a log assignment matrix [Bx M+1 x N+1]"""
    max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    m0, m1 = max0.indices, max1.indices
    indices0 = torch.arange(m0.shape[1], device=m0.device)[None]
    indices1 = torch.arange(m1.shape[1], device=m1.device)[None]
    mutual0 = indices0 == m1.gather(1, m0)
    mutual1 = indices1 == m0.gather(1, m1)
    max0_exp = max0.values.exp()
    zero = max0_exp.new_tensor(0)
    mscores0 = torch.where(mutual0, max0_exp, zero)
    mscores1 = torch.where(mutual1, mscores0.gather(1, m1), zero)
    valid0 = mutual0 & (mscores0 > th)
    valid1 = mutual1 & valid0.gather(1, m1)
    m0 = torch.where(valid0, m0, -1)
    m1 = torch.where(valid1, m1, -1)
    return m0, m1, mscores0, mscores1


class LightGlue(nn.Module):
    default_conf = {
        "name": "lightglue",  # just for interfacing
        "input_dim": 256,  # input descriptor dimension (autoselected from weights)
        "descriptor_dim": 256,
        "add_scale_ori": False,
        "n_layers": 9,
        "num_heads": 4,
        "flash": True,  # enable FlashAttention if available.
        "mp": False,  # enable mixed precision
        "depth_confidence": 0.95,  # early stopping, disable with -1
        "width_confidence": 0.99,  # point pruning, disable with -1
        "filter_threshold": 0.1,  # match threshold
        "weights": None,
    }

    # Point pruning involves an overhead (gather).
    # Therefore, we only activate it if there are enough keypoints.
    pruning_keypoint_thresholds = {
        "cpu": -1,
        "mps": -1,
        "cuda": 1024,
        "flash": 1536,
    }

    required_data_keys = ["image0", "image1"]

    version = "v0.1_arxiv"
    url = "https://github.com/cvg/LightGlue/releases/download/{}/{}_lightglue.pth"

    features = {
        "superpoint": {
            "weights": "superpoint_lightglue",
            "input_dim": 256,
        },
        "disk": {
            "weights": "disk_lightglue",
            "input_dim": 128,
        },
        "aliked": {
            "weights": "aliked_lightglue",
            "input_dim": 128,
        },
        "sift": {
            "weights": "sift_lightglue",
            "input_dim": 128,
            "add_scale_ori": True,
        },
        "doghardnet": {
            "weights": "doghardnet_lightglue",
            "input_dim": 128,
            "add_scale_ori": True,
        },
    }

    def __init__(self, features="superpoint", **conf) -> None:
        super().__init__()
        self.conf = conf = SimpleNamespace(**{**self.default_conf, **conf})
        if features is not None:
            if features not in self.features:
                raise ValueError(
                    f"Unsupported features: {features} not in "
                    f"{{{','.join(self.features)}}}"
                )
            for k, v in self.features[features].items():
                setattr(conf, k, v)

        if conf.input_dim != conf.descriptor_dim:
            self.input_proj = nn.Linear(conf.input_dim, conf.descriptor_dim, bias=True)
        else:
            self.input_proj = nn.Identity()

        head_dim = conf.descriptor_dim // conf.num_heads
        self.posenc = LearnableFourierPositionalEncoding(
            2 + 2 * self.conf.add_scale_ori, head_dim, head_dim
        )

        h, n, d = conf.num_heads, conf.n_layers, conf.descriptor_dim

        self.transformers = nn.ModuleList(
            [TransformerLayer(d, h, conf.flash) for _ in range(n)]
        )

        self.log_assignment = nn.ModuleList([MatchAssignment(d) for _ in range(n)])
        self.token_confidence = nn.ModuleList(
            [TokenConfidence(d) for _ in range(n - 1)]
        )
        self.register_buffer(
            "confidence_thresholds",
            torch.Tensor(
                [self.confidence_threshold(i) for i in range(self.conf.n_layers)]
            ),
        )

        state_dict = None
        if features is not None:
            fname = f"{conf.weights}_{self.version.replace('.', '-')}.pth"
            state_dict = torch.hub.load_state_dict_from_url(
                self.url.format(self.version, features), file_name=fname
            )
            self.load_state_dict(state_dict, strict=False)
        elif conf.weights is not None:
            path = Path(__file__).parent
            path = path / "weights/{}.pth".format(self.conf.weights)
            state_dict = torch.load(str(path), map_location="cpu")

        if state_dict:
            # rename old state dict entries
            for i in range(self.conf.n_layers):
                pattern = f"self_attn.{i}", f"transformers.{i}.self_attn"
                state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
                pattern = f"cross_attn.{i}", f"transformers.{i}.cross_attn"
                state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
            self.load_state_dict(state_dict, strict=False)

        # static lengths LightGlue is compiled for (only used with torch.compile)
        self.static_lengths = None

    def compile(
        self, mode="reduce-overhead", static_lengths=[256, 512, 768, 1024, 1280, 1536]
    ):
        if self.conf.width_confidence != -1:
            warnings.warn(
                "Point pruning is partially disabled for compiled forward.",
                stacklevel=2,
            )

        torch._inductor.cudagraph_mark_step_begin()
        for i in range(self.conf.n_layers):
            self.transformers[i].masked_forward = torch.compile(
                self.transformers[i].masked_forward, mode=mode, fullgraph=True
            )

        self.static_lengths = static_lengths

    def forward(self, data: dict) -> dict:
        """
        Match keypoints and descriptors between two images

        Input (dict):
            image0: dict
                keypoints: [B x M x 2]
                descriptors: [B x M x D]
                image: [B x C x H x W] or image_size: [B x 2]
            image1: dict
                keypoints: [B x N x 2]
                descriptors: [B x N x D]
                image: [B x C x H x W] or image_size: [B x 2]
        Output (dict):
            matches0: [B x M]
            matching_scores0: [B x M]
            matches1: [B x N]
            matching_scores1: [B x N]
            matches: List[[Si x 2]]
            scores: List[[Si]]
            stop: int
            prune0: [B x M]
            prune1: [B x N]
        """
        with torch.autocast(enabled=self.conf.mp, device_type="cuda"):
            return self._forward(data)

    def _forward(self, data: dict) -> dict:
        for key in self.required_data_keys:
            assert key in data, f"Missing key {key} in data"
        data0, data1 = data["image0"], data["image1"]
        kpts0, kpts1 = data0["keypoints"], data1["keypoints"]
        b, m, _ = kpts0.shape
        b, n, _ = kpts1.shape
        device = kpts0.device
        size0, size1 = data0.get("image_size"), data1.get("image_size")
        kpts0 = normalize_keypoints(kpts0, size0).clone()
        kpts1 = normalize_keypoints(kpts1, size1).clone()

        if self.conf.add_scale_ori:
            kpts0 = torch.cat(
                [kpts0] + [data0[k].unsqueeze(-1) for k in ("scales", "oris")], -1
            )
            kpts1 = torch.cat(
                [kpts1] + [data1[k].unsqueeze(-1) for k in ("scales", "oris")], -1
            )
        desc0 = data0["descriptors"].detach().contiguous()
        desc1 = data1["descriptors"].detach().contiguous()

        assert desc0.shape[-1] == self.conf.input_dim
        assert desc1.shape[-1] == self.conf.input_dim

        if torch.is_autocast_enabled():
            desc0 = desc0.half()
            desc1 = desc1.half()

        mask0, mask1 = None, None
        c = max(m, n)
        do_compile = self.static_lengths and c <= max(self.static_lengths)
        if do_compile:
            kn = min([k for k in self.static_lengths if k >= c])
            desc0, mask0 = pad_to_length(desc0, kn)
            desc1, mask1 = pad_to_length(desc1, kn)
            kpts0, _ = pad_to_length(kpts0, kn)
            kpts1, _ = pad_to_length(kpts1, kn)
        desc0 = self.input_proj(desc0)
        desc1 = self.input_proj(desc1)
        # cache positional embeddings
        encoding0 = self.posenc(kpts0)
        encoding1 = self.posenc(kpts1)

        # GNN + final_proj + assignment
        do_early_stop = self.conf.depth_confidence > 0
        do_point_pruning = self.conf.width_confidence > 0 and not do_compile
        pruning_th = self.pruning_min_kpts(device)
        if do_point_pruning:
            ind0 = torch.arange(0, m, device=device)[None]
            ind1 = torch.arange(0, n, device=device)[None]
            # We store the index of the layer at which pruning is detected.
            prune0 = torch.ones_like(ind0)
            prune1 = torch.ones_like(ind1)
        token0, token1 = None, None
        for i in range(self.conf.n_layers):
            if desc0.shape[1] == 0 or desc1.shape[1] == 0:  # no keypoints
                break
            desc0, desc1 = self.transformers[i](
                desc0, desc1, encoding0, encoding1, mask0=mask0, mask1=mask1
            )
            if i == self.conf.n_layers - 1:
                continue  # no early stopping or adaptive width at last layer

            if do_early_stop:
                token0, token1 = self.token_confidence[i](desc0, desc1)
                if self.check_if_stop(token0[..., :m], token1[..., :n], i, m + n):
                    break
            if do_point_pruning and desc0.shape[-2] > pruning_th:
                scores0 = self.log_assignment[i].get_matchability(desc0)
                prunemask0 = self.get_pruning_mask(token0, scores0, i)
                keep0 = torch.where(prunemask0)[1]
                ind0 = ind0.index_select(1, keep0)
                desc0 = desc0.index_select(1, keep0)
                encoding0 = encoding0.index_select(-2, keep0)
                prune0[:, ind0] += 1
            if do_point_pruning and desc1.shape[-2] > pruning_th:
                scores1 = self.log_assignment[i].get_matchability(desc1)
                prunemask1 = self.get_pruning_mask(token1, scores1, i)
                keep1 = torch.where(prunemask1)[1]
                ind1 = ind1.index_select(1, keep1)
                desc1 = desc1.index_select(1, keep1)
                encoding1 = encoding1.index_select(-2, keep1)
                prune1[:, ind1] += 1

        if desc0.shape[1] == 0 or desc1.shape[1] == 0:  # no keypoints
            m0 = desc0.new_full((b, m), -1, dtype=torch.long)
            m1 = desc1.new_full((b, n), -1, dtype=torch.long)
            mscores0 = desc0.new_zeros((b, m))
            mscores1 = desc1.new_zeros((b, n))
            matches = desc0.new_empty((b, 0, 2), dtype=torch.long)
            mscores = desc0.new_empty((b, 0))
            if not do_point_pruning:
                prune0 = torch.ones_like(mscores0) * self.conf.n_layers
                prune1 = torch.ones_like(mscores1) * self.conf.n_layers
            return {
                "matches0": m0,
                "matches1": m1,
                "matching_scores0": mscores0,
                "matching_scores1": mscores1,
                "stop": i + 1,
                "matches": matches,
                "scores": mscores,
                "prune0": prune0,
                "prune1": prune1,
            }

        desc0, desc1 = desc0[..., :m, :], desc1[..., :n, :]  # remove padding
        scores, _ = self.log_assignment[i](desc0, desc1)
        m0, m1, mscores0, mscores1 = filter_matches(scores, self.conf.filter_threshold)
        matches, mscores = [], []
        for k in range(b):
            valid = m0[k] > -1
            m_indices_0 = torch.where(valid)[0]
            m_indices_1 = m0[k][valid]
            if do_point_pruning:
                m_indices_0 = ind0[k, m_indices_0]
                m_indices_1 = ind1[k, m_indices_1]
            matches.append(torch.stack([m_indices_0, m_indices_1], -1))
            mscores.append(mscores0[k][valid])

        # TODO: Remove when hloc switches to the compact format.
        if do_point_pruning:
            m0_ = torch.full((b, m), -1, device=m0.device, dtype=m0.dtype)
            m1_ = torch.full((b, n), -1, device=m1.device, dtype=m1.dtype)
            m0_[:, ind0] = torch.where(m0 == -1, -1, ind1.gather(1, m0.clamp(min=0)))
            m1_[:, ind1] = torch.where(m1 == -1, -1, ind0.gather(1, m1.clamp(min=0)))
            mscores0_ = torch.zeros((b, m), device=mscores0.device)
            mscores1_ = torch.zeros((b, n), device=mscores1.device)
            mscores0_[:, ind0] = mscores0
            mscores1_[:, ind1] = mscores1
            m0, m1, mscores0, mscores1 = m0_, m1_, mscores0_, mscores1_
        else:
            prune0 = torch.ones_like(mscores0) * self.conf.n_layers
            prune1 = torch.ones_like(mscores1) * self.conf.n_layers

        return {
            "matches0": m0,
            "matches1": m1,
            "matching_scores0": mscores0,
            "matching_scores1": mscores1,
            "stop": i + 1,
            "matches": matches,
            "scores": mscores,
            "prune0": prune0,
            "prune1": prune1,
        }

    def confidence_threshold(self, layer_index: int) -> float:
        """scaled confidence threshold"""
        threshold = 0.8 + 0.1 * np.exp(-4.0 * layer_index / self.conf.n_layers)
        return np.clip(threshold, 0, 1)

    def get_pruning_mask(
        self, confidences: torch.Tensor, scores: torch.Tensor, layer_index: int
    ) -> torch.Tensor:
        """mask points which should be removed"""
        keep = scores > (1 - self.conf.width_confidence)
        if confidences is not None:  # Low-confidence points are never pruned.
            keep |= confidences <= self.confidence_thresholds[layer_index]
        return keep

    def check_if_stop(
        self,
        confidences0: torch.Tensor,
        confidences1: torch.Tensor,
        layer_index: int,
        num_points: int,
    ) -> torch.Tensor:
        """evaluate stopping condition"""
        confidences = torch.cat([confidences0, confidences1], -1)
        threshold = self.confidence_thresholds[layer_index]
        ratio_confident = 1.0 - (confidences < threshold).float().sum() / num_points
        return ratio_confident > self.conf.depth_confidence

    def pruning_min_kpts(self, device: torch.device):
        if self.conf.flash and FLASH_AVAILABLE and device.type == "cuda":
            return self.pruning_keypoint_thresholds["flash"]
        else:
            return self.pruning_keypoint_thresholds[device.type]

