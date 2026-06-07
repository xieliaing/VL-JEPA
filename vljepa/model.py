"""
VL-JEPA — from-scratch, paper-faithful implementation.

Paper: "VL-JEPA: Joint Embedding Predictive Architecture for Vision-language"
       Chen et al., Meta FAIR, arXiv:2512.10942v2 (Feb 2026), Sec. 3.1.

Architecture (paper §3.1), final model = 1.6B params:
  - X-Encoder : FROZEN V-JEPA 2 ViT-L (304M). Video -> sequence of visual tokens
                at 256^2; images duplicated across frames.
  - Predictor : the LAST 8 transformer layers of Llama-3.2-1B (490M trainable),
                Llama tokenizer + token embeddings for the query, max 512 query
                tokens with [PAD], CAUSAL MASK DISABLED so vision + query attend
                jointly, average pool over non-[PAD] tokens.
  - Y-Encoder : EmbeddingGemma-300M, trained with LR x0.05.
  - Shared    : linear projection heads -> 1536-d space where the loss lives.
  - Loss      : bi-directional InfoNCE, Predictor + Y-Encoder trained jointly.

The Llama-3 predictor blocks are implemented FROM SCRATCH (RMSNorm, RoPE, GQA,
SwiGLU) with a fused SDPA attention path (default). The real gated HuggingFace
weights can be swapped in via backend="hf"; see `paper_config`/`build_paper_model_hf`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Config
# ============================================================================

@dataclass
class VLJEPAConfig:
    # ---- Predictor (Llama-3.2-1B "last 8 layers"); paper defaults in comments
    pred_hidden: int = 128          # paper: 2048
    pred_layers: int = 2            # paper: 8 (the last 8 of Llama-3.2-1B's 16)
    pred_heads: int = 4             # paper: 32
    pred_kv_heads: int = 2          # paper: 8  (GQA)
    pred_intermediate: int = 256    # paper: 8192 (SwiGLU)
    rope_theta: float = 500000.0    # Llama-3.2 default
    rms_eps: float = 1e-5
    attn_impl: str = "sdpa"         # "sdpa" (fused flash, default) or "eager" (manual)
    max_query_tokens: int = 16      # paper: 512
    vocab_size: int = 512           # paper: 128256 (Llama-3.2 tokenizer)

    # ---- X-Encoder (V-JEPA 2 ViT-L)
    vis_dim: int = 64               # paper: 1024 (ViT-L hidden)
    num_visual_tokens: int = 8      # stand-in token count per sample
    freeze_x_encoder: bool = True   # paper: frozen

    # ---- Y-Encoder (EmbeddingGemma-300M)
    y_hidden: int = 96              # paper: 768 (EmbeddingGemma hidden)
    y_layers: int = 2
    y_lr_multiplier: float = 0.05   # paper §3.1 / Tab. 7(b)
    y_pool: str = "mean"            # "mean" (EmbeddingGemma) or "cls" (BGE-M3 / XLM-R)

    # ---- Shared embedding space
    shared_dim: int = 1536          # paper: 1536
    temperature: float = 0.07       # CLIP-style; paper cites Radford et al. 2021

    # ---- Pooling: paper says "average pooling on non-[PAD] tokens".
    # "all" = pool visual + non-pad query tokens (literal reading).
    # "query" = pool only non-pad query tokens.
    pool: str = "all"

    # ---- Backend: "scratch" = from-scratch Llama blocks (runs anywhere);
    #               "hf" = load the real gated HuggingFace weights.
    backend: str = "scratch"
    x_encoder_name: str = "facebook/vjepa2-vitl-fpc64-256"   # V-JEPA 2 ViT-L
    predictor_name: str = "meta-llama/Llama-3.2-1B"
    predictor_last_n_layers: int = 8                          # paper: last 8 of 16
    freeze_query_embedding: bool = False                      # token-embed trainable?
    y_encoder_name: str = "google/embeddinggemma-300m"


def paper_config(backend: str = "scratch") -> VLJEPAConfig:
    """The real paper-scale configuration (for the budget report or HF backend)."""
    return VLJEPAConfig(
        pred_hidden=2048, pred_layers=8, pred_heads=32, pred_kv_heads=8,
        pred_intermediate=8192, max_query_tokens=512, vocab_size=128256,
        vis_dim=1024, num_visual_tokens=256, y_hidden=768, y_layers=24,
        shared_dim=1536, backend=backend,
    )


def build_paper_model_hf() -> "VLJEPA":
    """Construct the real 1.6B VL-JEPA with actual HuggingFace weights.

    Requires HuggingFace access to the gated checkpoints and a token:
        huggingface-cli login
    then accept the licenses for meta-llama/Llama-3.2-1B and
    google/embeddinggemma-300m. Downloads several GB and needs a GPU.
    """
    return VLJEPA(paper_config(backend="hf"))


# ============================================================================
# Llama-3 building blocks (from scratch)
# ============================================================================

class RMSNorm(nn.Module):
    """Llama RMSNorm: x / sqrt(mean(x^2) + eps) * weight (no mean-centering)."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def build_rope_cache(head_dim: int, max_pos: int, theta: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables for rotary position embeddings (Llama/HF form)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_pos).float()
    freqs = torch.outer(t, inv_freq)              # [max_pos, head_dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)        # [max_pos, head_dim]
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q,k: [B, H, S, D]; cos,sin: [S, D]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q_out = q * cos + rotate_half(q) * sin
    k_out = k * cos + rotate_half(k) * sin
    return q_out, k_out


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads to match Q heads for grouped-query attention."""
    b, kvh, s, d = x.shape
    if n_rep == 1:
        return x
    return x[:, :, None, :, :].expand(b, kvh, n_rep, s, d).reshape(b, kvh * n_rep, s, d)


class LlamaAttention(nn.Module):
    """Grouped-query attention with RoPE. Non-causal when is_causal=False.

    attn_impl="sdpa" uses the fused flash/mem-efficient kernel (default, faster
    and lower-memory); "eager" is the explicit softmax(QK^T)V for reference.
    """

    def __init__(self, cfg: VLJEPAConfig) -> None:
        super().__init__()
        self.h = cfg.pred_heads
        self.kvh = cfg.pred_kv_heads
        self.d = cfg.pred_hidden // cfg.pred_heads
        self.n_rep = self.h // self.kvh
        self.q_proj = nn.Linear(cfg.pred_hidden, self.h * self.d, bias=False)
        self.k_proj = nn.Linear(cfg.pred_hidden, self.kvh * self.d, bias=False)
        self.v_proj = nn.Linear(cfg.pred_hidden, self.kvh * self.d, bias=False)
        self.o_proj = nn.Linear(self.h * self.d, cfg.pred_hidden, bias=False)
        self.attn_impl = cfg.attn_impl

    def forward(self, x, cos, sin, key_padding_mask=None, is_causal=False):
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.h, self.d).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.kvh, self.d).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.kvh, self.d).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        if self.attn_impl == "sdpa":
            # Boolean mask: True = allowed to attend. Visual keys always valid;
            # [PAD] query keys masked. Non-causal unless is_causal requested.
            allow = None
            if key_padding_mask is not None:
                allow = key_padding_mask[:, None, None, :].expand(b, 1, s, s)
            if is_causal:
                causal = torch.tril(torch.ones(s, s, device=x.device, dtype=torch.bool))
                allow = causal[None, None] if allow is None else (allow & causal[None, None])
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=allow)
        else:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d)  # [B,H,S,S]
            if is_causal:
                causal = torch.triu(torch.ones(s, s, device=x.device, dtype=torch.bool), 1)
                scores = scores.masked_fill(causal[None, None], float("-inf"))
            if key_padding_mask is not None:
                pad = (~key_padding_mask)[:, None, None, :]  # [B,S], True = valid
                scores = scores.masked_fill(pad, float("-inf"))
            attn = scores.softmax(dim=-1)
            out = attn @ v
        out = out.transpose(1, 2).reshape(b, s, self.h * self.d)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg: VLJEPAConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(cfg.pred_hidden, cfg.pred_intermediate, bias=False)
        self.up = nn.Linear(cfg.pred_hidden, cfg.pred_intermediate, bias=False)
        self.down = nn.Linear(cfg.pred_intermediate, cfg.pred_hidden, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class LlamaBlock(nn.Module):
    """Pre-norm Llama-3 transformer block."""

    def __init__(self, cfg: VLJEPAConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.pred_hidden, cfg.rms_eps)
        self.attn = LlamaAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.pred_hidden, cfg.rms_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, key_padding_mask=None, is_causal=False):
        x = x + self.attn(self.attn_norm(x), cos, sin, key_padding_mask, is_causal)
        x = x + self.mlp(self.mlp_norm(x))
        return x


# ============================================================================
# X-Encoder (frozen V-JEPA 2 stand-in) and Y-Encoder (EmbeddingGemma stand-in)
# ============================================================================

class StandInXEncoder(nn.Module):
    """Frozen visual encoder producing [B, num_visual_tokens, vis_dim].

    Stands in for V-JEPA 2 ViT-L so the model runs with no gated downloads.
    With backend="hf" this is replaced by transformers VJEPA2Model.
    """

    def __init__(self, cfg: VLJEPAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch = nn.Conv2d(3, cfg.vis_dim, kernel_size=16, stride=16)
        self.blocks = nn.Sequential(
            nn.LayerNorm(cfg.vis_dim), nn.Linear(cfg.vis_dim, cfg.vis_dim), nn.GELU()
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: [B, T, C, H, W]; treat each frame, then take first N tokens.
        b, t, c, h, w = frames.shape
        x = frames.view(b * t, c, h, w)
        x = self.patch(x).flatten(2).transpose(1, 2)  # [B*T, patches, vis_dim]
        x = self.blocks(x)
        x = x.view(b, t * x.shape[1], -1)              # flatten time into tokens
        n = self.cfg.num_visual_tokens
        if x.shape[1] >= n:
            x = x[:, :n, :]
        else:
            x = F.pad(x, (0, 0, 0, n - x.shape[1]))
        return x                                        # [B, N, vis_dim]


class StandInYEncoder(nn.Module):
    """Text target encoder producing one embedding per sequence (mean-pooled).

    Stands in for EmbeddingGemma-300M. With backend="hf" this is replaced by
    google/embeddinggemma-300m loaded via AutoModel; trained at LR x0.05.
    """

    def __init__(self, cfg: VLJEPAConfig) -> None:
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.y_hidden)
        enc = nn.TransformerEncoderLayer(
            cfg.y_hidden, nhead=4, dim_feedforward=cfg.y_hidden * 4,
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, cfg.y_layers)
        self.hidden = cfg.y_hidden

    def forward(self, input_ids, attention_mask):
        x = self.embed(input_ids)
        x = self.encoder(x, src_key_padding_mask=(attention_mask == 0))
        m = attention_mask.unsqueeze(-1).float()
        return (x * m).sum(1) / m.sum(1).clamp_min(1)   # mean pool -> [B, y_hidden]


# ============================================================================
# HuggingFace backend — real V-JEPA 2, Llama-3.2-1B layers, EmbeddingGemma
# ============================================================================
# Constructed only when cfg.backend == "hf"; the from-scratch path above runs
# with no downloads. API tested against transformers >= 4.52 (ships VJEPA2Model).

class HFXEncoder(nn.Module):
    """Frozen V-JEPA 2 ViT-L via HuggingFace -> visual tokens [B, N, hidden]."""

    def __init__(self, cfg: VLJEPAConfig) -> None:
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install transformers to use backend='hf'.") from e
        self.model = AutoModel.from_pretrained(cfg.x_encoder_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.out_dim = self.model.config.hidden_size  # ViT-L: 1024

    @torch.no_grad()
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: [B, T, C, H, W]. The exact input kwarg has varied; try both.
        for kw in ("pixel_values_videos", "pixel_values"):
            try:
                out = self.model(**{kw: frames})
                break
            except TypeError:
                continue
        else:  # pragma: no cover
            raise RuntimeError("Could not call V-JEPA2 model; check input kwarg name.")
        return out.last_hidden_state  # [B, num_tokens, hidden]


class HFYEncoder(nn.Module):
    """Text Y-Encoder via HuggingFace AutoModel -> pooled sentence embedding.

    Pooling follows the encoder's convention: mean over non-pad tokens for
    EmbeddingGemma; the [CLS] token for BGE-M3 / XLM-R-style encoders.
    """

    def __init__(self, cfg: VLJEPAConfig) -> None:
        super().__init__()
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(cfg.y_encoder_name)
        self.hidden = self.model.config.hidden_size
        self.pool = getattr(cfg, "y_pool", "mean")

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state
        if self.pool == "cls":
            return h[:, 0]  # BGE-M3 / XLM-R dense embedding = [CLS] token
        m = attention_mask.unsqueeze(-1).float()
        return (h * m).sum(1) / m.sum(1).clamp_min(1)


class HFLlamaPredictor(nn.Module):
    """The last N transformer layers of Llama-3.2-1B, run bidirectionally.

    Reuses the real LlamaDecoderLayer weights, token embedding, rotary
    embedding, and final norm. The causal mask is replaced by a padding-only
    (bidirectional) mask so visual and query tokens attend jointly (paper §3.1).
    """

    def __init__(self, cfg: VLJEPAConfig) -> None:
        super().__init__()
        from transformers import AutoModel
        base = AutoModel.from_pretrained(cfg.predictor_name)  # LlamaModel
        n = cfg.predictor_last_n_layers
        self.embed_tokens = base.embed_tokens                 # Llama token embedding
        self.layers = base.layers[-n:]                        # last N decoder layers
        self.norm = base.norm                                 # final RMSNorm
        self.rotary_emb = base.rotary_emb                     # Llama-3 RoPE (with scaling)
        self.hidden = base.config.hidden_size                 # 2048
        if cfg.freeze_query_embedding:
            for p in self.embed_tokens.parameters():
                p.requires_grad = False

    def embed_query(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(self, inputs_embeds: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        b, s, _ = inputs_embeds.shape
        device = inputs_embeds.device
        position_ids = torch.arange(s, device=device).unsqueeze(0).expand(b, -1)
        # Llama RoPE cos/sin (handles llama3 rope-scaling internally).
        cos, sin = self.rotary_emb(inputs_embeds, position_ids)
        # Non-causal 4D additive mask: 0 for valid keys, -inf for [PAD] keys.
        min_val = torch.finfo(inputs_embeds.dtype).min
        attn_mask = torch.zeros(b, 1, s, s, dtype=inputs_embeds.dtype, device=device)
        attn_mask = attn_mask.masked_fill(~key_padding_mask[:, None, None, :], min_val)
        h = inputs_embeds
        for layer in self.layers:
            h = layer(
                h,
                attention_mask=attn_mask,
                position_ids=position_ids,
                position_embeddings=(cos, sin),  # adapt if your transformers differs
            )[0]
        return self.norm(h)


# ============================================================================
# VL-JEPA model
# ============================================================================

class VLJEPA(nn.Module):
    def __init__(self, cfg: VLJEPAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backend = cfg.backend

        # X-Encoder (frozen) + projection into predictor space.
        self.x_encoder = HFXEncoder(cfg) if cfg.backend == "hf" else StandInXEncoder(cfg)
        if cfg.freeze_x_encoder:
            for p in self.x_encoder.parameters():
                p.requires_grad = False
        vis_dim = getattr(self.x_encoder, "out_dim", cfg.vis_dim)
        self.vis_in_proj = nn.Linear(vis_dim, cfg.pred_hidden)

        # Predictor: real Llama layers (hf) or from-scratch Llama blocks.
        if cfg.backend == "hf":
            self.predictor = HFLlamaPredictor(cfg)
        else:
            self.query_embed = nn.Embedding(cfg.vocab_size, cfg.pred_hidden)
            self.blocks = nn.ModuleList([LlamaBlock(cfg) for _ in range(cfg.pred_layers)])
            self.pred_norm = RMSNorm(cfg.pred_hidden, cfg.rms_eps)
            cos, sin = build_rope_cache(
                cfg.pred_hidden // cfg.pred_heads,
                cfg.num_visual_tokens + cfg.max_query_tokens,
                cfg.rope_theta,
            )
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        self.pred_proj = nn.Linear(cfg.pred_hidden, cfg.shared_dim)

        # Y-Encoder + projection into the shared space.
        self.y_encoder = HFYEncoder(cfg) if cfg.backend == "hf" else StandInYEncoder(cfg)
        y_hidden = getattr(self.y_encoder, "hidden", cfg.y_hidden)
        self.y_proj = nn.Linear(y_hidden, cfg.shared_dim)

    # ---- Frozen visual features (cacheable) -------------------------------
    def visual_features(self, frames: torch.Tensor) -> torch.Tensor:
        """X-Encoder output S_V — deterministic when frozen, so it can be cached."""
        return self.x_encoder(frames)                      # [B, Nv, vis_dim]

    # ---- Predictor branch: (S_V, X_Q) -> Ŝ_Y ------------------------------
    def predict(self, frames, q_ids, q_mask, vis_feats=None) -> torch.Tensor:
        # vis_feats lets callers pass precomputed frozen features to skip the
        # X-Encoder (the feature-cache fast path).
        vis = self.visual_features(frames) if vis_feats is None else vis_feats
        vis = self.vis_in_proj(vis)                        # [B, Nv, H]
        b, nv, _ = vis.shape

        # Query token embeddings (Llama embedding table in both backends).
        if self.backend == "hf":
            q = self.predictor.embed_query(q_ids)          # [B, Nq, H]
        else:
            q = self.query_embed(q_ids)
        x = torch.cat([vis, q], dim=1)                     # [B, Nv+Nq, H]

        vis_mask = torch.ones(b, nv, device=x.device, dtype=q_mask.dtype)
        key_mask = torch.cat([vis_mask, q_mask], dim=1).bool()  # True = valid

        if self.backend == "hf":
            h = self.predictor(x, key_mask)                # bidirectional Llama layers
        else:
            cos = self.rope_cos[: x.shape[1]].to(x.dtype)
            sin = self.rope_sin[: x.shape[1]].to(x.dtype)
            for blk in self.blocks:
                x = blk(x, cos, sin, key_padding_mask=key_mask, is_causal=False)
            h = self.pred_norm(x)

        # Average pool over non-[PAD] tokens (paper §3.1).
        if self.cfg.pool == "query":
            pm = q_mask.unsqueeze(-1).float()
            pooled = (h[:, nv:, :] * pm).sum(1) / pm.sum(1).clamp_min(1)
        else:  # "all": visual + non-pad query tokens
            km = key_mask.unsqueeze(-1).float()
            pooled = (h * km).sum(1) / km.sum(1).clamp_min(1)
        return self.pred_proj(pooled)                      # [B, shared_dim]

    # ---- Y-Encoder branch: Y -> S_Y ---------------------------------------
    def encode_target(self, t_ids, t_mask) -> torch.Tensor:
        return self.y_proj(self.y_encoder(t_ids, t_mask))  # [B, shared_dim]

    def forward(self, frames, q_ids, q_mask, t_ids, t_mask, vis_feats=None) -> Dict[str, torch.Tensor]:
        return {
            "pred": self.predict(frames, q_ids, q_mask, vis_feats=vis_feats),
            "target": self.encode_target(t_ids, t_mask),
        }

    # ---- Optimizer parameter groups (Y-Encoder at x0.05) ------------------
    def param_groups(self, lr: float, weight_decay: float = 0.0) -> List[Dict]:
        y, base = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (y if name.startswith("y_encoder.") else base).append(p)
        groups = []
        if base:
            groups.append({"params": base, "lr": lr, "weight_decay": weight_decay})
        if y:
            groups.append({"params": y, "lr": lr * self.cfg.y_lr_multiplier,
                           "weight_decay": weight_decay})
        return groups


def bidirectional_infonce(pred, target, temperature=0.07):
    """Bi-directional InfoNCE (paper §2): alignment + in-batch uniformity."""
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    logits = (pred @ target.T) / temperature
    labels = torch.arange(pred.shape[0], device=pred.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def random_batch(cfg: VLJEPAConfig, b: int, seed: int = 0):
    """Synthetic (frames, q_ids, q_mask, t_ids, t_mask) batch for tests/verify."""
    g = torch.Generator().manual_seed(seed)
    frames = torch.randn(b, 1, 3, 64, 64, generator=g)
    q_ids = torch.randint(1, cfg.vocab_size, (b, cfg.max_query_tokens), generator=g)
    q_mask = torch.ones(b, cfg.max_query_tokens, dtype=torch.long)
    q_mask[:, cfg.max_query_tokens // 2:] = 0  # half padded, to exercise [PAD]
    t_ids = torch.randint(1, cfg.vocab_size, (b, 8), generator=g)
    t_mask = torch.ones(b, 8, dtype=torch.long)
    return frames, q_ids, q_mask, t_ids, t_mask
