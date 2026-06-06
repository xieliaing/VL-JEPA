"""Pytest correctness suite for the from-scratch VL-JEPA.

Verifies architecture, numerics, and learnability (paper-benchmark reproduction
is out of scope — it needs the full training run). The HF-backend test is
skipped unless VLJEPA_RUN_HF=1 (it needs gated downloads).
"""

import math
import os

import pytest
import torch
import torch.nn.functional as F

from vljepa.model import (
    RMSNorm,
    VLJEPA,
    VLJEPAConfig,
    apply_rope,
    bidirectional_infonce,
    build_rope_cache,
    paper_config,
    random_batch,
)


@pytest.fixture
def cfg() -> VLJEPAConfig:
    return VLJEPAConfig()


@pytest.fixture
def model(cfg) -> VLJEPA:
    torch.manual_seed(0)
    return VLJEPA(cfg)


@pytest.fixture
def batch(cfg):
    return random_batch(cfg, b=8, seed=42)


# ---- Parameter freeze / trainability -------------------------------------

def test_x_encoder_fully_frozen(model):
    trainable = sum(p.numel() for p in model.x_encoder.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.x_encoder.parameters() if not p.requires_grad)
    assert trainable == 0
    assert frozen > 0


def test_predictor_and_y_encoder_trainable(model):
    pred = sum(p.numel() for n, p in model.named_parameters()
               if p.requires_grad and n.startswith("blocks"))
    y = sum(p.numel() for p in model.y_encoder.parameters() if p.requires_grad)
    assert pred > 0
    assert y > 0


# ---- Y-Encoder LR multiplier ---------------------------------------------

def test_y_encoder_lr_multiplier(model):
    groups = model.param_groups(lr=1e-3, weight_decay=0.01)
    base_lr = max(g["lr"] for g in groups)
    y_lr = min(g["lr"] for g in groups)
    assert abs(y_lr / base_lr - 0.05) < 1e-6


# ---- Shapes ---------------------------------------------------------------

def test_forward_shapes(model, cfg, batch):
    out = model(*batch)
    assert out["pred"].shape == (8, cfg.shared_dim)
    assert out["target"].shape == (8, cfg.shared_dim)


# ---- RMSNorm --------------------------------------------------------------

def test_rmsnorm_matches_reference():
    rn = RMSNorm(32)
    x = torch.randn(4, 32)
    ref = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + rn.eps) * rn.weight
    assert torch.allclose(rn(x), ref, atol=1e-5)


# ---- RoPE -----------------------------------------------------------------

def test_rope_relative_position_invariance():
    D = 16
    cos, sin = build_rope_cache(D, 64, theta=10000.0)
    torch.manual_seed(0)
    qv, kv = torch.randn(1, 1, 1, D), torch.randn(1, 1, 1, D)

    def dot_at(m, n):
        qm, _ = apply_rope(qv, qv, cos[m:m + 1], sin[m:m + 1])
        _, kn = apply_rope(kv, kv, cos[n:n + 1], sin[n:n + 1])
        return (qm * kn).sum().item()

    assert abs(dot_at(5, 2) - dot_at(8, 5)) < 1e-4


# ---- Bidirectional vs causal attention -----------------------------------

def _visual_output_grad_wrt_query(model, batch, causal: bool) -> float:
    frames, q_ids, q_mask, _, _ = batch
    model.eval()
    vis = model.vis_in_proj(model.x_encoder(frames[:1]))
    q = model.query_embed(q_ids[:1]).clone().requires_grad_(True)
    x = torch.cat([vis, q], dim=1)
    km = torch.cat([torch.ones(1, vis.shape[1]), q_mask[:1].float()], dim=1).bool()
    cos = model.rope_cos[: x.shape[1]]
    sin = model.rope_sin[: x.shape[1]]
    for blk in model.blocks:
        x = blk(x, cos, sin, key_padding_mask=km, is_causal=causal)
    grad = torch.autograd.grad(x[:, 0, :].sum(), q)[0]
    return grad.abs().sum().item()


def test_bidirectional_vision_attends_to_query(model, batch):
    assert _visual_output_grad_wrt_query(model, batch, causal=False) > 1e-6


def test_causal_vision_does_not_attend_to_query(model, batch):
    assert _visual_output_grad_wrt_query(model, batch, causal=True) < 1e-9


# ---- Bi-directional InfoNCE ----------------------------------------------

def test_infonce_init_loss_approx_ln_batch():
    B = 64
    torch.manual_seed(0)
    pred, target = torch.randn(B, 256), torch.randn(B, 256)
    loss = bidirectional_infonce(pred, target, 0.07).item()
    assert abs(loss - math.log(B)) / math.log(B) < 0.15


def test_infonce_symmetric():
    torch.manual_seed(0)
    pred, target = torch.randn(16, 64), torch.randn(16, 64)
    assert abs(bidirectional_infonce(pred, target).item()
               - bidirectional_infonce(target, pred).item()) < 1e-5


def test_infonce_zero_for_aligned():
    torch.manual_seed(0)
    pred = torch.randn(16, 64)
    assert bidirectional_infonce(pred, pred.clone(), 0.07).item() < 1e-2


# ---- Gradient flow --------------------------------------------------------

def test_frozen_x_receives_no_gradient(model, batch, cfg):
    model.zero_grad()
    out = model(*batch)
    bidirectional_infonce(out["pred"], out["target"], cfg.temperature).backward()
    x_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                 for p in model.x_encoder.parameters())
    assert not x_grad


def test_predictor_and_y_receive_gradient(model, batch, cfg):
    model.zero_grad()
    out = model(*batch)
    bidirectional_infonce(out["pred"], out["target"], cfg.temperature).backward()
    pred_grad = sum(p.grad.abs().sum().item() for n, p in model.named_parameters()
                    if p.grad is not None and n.startswith("blocks"))
    y_grad = sum(p.grad.abs().sum().item() for p in model.y_encoder.parameters()
                 if p.grad is not None)
    assert pred_grad > 0
    assert y_grad > 0


# ---- Overfit-a-batch ------------------------------------------------------

def test_overfit_a_batch_reaches_full_retrieval():
    torch.manual_seed(1)
    cfg = VLJEPAConfig()
    model = VLJEPA(cfg)
    frames, q_ids, q_mask, t_ids, t_mask = random_batch(cfg, b=8, seed=123)
    opt = torch.optim.AdamW(model.param_groups(lr=3e-3), betas=(0.9, 0.95))
    first = None
    for stepi in range(200):
        out = model(frames, q_ids, q_mask, t_ids, t_mask)
        loss = bidirectional_infonce(out["pred"], out["target"], cfg.temperature)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if stepi == 0:
            first = loss.item()
    with torch.no_grad():
        p = F.normalize(out["pred"], dim=-1)
        t = F.normalize(out["target"], dim=-1)
        acc = ((p @ t.T).argmax(1) == torch.arange(8)).float().mean().item()
    assert loss.item() < 0.5 * first
    assert acc == 1.0


# ---- Analytical paper-scale predictor budget -----------------------------

def test_paper_predictor_param_budget_matches_490M():
    cfg = paper_config()
    H, L, kvh = cfg.pred_hidden, cfg.pred_layers, cfg.pred_kv_heads
    hd = H // cfg.pred_heads
    per_block = (H * H + 2 * H * (kvh * hd) + H * H) + 3 * H * cfg.pred_intermediate + 2 * H
    predictor = L * per_block
    assert 480e6 < predictor < 500e6


# ---- SDPA == eager equivalence -------------------------------------------

def test_sdpa_matches_eager():
    torch.manual_seed(0)
    eager = VLJEPA(VLJEPAConfig(attn_impl="eager")).eval()
    torch.manual_seed(0)
    sdpa = VLJEPA(VLJEPAConfig(attn_impl="sdpa")).eval()
    sdpa.load_state_dict(eager.state_dict())
    batch = random_batch(VLJEPAConfig(), b=8, seed=1)
    with torch.no_grad():
        oe = eager(*batch)
        os_ = sdpa(*batch)
    assert torch.allclose(oe["pred"], os_["pred"], atol=1e-4)
    assert torch.allclose(oe["target"], os_["target"], atol=1e-4)


# ---- HF backend smoke test (gated; opt-in) -------------------------------

@pytest.mark.skipif(not os.environ.get("VLJEPA_RUN_HF"),
                    reason="set VLJEPA_RUN_HF=1 with HuggingFace gated access to run")
def test_hf_backend_builds_and_forwards():
    from vljepa.model import build_paper_model_hf
    model = build_paper_model_hf().eval()
    cfg = model.cfg
    frames = torch.randn(2, 8, 3, 256, 256)
    q_ids = torch.randint(1, 1000, (2, 16))
    q_mask = torch.ones(2, 16, dtype=torch.long)
    t_ids = torch.randint(1, 1000, (2, 16))
    t_mask = torch.ones(2, 16, dtype=torch.long)
    with torch.no_grad():
        out = model(frames, q_ids, q_mask, t_ids, t_mask)
    assert out["pred"].shape == (2, cfg.shared_dim)
    assert out["target"].shape == (2, cfg.shared_dim)
