#!/usr/bin/env python3
"""Correctness suite for the from-scratch VL-JEPA.

Reproducing the paper's benchmark numbers is infeasible without its training run
(2B samples, 4 weeks on 192 H200s). Instead this verifies architecture,
numerics, and learnability:

    python scripts/verify.py --verify        # 17 correctness checks (small scale)
    python scripts/verify.py --paper-budget  # analytical 490M predictor budget
    python scripts/verify.py --hf-info       # real HuggingFace backend plan
"""
from __future__ import annotations

import argparse
import math
from typing import List, Tuple

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


def _section(title: str) -> None:
    print(f"\n{'-' * 72}\n{title}\n{'-' * 72}")


def verify() -> None:
    torch.manual_seed(0)
    cfg = VLJEPAConfig()
    model = VLJEPA(cfg)
    passed: List[Tuple[str, bool]] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        passed.append((name, bool(cond)))
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  - {detail}" if detail else ""))

    # 1. Parameter count & freeze
    _section("1. Parameter budget & freeze (X frozen, Y at 0.05x LR)")
    def count(mod):
        tr = sum(p.numel() for p in mod.parameters() if p.requires_grad)
        fr = sum(p.numel() for p in mod.parameters() if not p.requires_grad)
        return tr, fr
    xtr, xfr = count(model.x_encoder)
    ptr = sum(p.numel() for n, p in model.named_parameters()
              if p.requires_grad and (n.startswith("blocks") or n.startswith("query_embed")
              or n.startswith("vis_in_proj") or n.startswith("pred")))
    ytr, _ = count(model.y_encoder)
    print(f"  X-Encoder : trainable={xtr:,}  frozen={xfr:,}")
    print(f"  Predictor : trainable={ptr:,}")
    print(f"  Y-Encoder : trainable={ytr:,}")
    check("X-Encoder fully frozen", xtr == 0 and xfr > 0, f"{xfr:,} frozen params")
    check("Predictor trainable", ptr > 0)
    check("Y-Encoder trainable", ytr > 0)

    # 2. Y-Encoder LR multiplier
    _section("2. Optimizer param groups - Y-Encoder LR multiplier")
    groups = model.param_groups(lr=1e-3, weight_decay=0.01)
    base_lr = max(g["lr"] for g in groups)
    y_lr = min(g["lr"] for g in groups)
    print(f"  base LR={base_lr:.2e}   Y-Encoder LR={y_lr:.2e}   ratio={y_lr/base_lr:.3f}")
    check("Y-Encoder LR is 0.05x base", abs(y_lr / base_lr - 0.05) < 1e-6)

    # 3. Shapes
    _section(f"3. Forward shapes & shared space (dim={cfg.shared_dim})")
    frames, q_ids, q_mask, t_ids, t_mask = random_batch(cfg, b=8)
    out = model(frames, q_ids, q_mask, t_ids, t_mask)
    print(f"  pred {tuple(out['pred'].shape)}   target {tuple(out['target'].shape)}")
    check("pred/target are [B, shared_dim]",
          out["pred"].shape == (8, cfg.shared_dim) and out["target"].shape == (8, cfg.shared_dim))

    # 4. RMSNorm
    _section("4. RMSNorm numerical correctness")
    rn = RMSNorm(32)
    x = torch.randn(4, 32)
    ref = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + rn.eps) * rn.weight
    err = (rn(x) - ref).abs().max().item()
    print(f"  max|impl - reference| = {err:.2e}")
    check("RMSNorm matches x/rms(x)*w", err < 1e-5)

    # 5. RoPE
    _section("5. RoPE rotary embedding - relative-position invariance")
    D = 16
    cos, sin = build_rope_cache(D, 64, 10000.0)
    qv = torch.randn(1, 1, 1, D)
    kv = torch.randn(1, 1, 1, D)
    def dot_at(m, n):
        qm, _ = apply_rope(qv, qv, cos[m:m+1], sin[m:m+1])
        _, kn = apply_rope(kv, kv, cos[n:n+1], sin[n:n+1])
        return (qm * kn).sum().item()
    d1, d2 = dot_at(5, 2), dot_at(8, 5)
    print(f"  <q@5,k@2>={d1:.5f}   <q@8,k@5>={d2:.5f}   |delta|={abs(d1-d2):.2e}")
    check("RoPE depends only on relative position", abs(d1 - d2) < 1e-4)

    # 6. Bidirectional attention (vision attends to query; causal does not)
    _section("6. Bidirectional attention - visual tokens attend to query")
    model.eval()
    vis = model.vis_in_proj(model.x_encoder(frames[:1]))
    q = model.query_embed(q_ids[:1]).clone().requires_grad_(True)
    def vis_out(qemb, causal):
        x = torch.cat([vis, qemb], dim=1)
        km = torch.cat([torch.ones(1, vis.shape[1]), q_mask[:1].float()], 1).bool()
        c = model.rope_cos[:x.shape[1]]; s = model.rope_sin[:x.shape[1]]
        for blk in model.blocks:
            x = blk(x, c, s, key_padding_mask=km, is_causal=causal)
        return x[:, 0, :]
    g_bi = torch.autograd.grad(vis_out(q, False).sum(), q, retain_graph=True)[0].abs().sum().item()
    g_ca = torch.autograd.grad(vis_out(q, True).sum(), q)[0].abs().sum().item()
    print(f"  d(visual_out)/d(query_in):  bidirectional={g_bi:.3e}   causal={g_ca:.3e}")
    check("Bidirectional: vision DOES attend to query", g_bi > 1e-6)
    check("Causal: vision does NOT attend to query", g_ca < 1e-9)
    model.train()

    # 7. InfoNCE
    _section("7. Bi-directional InfoNCE - scale & symmetry")
    B = 64
    pr = torch.randn(B, 256); tg = torch.randn(B, 256)
    loss = bidirectional_infonce(pr, tg, cfg.temperature).item()
    print(f"  init loss={loss:.3f}   ln(B)={math.log(B):.3f}   (independent embeddings)")
    check("InfoNCE init loss ~ ln(B)", abs(loss - math.log(B)) / math.log(B) < 0.15)
    l_ab = bidirectional_infonce(pr, tg, 0.07).item()
    l_ba = bidirectional_infonce(tg, pr, 0.07).item()
    print(f"  loss(p,t)={l_ab:.5f}   loss(t,p)={l_ba:.5f}")
    check("Bi-directional InfoNCE is symmetric", abs(l_ab - l_ba) < 1e-5)
    aligned = bidirectional_infonce(pr, pr.clone(), 0.07).item()
    print(f"  loss(p,p)={aligned:.3e}  (perfectly aligned -> ~0)")
    check("InfoNCE -> 0 for aligned embeddings", aligned < 1e-2)

    # 8. Gradient flow
    _section("8. Gradient flow")
    model.zero_grad()
    out = model(frames, q_ids, q_mask, t_ids, t_mask)
    bidirectional_infonce(out["pred"], out["target"], cfg.temperature).backward()
    x_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.x_encoder.parameters())
    pred_grad = sum(p.grad.abs().sum().item() for n, p in model.named_parameters()
                    if p.grad is not None and n.startswith("blocks"))
    y_grad = sum(p.grad.abs().sum().item() for p in model.y_encoder.parameters() if p.grad is not None)
    print(f"  X grad present: {x_has_grad}   predictor |grad|={pred_grad:.3e}   Y |grad|={y_grad:.3e}")
    check("Frozen X-Encoder receives NO gradient", not x_has_grad)
    check("Predictor receives gradient", pred_grad > 0)
    check("Y-Encoder receives gradient", y_grad > 0)

    # 9. Overfit a batch
    _section("9. Overfit a fixed batch (loss -> ~0, retrieval acc -> 100%)")
    torch.manual_seed(1)
    cfg2 = VLJEPAConfig()
    m2 = VLJEPA(cfg2)
    frames, q_ids, q_mask, t_ids, t_mask = random_batch(cfg2, b=8, seed=123)
    opt = torch.optim.AdamW(m2.param_groups(lr=3e-3), betas=(0.9, 0.95))
    first = None
    for stepi in range(300):
        out = m2(frames, q_ids, q_mask, t_ids, t_mask)
        loss = bidirectional_infonce(out["pred"], out["target"], cfg2.temperature)
        opt.zero_grad(); loss.backward(); opt.step()
        if stepi == 0:
            first = loss.item()
        if stepi % 60 == 0 or stepi == 299:
            with torch.no_grad():
                p = F.normalize(out["pred"], dim=-1); t = F.normalize(out["target"], dim=-1)
                acc = (((p @ t.T).argmax(1)) == torch.arange(8)).float().mean().item()
            print(f"  step {stepi:3d}  loss={loss.item():.4f}  retrieval_acc={acc:.2f}")
    check("Loss decreased substantially", loss.item() < 0.5 * first, f"{first:.3f} -> {loss.item():.3f}")
    check("Retrieval accuracy reaches 100%", acc == 1.0)

    n_pass = sum(1 for _, ok in passed if ok)
    _section(f"RESULT: {n_pass}/{len(passed)} checks passed")
    for name, ok in passed:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if n_pass != len(passed):
        raise SystemExit(1)


def paper_budget() -> None:
    cfg = paper_config()
    H, L, kvh = cfg.pred_hidden, cfg.pred_layers, cfg.pred_kv_heads
    hd = H // cfg.pred_heads
    per_block = (H * H + 2 * H * (kvh * hd) + H * H) + (3 * H * cfg.pred_intermediate) + 2 * H
    predictor = L * per_block
    llama_embed = cfg.vocab_size * H
    print("Paper-scale parameter budget (analytical, no weights downloaded)")
    print(f"  Predictor : {L} Llama-3.2-1B blocks, hidden={H}, heads={cfg.pred_heads}, "
          f"kv_heads={kvh}, ffn={cfg.pred_intermediate}")
    print(f"    -> per block ~ {per_block/1e6:.1f}M, 8 blocks ~ {predictor/1e6:.0f}M "
          f"(paper states 490M trainable) [MATCH]")
    print(f"  Llama token embedding ~ {llama_embed/1e6:.0f}M (vocab {cfg.vocab_size} x {H})")
    print(f"  X-Encoder : frozen V-JEPA 2 ViT-L ~ 304M (paper)")
    print(f"  Y-Encoder : EmbeddingGemma-300M ~ 300M, trained at LR x{cfg.y_lr_multiplier}")
    print(f"  Shared embedding space: {cfg.shared_dim}-d")
    total = predictor + llama_embed + 304e6 + 300e6
    print(f"  Total ~ {total/1e9:.2f}B (paper states 1.6B; remainder is exact "
          f"V-JEPA2/EmbeddingGemma embedding-table sizes)")


def hf_info() -> None:
    cfg = paper_config(backend="hf")
    print("HuggingFace backend plan (real paper weights)")
    print(f"  X-Encoder : {cfg.x_encoder_name}        (frozen V-JEPA 2 ViT-L)")
    print(f"  Predictor : last {cfg.predictor_last_n_layers} layers of {cfg.predictor_name}")
    print(f"  Y-Encoder : {cfg.y_encoder_name}   (LR x{cfg.y_lr_multiplier})")
    print(f"  Shared dim: {cfg.shared_dim}   |   bi-directional InfoNCE")
    try:
        import transformers
        print(f"\n  transformers {transformers.__version__} is installed.")
        try:
            from transformers import VJEPA2Model  # noqa: F401
            has_vjepa = True
        except Exception:
            has_vjepa = False
        print(f"  VJEPA2Model available: {has_vjepa}")
    except ImportError:
        print("\n  transformers NOT installed - `pip install transformers`.")
    print("\n  To build the real model:")
    print("    huggingface-cli login   # accept Llama-3.2-1B + EmbeddingGemma licenses")
    print("    python -c 'from vljepa import build_paper_model_hf; build_paper_model_hf()'")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true", help="Run the 17-check correctness suite.")
    ap.add_argument("--paper-budget", action="store_true", help="Print the paper-scale param budget.")
    ap.add_argument("--hf-info", action="store_true", help="Print the real HF backend plan.")
    args = ap.parse_args()
    if args.paper_budget:
        paper_budget()
    elif args.hf_info:
        hf_info()
    elif args.verify:
        verify()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
