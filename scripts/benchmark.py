#!/usr/bin/env python3
"""Throughput benchmark for the from-scratch VL-JEPA on real DataComp+YFCC.

Measures training-step throughput (img/s) on the smoke manifest (default 3000
images), bf16 on GPU. Sweeps:
  - predictor : proxy (4-layer/768) | paper (8-layer/2048, the real 490M)
  - attn      : sdpa (fused, default) | eager (manual softmax)
  - vision    : conv (lightweight stand-in) | vit_b_16 (real frozen ViT-B)
  - path      : live (X-Encoder every step) | cached (precomputed frozen feats)

Get the data first:
    python scripts/download_smoke_data.py --n-per-source 1500 \
        --manifest data/smoke_pretrain_manifest_3k.jsonl

Then, e.g.:
    python scripts/benchmark.py --predictor proxy --attn sdpa --batch 64
"""
from __future__ import annotations

import argparse
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from vljepa.data import VisionLanguageJsonlDataset
from vljepa.model import VLJEPA, VLJEPAConfig, bidirectional_infonce


class ViTBXEncoder(torch.nn.Module):
    """Frozen torchvision ViT-B/16 -> patch tokens [B, 196, 768]."""

    def __init__(self) -> None:
        super().__init__()
        from torchvision.models import vit_b_16
        self.net = vit_b_16(weights=None)
        self.out_dim = self.net.hidden_dim  # 768
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, frames):
        b, t, c, h, w = frames.shape
        x = frames.view(b * t, c, h, w)
        x = self.net._process_input(x)
        cls = self.net.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.net.encoder(x)
        return x[:, 1:, :].reshape(b, -1, self.out_dim)  # drop CLS -> [B,196,768]


def make_config(predictor: str, attn_impl: str) -> VLJEPAConfig:
    common = dict(vocab_size=50257, vis_dim=768, num_visual_tokens=196,
                  max_query_tokens=32, y_hidden=768, y_layers=2, shared_dim=1536,
                  attn_impl=attn_impl)
    if predictor == "proxy":
        return VLJEPAConfig(pred_hidden=768, pred_layers=4, pred_heads=12,
                            pred_kv_heads=4, pred_intermediate=2048, **common)
    if predictor == "paper":
        return VLJEPAConfig(pred_hidden=2048, pred_layers=8, pred_heads=32,
                            pred_kv_heads=8, pred_intermediate=8192, **common)
    raise ValueError(predictor)


class Collate:
    def __init__(self, tok, max_q=32, max_t=48):
        self.tok, self.max_q, self.max_t = tok, max_q, max_t

    def __call__(self, batch):
        frames = torch.stack([b["frames"] for b in batch], 0)
        q = self.tok([b["query"] for b in batch], max_length=self.max_q, padding="max_length",
                     truncation=True, return_tensors="pt")
        t = self.tok([b["target"] for b in batch], max_length=self.max_t, padding="max_length",
                     truncation=True, return_tensors="pt")
        return {"frames": frames,
                "q_ids": q["input_ids"], "q_mask": q["attention_mask"],
                "t_ids": t["input_ids"], "t_mask": t["attention_mask"]}


def step(model, batch, opt, device, dtype, vis_feats=None):
    with torch.autocast(device_type=device.type, dtype=dtype):
        out = model(
            None if vis_feats is not None else batch["frames"].to(device, non_blocking=True),
            batch["q_ids"].to(device), batch["q_mask"].to(device),
            batch["t_ids"].to(device), batch["t_mask"].to(device),
            vis_feats=vis_feats,
        )
        loss = bidirectional_infonce(out["pred"], out["target"], 0.07)
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)


def run_live(name, model, loader, device, dtype, steps, warmup):
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    it = iter(loader)

    def nxt(it):
        try:
            return next(it), it
        except StopIteration:
            it = iter(loader)
            return next(it), it

    for _ in range(warmup):
        b, it = nxt(it); step(model, b, opt, device, dtype)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter(); imgs = 0
    for _ in range(steps):
        b, it = nxt(it); step(model, b, opt, device, dtype); imgs += b["frames"].shape[0]
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"  {name:34s} {imgs/dt:8.1f} img/s   ({steps/dt:5.2f} steps/s)")
    return imgs / dt


def run_cached(name, model, feats, toks, device, dtype, steps, warmup, batch):
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    n = feats.shape[0]

    def sample():
        idx = torch.randint(0, n, (batch,))
        b = {"q_ids": toks["q_ids"][idx], "q_mask": toks["q_mask"][idx],
             "t_ids": toks["t_ids"][idx], "t_mask": toks["t_mask"][idx]}
        return b, feats[idx].to(device, non_blocking=True).float()

    for _ in range(warmup):
        b, vf = sample(); step(model, b, opt, device, dtype, vis_feats=vf)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        b, vf = sample(); step(model, b, opt, device, dtype, vis_feats=vf)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    imgs = steps * batch
    print(f"  {name:34s} {imgs/dt:8.1f} img/s   ({steps/dt:5.2f} steps/s)")
    return imgs / dt


def precompute(model, ds, collate, device, dtype, batch, workers):
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers,
                        pin_memory=True, collate_fn=collate, drop_last=False)
    feats, qi, qm, ti, tm = [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for b in tqdm(loader, desc="precompute", leave=False):
            fr = b["frames"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=dtype):
                feats.append(model.visual_features(fr).half().cpu())
            qi.append(b["q_ids"]); qm.append(b["q_mask"]); ti.append(b["t_ids"]); tm.append(b["t_mask"])
    model.train()
    toks = {"q_ids": torch.cat(qi), "q_mask": torch.cat(qm),
            "t_ids": torch.cat(ti), "t_mask": torch.cat(tm)}
    return torch.cat(feats, 0), toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/smoke_pretrain_manifest_3k.jsonl")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--predictor", choices=["proxy", "paper"], default="proxy")
    ap.add_argument("--attn", choices=["eager", "sdpa"], default="sdpa")
    ap.add_argument("--vision", default="conv,vit_b_16", help="comma list: conv, vit_b_16")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    ds = VisionLanguageJsonlDataset(args.manifest, num_frames=1, image_size=args.image_size)
    collate = Collate(tok, max_q=32, max_t=48)

    print(f"from-scratch VL-JEPA  predictor={args.predictor}  attn={args.attn}  device={device}  "
          f"batch={args.batch}  steps={args.steps}  dataset={len(ds)} imgs")

    results = {}
    for vision in [v.strip() for v in args.vision.split(",") if v.strip()]:
        print("=" * 70)
        print(f"VISION = {vision}")
        cfg = make_config(args.predictor, args.attn)
        model = VLJEPA(cfg).to(device)
        if vision == "vit_b_16":
            model.x_encoder = ViTBXEncoder().to(device)

        loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                            pin_memory=True, collate_fn=collate, drop_last=True,
                            persistent_workers=(args.workers > 0),
                            prefetch_factor=(4 if args.workers > 0 else None))
        live = run_live(f"{vision} LIVE (X-Enc each step)", model, loader, device, dtype,
                        args.steps, args.warmup)
        feats, toks = precompute(model, ds, collate, device, dtype, args.batch, args.workers)
        cached = run_cached(f"{vision} CACHED (frozen feats)", model, feats, toks, device, dtype,
                            args.steps, args.warmup, args.batch)
        results[vision] = (live, cached)
        del model, loader, feats, toks
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("=" * 70)
    for vision, (live, cached) in results.items():
        print(f"  {vision:9s}  live={live:7.1f} img/s   cached={cached:7.1f} img/s   "
              f"cache speedup={cached/live:4.2f}x")


if __name__ == "__main__":
    main()
