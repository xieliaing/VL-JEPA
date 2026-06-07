# VL-JEPA — From-Scratch Implementation

A **from-scratch, paper-faithful** PyTorch implementation of **VL-JEPA**
(*Joint Embedding Predictive Architecture for Vision-language*, Chen et al.,
Meta FAIR, [arXiv:2512.10942v2](https://arxiv.org/abs/2512.10942)).

Instead of autoregressively generating tokens, VL-JEPA predicts the **continuous
embedding** of the target text from a visual input and a textual query, training
in an abstract representation space with a bi-directional InfoNCE objective.

This repository implements the architecture from the ground up — including the
Llama-3 predictor blocks (RMSNorm, RoPE, grouped-query attention, SwiGLU) — with
a **fused SDPA attention path as the default**. The real gated HuggingFace
backbones (V-JEPA 2, Llama-3.2-1B, EmbeddingGemma-300M) can be swapped in via a
config switch.

## Architecture (paper §3.1)

| Component | This implementation | Paper |
|-----------|---------------------|-------|
| **X-Encoder** | frozen, stand-in conv ViT (or real V-JEPA 2 via HF) → visual tokens | frozen V-JEPA 2 ViT-L (304M) |
| **Predictor** | from-scratch Llama-3 blocks (RoPE/GQA/SwiGLU), **bidirectional**, avg-pool non-[PAD] | last 8 layers of Llama-3.2-1B (490M) |
| **Y-Encoder** | stand-in text encoder (or real EmbeddingGemma via HF), **LR ×0.05** | EmbeddingGemma-300M |
| **Shared space** | linear projection heads → **1536-d** | 1536-d |
| **Loss** | **bi-directional InfoNCE** (alignment + in-batch uniformity) | same |
| **Attention** | **SDPA (fused) by default**, eager available | causal mask disabled (bidirectional) |

The predictor uses **bidirectional** attention (causal mask disabled) so visual
and query tokens attend jointly — verified explicitly in the correctness suite.

## Repository structure

```
vljepa/
  __init__.py          # package exports
  model.py             # VLJEPAConfig, Llama-3 blocks (RoPE/RMSNorm/GQA+SDPA/SwiGLU),
                       # X/Y encoders, HF backend, VLJEPA, bidirectional InfoNCE
  data.py              # VisionLanguageJsonlDataset (image/video JSONL manifests)
scripts/
  verify.py            # correctness suite (--verify), --paper-budget, --hf-info
  benchmark.py         # throughput benchmark on DataComp+YFCC
  download_smoke_data.py  # fetch image-text pairs from DataComp + YFCC/CC3M
tests/
  test_model.py        # pytest: architecture, numerics, learnability, SDPA==eager
benchmarks/
  RESULTS.md           # recorded 3K-image throughput results
```

## Install

```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick start

```bash
# Correctness suite (small scale, CPU-friendly): 17 checks
PYTHONPATH=. python scripts/verify.py --verify

# Analytical parameter budget for the real paper config (no downloads)
PYTHONPATH=. python scripts/verify.py --paper-budget

# Plan for loading the real gated HuggingFace backbones
PYTHONPATH=. python scripts/verify.py --hf-info
```

`--verify` confirms: X-Encoder frozen / predictor & Y-Encoder trainable, Y-Encoder
LR ×0.05, 1536-d outputs, RMSNorm & RoPE vs reference, bidirectional-vs-causal
attention, InfoNCE scale (≈ ln B) & symmetry, gradient flow, and an
**overfit-a-batch** learnability test (loss → 0, retrieval acc → 100%).

> Reproducing the paper's *benchmark numbers* (e.g. 61.6% ImageNet zero-shot) is
> out of scope — that needs the full training run (2B samples, ~4 weeks on
> 192× H200). Correctness here means architecture, numerics, and learnability.

## Benchmark (DataComp + YFCC)

```bash
# 1. Fetch ~3000 real image-text pairs
PYTHONPATH=. python scripts/download_smoke_data.py --n-per-source 1500 \
    --manifest data/smoke_pretrain_manifest_3k.jsonl

# 2. Benchmark throughput (default: SDPA attention)
PYTHONPATH=. python scripts/benchmark.py --predictor proxy --attn sdpa --batch 64
```

Measured on an RTX 4090 Laptop (16 GB), bf16, 3000 images — full table in
[`benchmarks/RESULTS.md`](benchmarks/RESULTS.md):

| Proxy predictor, vit_b_16 vision | eager | **SDPA (default)** |
|----------------------------------|------:|-------------------:|
| live (img/s)                     | 251   | **282** |
| cached (img/s)                   | 369   | **444** |

SDPA gives **+20%** cached throughput. The frozen-feature cache (precompute the
frozen X-Encoder once, then skip it) adds a further speedup that scales with
X-Encoder cost. The real 490M paper predictor is memory-bound on a 16 GB GPU
(see RESULTS.md) and needs datacenter GPUs to train at speed.

## Real HuggingFace backbones

```python
from vljepa import build_paper_model_hf
model = build_paper_model_hf()   # real V-JEPA2 + Llama-3.2-1B(last 8) + EmbeddingGemma
```

Requires `huggingface-cli login` and accepting the licenses for
`meta-llama/Llama-3.2-1B` and `google/embeddinggemma-300m` (multi-GB download,
GPU recommended). The architecture is identical to the from-scratch path; only
the module internals and pretrained weights change.

The benchmark can mix in real backbones individually (V-JEPA 2 is ungated;
EmbeddingGemma needs the license accepted):

```bash
# real V-JEPA 2 X-Encoder + real EmbeddingGemma Y-Encoder + from-scratch predictor
PYTHONPATH=. python scripts/benchmark.py --vision vjepa2 --y-encoder embeddinggemma \
    --predictor proxy --frames 2 --batch 4
```

`--y-encoder bge-m3` swaps in `BAAI/bge-m3` (XLM-R-large, CLS-pooled) for
multilingual / CJK targets in place of EmbeddingGemma. `--vision siglip2` or
`--vision dinov2` swaps the V-JEPA 2 video encoder for a still-image vision tower
— appropriate for static-image inputs (faster; SigLIP 2 is vision-language
pretrained, DINOv2 is strong self-supervised). See `benchmarks/RESULTS.md`.

Use `--precision bf16` (pure bf16 weights + bf16 Adam, vs the default AMP's fp32
master) to roughly halve optimizer memory — this makes the **full paper-size
predictor + both real encoders** fit and scale on a 16 GB GPU (~26 img/s cached
at batch 16). See `benchmarks/RESULTS.md` for the measured authentic-backbone
throughput and the AMP-vs-bf16 comparison.

## Tests

```bash
python -m pytest tests/ -q
```

## Reference

Delong Chen, Mustafa Shukor, Théo Moutakanni, Willy Chung, Jade Yu, Tejaswi
Kasarla, Yejin Bang, Allen Bolourchi, Yann LeCun, Pascale Fung.
*VL-JEPA: Joint Embedding Predictive Architecture for Vision-language.*
arXiv:2512.10942v2, February 2026.
