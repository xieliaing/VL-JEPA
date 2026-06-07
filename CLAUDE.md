# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A **from-scratch, paper-faithful** implementation of VL-JEPA (arXiv:2512.10942).
The model predicts continuous text embeddings from (visual input, textual query)
and trains with bi-directional InfoNCE in a 1536-d shared space. The Llama-3
predictor blocks are implemented from scratch; real HuggingFace backbones are an
optional `backend="hf"` switch.

There is **no training pipeline / CLI** in this repo — it is a reference
implementation plus correctness and throughput harnesses.

## Commands

```bash
PYTHONPATH=. python scripts/verify.py --verify        # 17-check correctness suite
PYTHONPATH=. python scripts/verify.py --paper-budget  # analytical 490M predictor budget
PYTHONPATH=. python scripts/verify.py --hf-info       # real HF backend plan
python -m pytest tests/ -q                            # pytest (test_model.py)

# Benchmark on real DataComp+YFCC (download first, then run)
PYTHONPATH=. python scripts/download_smoke_data.py --n-per-source 1500 \
    --manifest data/smoke_pretrain_manifest_3k.jsonl
PYTHONPATH=. python scripts/benchmark.py --predictor proxy --attn sdpa --batch 64
```

Run one pytest: `python -m pytest tests/test_model.py::test_sdpa_matches_eager -q`.

## Architecture map

- `vljepa/model.py` — the whole model:
  - `VLJEPAConfig` (small defaults; `paper_config()` for the real 490M scale)
  - Llama-3 blocks: `RMSNorm`, `build_rope_cache`/`apply_rope`, `repeat_kv`,
    `LlamaAttention` (GQA; `attn_impl` ∈ {`sdpa` default, `eager`}), `SwiGLU`, `LlamaBlock`
  - Encoders: `StandInXEncoder`/`StandInYEncoder` (no-download) and the HF backend
    (`HFXEncoder` = V-JEPA 2 video; `HFImageXEncoder` = SigLIP 2 / I-JEPA / DINOv2
    still-image vision tower; `HFYEncoder` = EmbeddingGemma/BGE-M3 with `y_pool`
    mean|cls; `HFLlamaPredictor`)
  - `VLJEPA` (forward returns `{pred, target}`; `param_groups` applies Y-Encoder LR ×0.05;
    `visual_features` exposes the frozen features for caching)
  - `bidirectional_infonce`, `random_batch`
- `vljepa/data.py` — `VisionLanguageJsonlDataset` (image/video JSONL → frames)
- `scripts/verify.py`, `scripts/benchmark.py`, `scripts/download_smoke_data.py`
- `tests/test_model.py`, `benchmarks/RESULTS.md`

## Key invariants (don't break these)

- **X-Encoder is frozen**; only predictor + Y-Encoder + projections train.
- **Predictor attention is bidirectional** (causal mask disabled) — the
  correctness suite asserts visual tokens depend on query tokens (and that the
  causal variant does not). If you touch `LlamaAttention`, keep this true.
- **`attn_impl="sdpa"` must stay numerically equal to `"eager"`**
  (`test_sdpa_matches_eager`).
- The from-scratch path must run with **no gated downloads**; HF code is only
  imported when `backend="hf"`.

## Scope note

Paper *benchmark* numbers (e.g. 61.6% ImageNet zero-shot) are not reproducible
here — they need the full training run (2B samples, ~4 weeks on 192× H200).
"Correctness" means architecture, numerics, and learnability (overfit-a-batch).
