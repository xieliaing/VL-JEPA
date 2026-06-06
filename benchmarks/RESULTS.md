# Throughput Benchmarks

Training-step throughput of the from-scratch VL-JEPA on **3000 real image–text
pairs** sampled from **DataComp + YFCC-100M** (via `scripts/download_smoke_data.py`).

- **Hardware:** NVIDIA RTX 4090 Laptop (16 GB), bf16 autocast
- **Data:** 1500 DataComp + 1500 YFCC, 224², 1 frame/sample, gpt2 tokenizer
- **Paths:** `live` = run the frozen X-Encoder every step; `cached` = precompute
  frozen visual features once, then skip the X-Encoder
- Reproduce: `python scripts/benchmark.py --predictor {proxy,paper} --attn {sdpa,eager}`

## Proxy predictor (4-layer / 768) — batch 64

The directly comparable config size. **SDPA is the default.**

| vision | attn | live (img/s) | cached (img/s) | cache speedup |
|--------|------|-------------:|---------------:|--------------:|
| conv (stand-in) | eager | 334 | 373 | 1.12× |
| conv (stand-in) | **sdpa** | **400** | **449** | 1.12× |
| vit_b_16 (real) | eager | 251 | 369 | 1.47× |
| vit_b_16 (real) | **sdpa** | **282** | **444** | 1.57× |

SDPA gives **+20% cached** throughput over eager. The frozen-feature cache gain
scales with X-Encoder cost (1.12× under the cheap conv stand-in, 1.57× under a
real ViT-B/16).

## Paper predictor (8-layer / 2048, the real 490M) — vit_b_16 vision

The full-size predictor is **memory-bound** on the 16 GB laptop GPU: above a
modest batch it spills to shared system memory (Windows WDDM) instead of OOMing.
A batch sweep finds **batch 8 with SDPA is the sweet spot**. (The paper trained
this on 192× H200, 141 GB each.)

| batch | attn | live (img/s) | cached (img/s) | note |
|-------|------|-------------:|---------------:|------|
| 4  | eager | 18.0 | 17.5 | fits VRAM; GPU underutilized at tiny batch |
| 4  | sdpa  | 19.5 | 17.2 | ≈ eager — attention not the bottleneck at b=4 |
| **8**  | eager | 19.2 | 19.3 | nearing VRAM pressure; can't exploit the bigger batch |
| **8**  | **sdpa**  | **30.4** | **29.6** | **best** — fits VRAM *and* utilizes GPU; **1.53× over eager** |
| 16 | eager | 2.2 | 8.6 | VRAM overflow → shared-memory spill |
| 16 | sdpa  | 9.4 | 8.8 | also spills at paper scale on 16 GB |

**Takeaways:** SDPA delivers a clean +20% where attention is on the critical path
(proxy config), and its memory efficiency raises the batch ceiling before VRAM
spill. For the paper predictor on 16 GB, **batch 8 + SDPA is optimal (~30 img/s,
1.53× over eager)**: batch 4 underutilizes the GPU (eager ≈ SDPA ≈ 17), batch 16
spills for both. Eager can't exploit batch 8 (its materialized score tensors hit
memory pressure, capping it at ~19), which is exactly where SDPA wins. The
paper-scale 490M predictor still needs datacenter GPUs to train at full speed.

## Authentic V-JEPA 2 X-Encoder (real `facebook/vjepa2-vitl-fpc64-256`)

The X-Encoder above is a lightweight stand-in (`vit_b_16`). This section uses the
**real frozen V-JEPA 2 ViT-L** (326M, hidden 1024) as in the paper, with the
from-scratch paper predictor (8-layer/2048) + SDPA. Images are duplicated to
**2 frames @ 256²** (tubelet 2 → **256 visual tokens**), matching the paper's
image stage. It runs on the 16 GB RTX 4090 Laptop at small batch.

| batch | live (img/s) | cached (img/s) | cache speedup |
|-------|-------------:|---------------:|--------------:|
| **4** | 12.3 | **17.5** | **1.42×** |
| 8     | 8.8  | 11.5 | 1.32× |

Reproduce: `python scripts/benchmark.py --predictor paper --attn sdpa --vision vjepa2 --frames 2 --batch 4`

**Takeaways:** With the real, heavy V-JEPA 2 encoder the **frozen-feature cache
now gives a clear speedup (1.42×)** — versus ~1.0× under the lightweight
`vit_b_16` stand-in — confirming that the cache benefit scales with X-Encoder
cost. Batch 4 is the sweet spot; at batch 8 the resident 326M V-JEPA 2 weights
plus the paper predictor exceed 16 GB and spill (cached 17.5 → 11.5). This is the
most paper-authentic configuration runnable on a 16 GB GPU (real V-JEPA 2 vision
+ paper-size Llama-3 predictor architecture); only the Llama/EmbeddingGemma
*weights* are stand-ins, swappable via the `hf` backend.

## Correctness

SDPA is numerically equivalent to eager (max output diff **3.6e-7**); see
`tests/test_model.py::test_sdpa_matches_eager`. Full architecture/numerics/
learnability suite: `python scripts/verify.py --verify` (17/17).
