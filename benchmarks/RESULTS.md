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

Takeaways. SDPA increases cached throughput by approximately 20% in the proxy
configuration, where attention accounts for a significant fraction of each step.
For the paper predictor on 16 GB, the highest throughput is at batch 8 with SDPA
(30.4 img/s live, 29.6 cached), which is 1.53 times the eager rate. At batch 4 the
GPU is underutilized and the two attention implementations are comparable
(approximately 17 img/s). At batch 16 both implementations exceed VRAM and spill
to shared system memory. Eager attention at batch 8 is limited to approximately
19 img/s because its materialized score tensors increase memory pressure. The
490M paper predictor requires datacenter-class GPUs for full-speed training.

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

Takeaways. With the real V-JEPA 2 encoder the frozen-feature cache provides a
1.42× speedup at batch 4, compared with approximately 1.0× under the lightweight
`vit_b_16` stand-in; the cache benefit is proportional to X-Encoder cost.
Throughput is highest at batch 4. At batch 8 the resident 326M V-JEPA 2 weights
together with the paper predictor exceed 16 GB and cached throughput falls from
17.5 to 11.5 img/s. This is the most paper-authentic configuration that runs on a
16 GB GPU, combining the real V-JEPA 2 vision encoder with the paper-size Llama-3
predictor architecture; only the Llama and EmbeddingGemma weights are stand-ins,
replaceable via the `hf` backend.

### + real EmbeddingGemma Y-Encoder (`google/embeddinggemma-300m`)

Both encoders now real: V-JEPA 2 ViT-L X-Encoder **and** EmbeddingGemma-300M
(302.9M, trained @0.05×) as the Y-Encoder. Only the Llama-3 predictor uses
from-scratch weights. Reproduce:
`python scripts/benchmark.py --vision vjepa2 --y-encoder embeddinggemma --predictor proxy --frames 2 --batch 4`

| predictor | batch | live (img/s) | cached (img/s) | note |
|-----------|-------|-------------:|---------------:|------|
| proxy (768/4)  | 4 | 14.7 | 15.6 | runs cleanly on 16 GB |
| paper (2048/8) | 2 | 0.5  | 0.6  | severe VRAM spill — does not fit |
| paper (2048/8) | 1 | 0.3  | 0.2  | *worse* than b=2 — overflow is optimizer-state, not activations |

Takeaways. The combination of real V-JEPA 2, real EmbeddingGemma, and the proxy
predictor runs on a 16 GB GPU at approximately 15 img/s at batch 4; only the
Llama-3 predictor weights remain from-scratch. The paper-size predictor with both
300M-parameter real encoders does not fit within 16 GB, and reducing the batch
size does not resolve this. The binding constraint is batch-independent
optimizer-state memory: the frozen V-JEPA 2 (approximately 1.3 GB) plus
approximately 893M trainable parameters at 16 bytes each (fp32 parameter,
gradient, and two Adam moments) total about 15 GB before activations. The
configuration spills to shared system memory at any batch size, and batch 1
(0.3 img/s) is slower than batch 2 (0.5 img/s) because there is no batch
parallelism to amortize the spill. Reducing this footprint requires pure-bf16
weights, 8-bit Adam, LoRA on the predictor or Y-Encoder, or freezing the query
embedding. The cache speedup in this configuration is approximately 1.06×,
because EmbeddingGemma's per-step forward and backward increase total step cost
while the cache removes only the V-JEPA 2 forward (cache speedup is approximately
X-Encoder cost divided by total step cost).

### Pure bf16 unlocks the authentic paper config on 16 GB

The runs above use **AMP** (fp32 master weights + autocast bf16); the fp32 Adam
states are what overflow 16 GB. Switching to **pure bf16** (`--precision bf16`:
bf16 weights + bf16 Adam) halves that fixed footprint (~15 GB → ~7.8 GB) and makes
the **fully-authentic paper config** — real V-JEPA 2 + real EmbeddingGemma +
paper predictor (2048/8) + SDPA — fit and scale:

| batch | AMP (fp32 master) | bf16 live (img/s) | bf16 cached (img/s) |
|-------|------------------:|------------------:|--------------------:|
| 1  | 0.3 (spill) | —    | —    |
| 2  | 0.5 (spill) | 6.4  | 5.3  |
| 4  | —           | 10.4 | 10.1 |
| 8  | —           | 16.2 | 17.3 |
| **16** | —       | **21.7** | **26.0** |
| 32 | —           | 1.5 (spill) | 2.5 |

Reproduce: `python scripts/benchmark.py --predictor paper --vision vjepa2 --y-encoder embeddinggemma --precision bf16 --frames 2 --batch 16`

Takeaways. Pure bf16 reduces the fixed memory footprint sufficiently for the
authentic paper configuration to fit on the 16 GB GPU. Throughput increases with
batch size, from 6.4 img/s at batch 2 to 21.7 img/s live and 26.0 img/s cached at
batch 16; batch 32 exceeds VRAM and spills. Under fp32-master AMP the same
configuration does not fit and runs at 0.3 to 0.5 img/s. bf16 master weights with
bf16 Adam are less numerically stable than fp32-master AMP for extended training,
because bf16's 8-bit mantissa reduces the precision of Adam's second-moment
estimate. fp32 master weights with 8-bit Adam provide a more stable alternative
at a higher memory cost; pure bf16 is appropriate for throughput and feasibility
measurement.

### BGE-M3 Y-Encoder (multilingual / CJK)

`BAAI/bge-m3` (XLM-RoBERTa-large, ~568M, hidden 1024, [CLS]-pooled) in place of
EmbeddingGemma, selected for multilingual targets (e.g. Korean/English/Chinese
e-commerce text). Real V-JEPA 2 + BGE-M3 + paper predictor + SDPA, pure bf16.
Reproduce: `python scripts/benchmark.py --predictor paper --vision vjepa2 --y-encoder bge-m3 --precision bf16 --frames 2 --batch 8`

| batch | live (img/s) | cached (img/s) |
|-------|-------------:|---------------:|
| 4  | 11.6 | 14.0 |
| 8  | 18.3 | 21.2 |
| 16 | 23.5 (23.0–23.9) | 28.1 (27.3–29.5) |

Batch-16 figures are the mean of three consecutive runs (ranges in parentheses).

Takeaways. BGE-M3 (568M) replaces EmbeddingGemma (303M) with comparable
throughput, because each step is dominated by the V-JEPA 2 forward and the paper
predictor and the Y-Encoder is a small fraction of the step (18.3 vs 16.2 img/s
live at batch 8). Throughput scales cleanly with batch size through 16 (cached
14.0 → 21.2 → 28.1), indicating the configuration fits within 16 GB with no spill;
batch 16 is reproducible across runs (~23.5 live, ~28.1 cached). From a clean GPU
state batch 16 is stable; transient low readings observed earlier were caused by
residual VRAM left by back-to-back in-process runs, not by the batch size itself.
Pooling is the [CLS] token, matching BGE-M3's dense-embedding convention. BGE-M3
is the choice for multilingual targets; for English-only data EmbeddingGemma is
lighter at similar throughput.

### SigLIP 2 X-Encoder (static images)

`google/siglip2-base-patch16-256` vision tower (92.9M, hidden 768, 256 tokens at
256²) in place of V-JEPA 2 ViT-L, for static-image inputs. Stack: SigLIP 2 +
BGE-M3 + paper predictor + SDPA, pure bf16. Reproduce:
`python scripts/benchmark.py --predictor paper --vision siglip2 --y-encoder bge-m3 --precision bf16 --batch 16`

| batch | live (img/s) | cached (img/s) | V-JEPA 2 live / cached |
|-------|-------------:|---------------:|------------------------|
| 8  | 23.3 | 23.0 | 18.3 / 21.2 |
| 16 | 33.1 | 32.0 | 23.5 / 28.1 |
| 32 | 2.8 (spill) | 1.6 (spill) | — |

Takeaways. SigLIP 2's vision tower (92.9M) replaces V-JEPA 2 ViT-L (326M) for
static images and increases live throughput by approximately 40% at batch 16
(33.1 vs 23.5 img/s), because it is a smaller image encoder and processes a
single image rather than V-JEPA 2's duplicated frames. The frozen-feature cache
provides essentially no speedup with SigLIP 2 (approximately 1.0×): its forward
is a small fraction of the step, so precomputation and feature storage are
unnecessary and live training is the simpler path. Batch 16 is the ceiling for
this stack on 16 GB; batch 32 spills. SigLIP 2 is vision-language pretrained,
which may align more closely with text targets for product-text retrieval. For
training, apply SigLIP 2's own image normalization (0.5 mean/std) rather than the
ImageNet normalization used in these throughput measurements.

## Correctness

SDPA is numerically equivalent to eager (max output diff **3.6e-7**); see
`tests/test_model.py::test_sdpa_matches_eager`. Full architecture/numerics/
learnability suite: `python scripts/verify.py --verify` (17/17).
