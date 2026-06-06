#!/usr/bin/env python3
from __future__ import annotations

"""Download a small sample of image-text pairs from DataComp and YFCC-100M.

DataComp: streams image-URL + alt-text caption metadata from HuggingFace
(mlfoundations/datacomp_1b or datacomp_small), then fetches images from source
URLs.  Many URLs are stale; the script retries the next row on failure.

YFCC-100M proxy: uses google-research-datasets/conceptual_captions (CC3M),
which is drawn from the same Flickr / Common Crawl pool as YFCC and ships the
same image-URL + caption schema.  Alternatively uses nlphuji/flickr30k-images
(Parquet-native build of Flickr30k, a canonical YFCC subset).

Fallback: if network access or HuggingFace is unavailable the script falls back
to synthetic PIL images so the smoke training run can proceed offline.

Requirements:
    pip install datasets requests Pillow

Usage:
    python scripts/download_smoke_data.py
    python scripts/download_smoke_data.py --n-per-source 100 --out-dir data/smoke
    python scripts/download_smoke_data.py --source datacomp
    python scripts/download_smoke_data.py --source yfcc
    python scripts/download_smoke_data.py --synthetic   # offline fallback only
"""

import argparse
import hashlib
import io
import json
import random
import sys
import time
from pathlib import Path
from typing import Iterator, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; VL-JEPA-smoke/1.0)"
    )
}
_TIMEOUT = 8


# ---------------------------------------------------------------------------
# Shared image fetch / save helpers
# ---------------------------------------------------------------------------

def _fetch_bytes(url: str) -> Optional[bytes]:
    try:
        with urlopen(Request(url, headers=_HEADERS), timeout=_TIMEOUT) as r:
            return r.read()
    except (URLError, HTTPError, OSError, ValueError):
        return None


def _save_pil(img, dest: Path) -> bool:
    try:
        img.convert("RGB").save(dest, format="JPEG", quality=90)
        return True
    except Exception:
        return False


def _save_bytes(raw: bytes, dest: Path) -> bool:
    try:
        from PIL import Image
        return _save_pil(Image.open(io.BytesIO(raw)), dest)
    except Exception:
        return False


def _url_fname(url: str, prefix: str) -> str:
    return f"{prefix}_{hashlib.md5(url.encode()).hexdigest()[:12]}.jpg"


# ---------------------------------------------------------------------------
# Dataset iterators
# ---------------------------------------------------------------------------

def _hf_stream(repo_id: str, split: str, **kw) -> Optional[object]:
    """Open a streaming HuggingFace dataset; return None on any error."""
    try:
        from datasets import load_dataset
        return load_dataset(repo_id, split=split, streaming=True, **kw)
    except Exception as exc:
        print(f"  [warn] {repo_id}/{split}: {exc}")
        return None


def _iter_datacomp() -> Iterator[dict]:
    """Yield {url, caption} rows from DataComp metadata."""
    print("[DataComp] Streaming from HuggingFace …")
    for repo, split in [
        ("mlfoundations/datacomp_small", "train"),
        ("mlfoundations/datacomp_1b",    "train"),
        # CC12M is comparable in origin and format to DataComp.
        ("conceptual_12m",               "train"),
    ]:
        ds = _hf_stream(repo, split)
        if ds is None:
            continue
        for row in ds:
            url = row.get("url") or row.get("image_url") or ""
            caption = (
                row.get("caption") or row.get("text") or row.get("alt_text") or ""
            ).strip()
            if url and caption:
                yield {"url": url, "caption": caption}
        return
    print("[DataComp] No accessible DataComp split found.")


def _iter_yfcc() -> Iterator[dict]:
    """Yield {url|pil_image, caption} rows from a YFCC / Flickr proxy."""
    print("[YFCC] Streaming from HuggingFace …")

    # Option A: Flickr30k images bundled directly in the dataset (Parquet).
    for repo, split in [
        # Parquet-native build — does not use a loading script.
        ("nlphuji/flickr30k-images", "test"),
        ("nlphuji/flickr30k-images", "train"),
    ]:
        ds = _hf_stream(repo, split)
        if ds is None:
            continue
        for row in ds:
            img_obj = row.get("image")
            captions = row.get("caption") or row.get("captions") or []
            if isinstance(captions, str):
                captions = [captions]
            if img_obj is not None and captions:
                yield {"pil_image": img_obj, "caption": captions[0].strip()}
        return

    # Option B: CC3M / Conceptual Captions — same URL + caption schema as YFCC.
    for repo, split in [
        ("google-research-datasets/conceptual_captions", "validation"),
        ("google-research-datasets/conceptual_captions", "train"),
    ]:
        ds = _hf_stream(repo, split)
        if ds is None:
            continue
        for row in ds:
            url = row.get("image_url") or row.get("url") or ""
            caption = (row.get("caption") or "").strip()
            if url and caption:
                yield {"url": url, "caption": caption}
        return

    print("[YFCC] No accessible YFCC/Flickr/CC split found.")


# ---------------------------------------------------------------------------
# Synthetic fallback
# ---------------------------------------------------------------------------

_SYNTH_OBJECTS = [
    ("a red car parked on a street", (200, 50, 50)),
    ("a blue boat on calm water", (50, 100, 200)),
    ("a green field under a cloudy sky", (60, 150, 60)),
    ("a yellow flower in a garden", (220, 200, 40)),
    ("a white cat sitting on a windowsill", (230, 230, 230)),
    ("a brown dog running in a park", (160, 100, 60)),
    ("a purple mountain at sunset", (120, 60, 180)),
    ("a black bicycle leaning against a wall", (30, 30, 30)),
    ("an orange sunset over the ocean", (240, 130, 30)),
    ("a grey city skyline at dusk", (130, 130, 140)),
]

def _make_synthetic_images(n: int, img_dir: Path, source: str) -> list:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("Pillow required: pip install Pillow")
        return []
    img_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        caption, color = _SYNTH_OBJECTS[i % len(_SYNTH_OBJECTS)]
        img = Image.new("RGB", (224, 224), color=color)
        draw = ImageDraw.Draw(img)
        # Add noise-like variation so batches aren't identical.
        for _ in range(30):
            x, y = random.randint(0, 223), random.randint(0, 223)
            r = random.randint(5, 30)
            rc = tuple(min(255, c + random.randint(-40, 40)) for c in color)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=rc)
        path = img_dir / f"{source}_{i:04d}.jpg"
        img.save(path, format="JPEG", quality=90)
        rows.append({"image": str(path), "query": "Describe this image.", "target": caption})
    return rows


# ---------------------------------------------------------------------------
# Download loop
# ---------------------------------------------------------------------------

def download_source(
    source_name: str,
    row_iter: Iterator[dict],
    n_want: int,
    img_dir: Path,
    manifest_rows: list,
    query_text: str = "Describe this image.",
) -> int:
    img_dir.mkdir(parents=True, exist_ok=True)
    downloaded = attempted = 0

    for row in row_iter:
        if downloaded >= n_want:
            break
        attempted += 1
        caption = row["caption"].strip()
        if not caption:
            continue

        if "pil_image" in row:
            fname = f"{source_name}_{attempted:06d}.jpg"
            dest = img_dir / fname
            ok = _save_pil(row["pil_image"], dest)
        else:
            url = row.get("url", "")
            if not url:
                continue
            raw = _fetch_bytes(url)
            if raw is None:
                continue
            dest = img_dir / _url_fname(url, source_name)
            ok = _save_bytes(raw, dest)
            # Polite pacing for URL-based sources.
            if attempted % 5 == 0:
                time.sleep(0.15)

        if ok and dest.exists():
            manifest_rows.append({
                "image": str(dest),
                "query": query_text,
                "target": caption,
            })
            downloaded += 1
            if downloaded % 10 == 0 or downloaded == n_want:
                print(f"  [{source_name}] {downloaded}/{n_want} saved "
                      f"(tried {attempted})")

    print(f"[{source_name}] Done: {downloaded} collected from {attempted} tried.")
    return downloaded


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download smoke-test image-text data for VL-JEPA"
    )
    parser.add_argument("--n-per-source", type=int, default=50)
    parser.add_argument("--out-dir", type=str, default="data/smoke")
    parser.add_argument("--manifest", type=str,
                        default="data/smoke_pretrain_manifest.jsonl")
    parser.add_argument("--source", choices=["datacomp", "yfcc", "both"],
                        default="both")
    parser.add_argument("--synthetic", action="store_true",
                        help="Skip network downloads; generate synthetic images only.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_rows: list = []
    n = args.n_per_source

    if args.synthetic:
        manifest_rows += _make_synthetic_images(n, out_dir / "synthetic_dc", "datacomp")
        manifest_rows += _make_synthetic_images(n, out_dir / "synthetic_yf", "yfcc")
    else:
        if args.source in ("datacomp", "both"):
            cnt = download_source("datacomp", _iter_datacomp(), n,
                                  out_dir / "datacomp", manifest_rows)
            if cnt == 0:
                print("[DataComp] Network download failed — using synthetic fallback.")
                manifest_rows += _make_synthetic_images(
                    n, out_dir / "synthetic_dc", "datacomp")

        if args.source in ("yfcc", "both"):
            cnt = download_source("yfcc", _iter_yfcc(), n,
                                  out_dir / "yfcc", manifest_rows)
            if cnt == 0:
                print("[YFCC] Network download failed — using synthetic fallback.")
                manifest_rows += _make_synthetic_images(
                    n, out_dir / "synthetic_yf", "yfcc")

    if not manifest_rows:
        print("No samples available. Exiting.")
        sys.exit(1)

    with open(manifest_path, "w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row) + "\n")

    print(f"\nWrote {len(manifest_rows)} samples → {manifest_path}")
    print("Run smoke training with:")
    print("  PYTHONPATH=. python scripts/train_pretrain.py "
          "--config vljepa/configs/pretrain_smoke.yaml")


if __name__ == "__main__":
    main()
