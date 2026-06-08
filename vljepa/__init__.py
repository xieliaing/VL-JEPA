"""VL-JEPA — from-scratch, paper-faithful implementation (arXiv:2512.10942)."""

from vljepa.model import (
    VLJEPA,
    VLJEPAConfig,
    bidirectional_infonce,
    bipartite_soft_match_merge,
    build_paper_model_hf,
    paper_config,
    random_batch,
)

__all__ = [
    "VLJEPA",
    "VLJEPAConfig",
    "bidirectional_infonce",
    "bipartite_soft_match_merge",
    "build_paper_model_hf",
    "paper_config",
    "random_batch",
]
