"""The three ablation knobs read from config reach LRUChromaMemoryManager."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.lru_chroma_memory import LRUChromaMemoryManager


class _NullLLM:
    def chat(self, prompt):
        return ""


def _build(config):
    """Mirror BookWorld's construction of the lru_chroma manager from config."""
    return LRUChromaMemoryManager(
        llm=_NullLLM(), embedding_fn=lambda ts: [[0.0, 0.0, 1.0] for _ in ts],
        role_codes=["alice"], consensus_threshold=10, miss_retrieve_k=5,
        cache_capacity=20, consensus_top_k=5, language="zh",
        consensus_enabled=config.get("consensus_enabled", True),
        cache_strategy=config.get("cache_strategy", "lru"),
        index_backend=config.get("index_backend", "chroma"),
    )


def test_defaults_preserve_current_behavior():
    m = _build({})
    assert m.consensus_enabled is True
    assert m.cache_strategy == "lru"
    assert m.index_backend == "chroma"


def test_config_overrides_all_three_knobs():
    m = _build({"consensus_enabled": False,
                "cache_strategy": "fifo",
                "index_backend": "keyword"})
    assert m.consensus_enabled is False
    assert m.cache_strategy == "fifo"
    assert m.index_backend == "keyword"
