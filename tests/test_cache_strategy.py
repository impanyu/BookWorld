"""Tests for the cache_strategy knob (lru / fifo / none)."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.lru_chroma_memory import (
    LRUChromaMemoryManager, RoleLRUCache, RoleFIFOCache,
)

_VECS = {
    "a storm wrecked the ship": [0.99, 0.01, 0.0],
    "a feast in the great hall": [0.0, 1.0, 0.0],
    "storm": [1.0, 0.0, 0.0],
}


def _embed(texts):
    return [_VECS.get(t, [0.0, 0.0, 1.0]) for t in texts]


class FakeLLM:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.prompts = []

    def chat(self, prompt):
        self.prompts.append(prompt)
        return self.responses.pop(0) if self.responses else ""


def _manager(llm, **kw):
    return LRUChromaMemoryManager(
        llm=llm, embedding_fn=_embed, role_codes=["alice", "bob"],
        persist_dir=None, consensus_threshold=999, miss_retrieve_k=5, **kw,
    )


def test_fifo_strategy_uses_fifo_cache():
    m = _manager(FakeLLM(), cache_strategy="fifo")
    assert isinstance(m._cache_for("alice"), RoleFIFOCache)


def test_lru_strategy_is_default():
    m = _manager(FakeLLM())
    assert isinstance(m._cache_for("alice"), RoleLRUCache)


def test_none_strategy_goes_straight_to_store_no_judge():
    llm = FakeLLM()  # no responses; judge must NOT be called
    m = _manager(llm, cache_strategy="none")
    m.add_memory("alice", "a storm wrecked the ship")

    res = m.retrieve_for_role("alice", "storm")

    assert "a storm wrecked the ship" in res
    assert llm.prompts == []              # no cache => no sufficiency judge
    assert m.cache_ids("alice") == []     # no cache state retained
    assert m._caches == {}                 # and none was lazily created
    assert m.current_cache_text("alice") == ""


def test_none_strategy_respects_owner_filter():
    m = _manager(FakeLLM(), cache_strategy="none")
    m.add_memory("bob", "a storm wrecked the ship")  # owned by bob

    res = m.retrieve_for_role("alice", "storm")
    assert "a storm wrecked the ship" not in res
