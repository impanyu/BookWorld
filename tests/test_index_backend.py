"""Tests for the index_backend knob (chroma / keyword) at the manager level."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.lru_chroma_memory import (
    LRUChromaMemoryManager, KeywordSQLiteStore, SharedChromaStore,
)


def _embed(texts):
    return [[0.0, 0.0, 1.0] for _ in texts]


class KwLLM:
    """LLM stub: 'chat' returns a JSON keyword list = space-split of the text
    that follows the '## Text:' marker; judge/consensus return canned values."""

    def __init__(self):
        self.responses = []

    def chat(self, prompt):
        if "## Text:" in prompt:  # keyword-extraction prompt
            body = prompt.split("## Text:")[1].split("Return ONLY")[0].strip()
            import json
            return json.dumps(sorted(set(body.lower().split())))
        return self.responses.pop(0) if self.responses else "no"


def _manager(**kw):
    kw.setdefault("consensus_threshold", 999)
    return LRUChromaMemoryManager(
        llm=KwLLM(), embedding_fn=_embed, role_codes=["alice", "bob"],
        persist_dir=None, miss_retrieve_k=5, **kw,
    )


def test_default_backend_is_chroma():
    m = _manager()
    assert isinstance(m.store, SharedChromaStore)


def test_keyword_backend_selected():
    m = _manager(index_backend="keyword")
    assert isinstance(m.store, KeywordSQLiteStore)


def test_keyword_backend_end_to_end_retrieval():
    m = _manager(index_backend="keyword", cache_strategy="none")
    m.add_memory("alice", "storm wrecked the ship")
    m.add_memory("alice", "a feast in the great hall")

    res = m.retrieve_for_role("alice", "storm ship")
    assert "storm wrecked the ship" in res


def test_keyword_backend_consensus_runs():
    # threshold=2 triggers consensus over the keyword store; LLM consensus
    # response merges the two near-duplicate memories into one.
    m = _manager(index_backend="keyword", consensus_threshold=2)
    m.llm.responses = ['["king died spring", "", ""]']  # consensus output
    m.add_memory("alice", "king died spring crops")
    m.add_memory("bob", "king died spring comet")

    texts = {t for _, t in m.store.query_global("king", top_k=10)}
    assert "king died spring" in texts
