"""Tests for the consensus on/off knob of LRUChromaMemoryManager."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.lru_chroma_memory import LRUChromaMemoryManager

_VECS = {"k1": [1.0, 0.0, 0.0], "k2": [0.99, 0.01, 0.0]}


def _embed(texts):
    return [_VECS.get(t, [0.0, 0.0, 1.0]) for t in texts]


class FakeLLM:
    def __init__(self):
        self.prompts = []

    def chat(self, prompt):
        self.prompts.append(prompt)
        return ""


def test_consensus_disabled_never_enqueues_or_calls_llm():
    llm = FakeLLM()
    m = LRUChromaMemoryManager(
        llm=llm, embedding_fn=_embed, role_codes=["alice", "bob"],
        persist_dir=None, consensus_threshold=2, consensus_enabled=False,
    )
    m.add_memory("alice", "k1")
    m.add_memory("bob", "k2")  # would trigger consensus if enabled (threshold=2)

    assert m.pending_count() == 0
    assert llm.prompts == []          # consensus LLM never invoked
    m.run_consensus()                 # explicit call is a no-op
    assert llm.prompts == []
    # both memories still independently present (not merged)
    texts = {t for _, t in m.store.query_global("k1", top_k=10)}
    assert texts == {"k1", "k2"}
