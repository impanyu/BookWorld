"""Tests for LRUChromaMemoryManager — consensus merge flow.

Consensus LLM contract: given input array [m, n1..nk] it returns a JSON array
[consensus, residual_0, residual_1, ..., residual_k] where an empty residual
means that input memory is fully covered by the consensus.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.lru_chroma_memory import LRUChromaMemoryManager


# Everything "king"-ish is mutually near so it becomes each other's neighbor.
_VECS = {
    "king died spring": [1.0, 0.0, 0.0],
    "king passed spring": [0.99, 0.01, 0.0],
    "king died spring crops failed": [0.98, 0.02, 0.0],
    "king died spring comet appeared": [0.97, 0.03, 0.0],
    "k1": [1.0, 0.0, 0.0],
    "k2": [0.99, 0.01, 0.0],
    "k3": [0.98, 0.02, 0.0],
    "ab": [1.0, 0.0, 0.0],
    "cd": [0.99, 0.01, 0.0],
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


def _manager(llm, threshold=999, roles=("alice", "bob", "carol")):
    return LRUChromaMemoryManager(
        llm=llm, embedding_fn=_embed, role_codes=list(roles),
        persist_dir=None, consensus_threshold=threshold,
        consensus_top_k=5, miss_retrieve_k=5, cache_capacity=20,
    )


def _all(m):
    """All (text -> owners) entries currently in the store."""
    entries = m.store.query_global("k1", top_k=50)
    return {t: m.store.owners_of(i) for i, t in entries}


def test_full_coverage_merges_into_single_consensus():
    llm = FakeLLM(['["The king died in spring.", "", ""]'])
    m = _manager(llm, threshold=2)
    m.add_memory("alice", "king died spring")   # pending = 1, no trigger
    m.add_memory("bob", "king passed spring")    # pending = 2 -> consensus runs

    entries = _all(m)
    assert list(entries) == ["The king died in spring."]
    assert entries["The king died in spring."] == {"alice", "bob"}
    # both originals' cache slots removed (fully covered)
    assert m.cache_ids("alice") == []
    assert m.cache_ids("bob") == []


def test_partial_coverage_keeps_residuals_with_original_owners():
    llm = FakeLLM(['["The king died in spring.", "crops failed", "comet appeared"]'])
    m = _manager(llm, threshold=2)
    m.add_memory("alice", "king died spring crops failed")
    m.add_memory("bob", "king died spring comet appeared")

    entries = _all(m)
    assert entries["The king died in spring."] == {"alice", "bob"}
    assert entries["crops failed"] == {"alice"}
    assert entries["comet appeared"] == {"bob"}

    # caches repoint from the deleted originals to the residuals
    alice_text = m.store.get(m.cache_ids("alice")[0])
    bob_text = m.store.get(m.cache_ids("bob")[0])
    assert alice_text == "crops failed"
    assert bob_text == "comet appeared"


def test_picks_n_with_minimum_total_chars():
    # n=1 output is long; n=2 output is tiny -> n=2 chosen, merging all three.
    llm = FakeLLM([
        '["a very long consensus that does not help", "resid_a", "resid_b"]',
        '["x", "", "", ""]',
    ])
    m = _manager(llm, threshold=999)
    m.add_memory("alice", "k1")
    m.add_memory("bob", "k2")
    m.add_memory("carol", "k3")

    m.run_consensus()

    entries = _all(m)
    assert list(entries) == ["x"]
    assert entries["x"] == {"alice", "bob", "carol"}


def test_safety_fallback_skips_when_no_compression():
    # consensus + residuals longer than the originals -> leave everything as is.
    llm = FakeLLM(['["xxxxx", "ab", "cd"]'])
    m = _manager(llm, threshold=999)
    a = m.add_memory("alice", "ab")
    b = m.add_memory("bob", "cd")

    m.run_consensus()

    entries = _all(m)
    assert set(entries) == {"ab", "cd"}
    assert m.cache_ids("alice") == [a]
    assert m.cache_ids("bob") == [b]


def test_residuals_do_not_retrigger_consensus():
    # After a merge produces residuals, they must not be enqueued again.
    llm = FakeLLM(['["The king died in spring.", "crops failed", "comet appeared"]'])
    m = _manager(llm, threshold=2)
    m.add_memory("alice", "king died spring crops failed")
    m.add_memory("bob", "king died spring comet appeared")

    # only the one consensus call should have happened; nothing pending re-runs
    assert len(llm.prompts) == 1
    assert m.pending_count() == 0
