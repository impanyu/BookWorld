"""Holographic sidecar: load reuses stored embeddings/keywords, no recompute."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.lru_chroma_memory import LRUChromaMemoryManager


_VECS = {
    "a storm wrecked the ship": [1.0, 0.0, 0.0],
    "storm": [1.0, 0.0, 0.0],
}


def test_chroma_sidecar_stores_embedding_and_skips_recompute(tmp_path):
    calls = {"n": 0}

    def counting_embed(texts):
        calls["n"] += len(texts)
        return [_VECS.get(t, [0.0, 0.0, 1.0]) for t in texts]

    m = LRUChromaMemoryManager(
        llm=None, embedding_fn=counting_embed, role_codes=["alice"],
        persist_dir=None, consensus_threshold=999)
    a = m.add_memory("alice", "a storm wrecked the ship")
    m.save_to_file(str(tmp_path))

    entries = m.store.all_entries()
    assert entries[0].get("embedding")            # embedding persisted

    after_add = calls["n"]
    m2 = LRUChromaMemoryManager(
        llm=None, embedding_fn=counting_embed, role_codes=["alice"],
        persist_dir=None, consensus_threshold=999)
    m2.load_from_file(str(tmp_path))

    assert calls["n"] == after_add                # no recompute on load
    assert m2.store.get(a) == "a storm wrecked the ship"


def test_chroma_restore_recomputes_when_embedding_absent(tmp_path):
    calls = {"n": 0}

    def counting_embed(texts):
        calls["n"] += len(texts)
        return [_VECS.get(t, [0.0, 0.0, 1.0]) for t in texts]

    m = LRUChromaMemoryManager(
        llm=None, embedding_fn=counting_embed, role_codes=["alice"],
        persist_dir=None, consensus_threshold=999)
    m.store.add("a storm wrecked the ship", {"alice"})
    legacy = [{"id": "x1", "text": "a storm wrecked the ship",
               "owners": ["alice"]}]  # old sidecar: no embedding field

    before = calls["n"]
    m.store.restore(legacy)
    assert calls["n"] == before + 1               # recompute fallback fired
