"""Tests for LRUChromaMemoryManager persistence (save/load roundtrip)."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.lru_chroma_memory import LRUChromaMemoryManager


_VECS = {
    "a storm wrecked the ship": [1.0, 0.0, 0.0],
    "a feast in the great hall": [0.0, 1.0, 0.0],
    "storm": [1.0, 0.0, 0.0],
}


def _embed(texts):
    return [_VECS.get(t, [0.0, 0.0, 1.0]) for t in texts]


class FakeLLM:
    def chat(self, prompt):
        return ""


def _manager():
    return LRUChromaMemoryManager(
        llm=FakeLLM(), embedding_fn=_embed, role_codes=["alice", "bob"],
        persist_dir=None, consensus_threshold=999,
    )


def test_save_and_load_roundtrip(tmp_path):
    m = _manager()
    a = m.add_memory("alice", "a storm wrecked the ship")
    b = m.add_memory("bob", "a feast in the great hall")
    m.save_to_file(str(tmp_path))

    m2 = _manager()
    m2.load_from_file(str(tmp_path))

    # store content + ownership preserved under the same ids
    assert m2.store.get(a) == "a storm wrecked the ship"
    assert m2.store.owners_of(a) == {"alice"}
    assert m2.store.get(b) == "a feast in the great hall"
    # caches and pending preserved
    assert m2.cache_ids("alice") == [a]
    assert m2.cache_ids("bob") == [b]
    assert m2.pending_count() == 2
    # embeddings rebuilt -> owner-filtered semantic search works again
    hits = [t for _, t in m2.store.query("storm", owner="alice", top_k=5)]
    assert "a storm wrecked the ship" in hits
