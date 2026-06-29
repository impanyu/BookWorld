"""Tests for SharedChromaStore — a single global vector store with owner sets.

Uses an injected deterministic fake embedding so nearest-neighbor ordering is
fully controlled, and an in-memory ChromaDB client (no persistence).
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.lru_chroma_memory import SharedChromaStore


# Deterministic fake embedding: known strings map to fixed vectors.
_VECS = {
    "storm": [1.0, 0.0, 0.0],
    "a storm wrecked the ship": [0.95, 0.05, 0.0],
    "thunder rolled at sea": [0.85, 0.15, 0.0],
    "a feast in the great hall": [0.0, 1.0, 0.0],
    "wine flowed at the banquet": [0.0, 0.95, 0.05],
}


def _fake_embed(texts):
    return [_VECS.get(t, [0.0, 0.0, 1.0]) for t in texts]


def _store():
    # persist_dir=None -> ephemeral in-memory client
    return SharedChromaStore(embedding_fn=_fake_embed, collection_name="test_mem",
                             persist_dir=None)


def test_add_and_get_roundtrip():
    s = _store()
    mid = s.add("a storm wrecked the ship", owners={"alice"})
    assert isinstance(mid, str)
    assert s.get(mid) == "a storm wrecked the ship"


def test_owners_of_returns_owner_set():
    s = _store()
    mid = s.add("a feast in the great hall", owners={"alice", "bob"})
    assert s.owners_of(mid) == {"alice", "bob"}


def test_query_filters_by_owner():
    s = _store()
    s.add("a storm wrecked the ship", owners={"alice"})
    s.add("a feast in the great hall", owners={"bob"})

    alice_hits = [t for _, t in s.query("storm", owner="alice", top_k=5)]
    assert "a storm wrecked the ship" in alice_hits
    assert "a feast in the great hall" not in alice_hits

    # bob does not own the storm memory -> it must not surface for him
    bob_hits = [t for _, t in s.query("storm", owner="bob", top_k=5)]
    assert "a storm wrecked the ship" not in bob_hits


def test_query_orders_by_similarity():
    s = _store()
    s.add("a storm wrecked the ship", owners={"alice"})   # closest to "storm"
    s.add("thunder rolled at sea", owners={"alice"})       # second
    s.add("a feast in the great hall", owners={"alice"})   # far

    hits = [t for _, t in s.query("storm", owner="alice", top_k=2)]
    assert hits == ["a storm wrecked the ship", "thunder rolled at sea"]


def test_query_global_ignores_owner_and_excludes_self():
    s = _store()
    self_id = s.add("storm", owners={"alice"})
    s.add("a storm wrecked the ship", owners={"bob"})
    s.add("thunder rolled at sea", owners={"carol"})

    hits = [t for _, t in s.query_global("storm", top_k=5, exclude_id=self_id)]
    assert "storm" not in hits  # self excluded
    # owned by different roles, still returned (no filter)
    assert "a storm wrecked the ship" in hits
    assert "thunder rolled at sea" in hits


def test_delete_removes_entry():
    s = _store()
    mid = s.add("a storm wrecked the ship", owners={"alice"})
    s.delete(mid)
    assert s.get(mid) is None
    hits = [t for _, t in s.query("storm", owner="alice", top_k=5)]
    assert "a storm wrecked the ship" not in hits
