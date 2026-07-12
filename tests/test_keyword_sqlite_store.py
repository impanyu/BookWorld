"""Tests for KeywordSQLiteStore — no-embedding keyword-match index backend."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.lru_chroma_memory import KeywordSQLiteStore


# Deterministic stand-in for the LLM keyword extractor: split on spaces.
def _kw(text):
    return sorted(set(text.lower().split()))


def test_add_get_owners_delete():
    s = KeywordSQLiteStore(_kw)
    i = s.add("storm wrecked ship", {"alice"})
    assert s.get(i) == "storm wrecked ship"
    assert s.owners_of(i) == {"alice"}
    assert s.count() == 1
    s.delete(i)
    assert s.get(i) is None
    assert s.count() == 0


def test_query_ranks_by_keyword_overlap_and_filters_owner():
    s = KeywordSQLiteStore(_kw)
    a = s.add("storm wrecked the ship", {"alice"})
    s.add("storm wrecked the ship", {"bob"})        # bob's, filtered out
    feast = s.add("a feast in the great hall", {"alice"})

    hits = s.query("storm ship", owner="alice", top_k=5)
    ids = [i for i, _ in hits]
    assert ids[0] == a                              # highest overlap first
    assert all(s.owners_of(i) == {"alice"} for i in ids)  # owner filter held
    if feast in ids:
        assert ids.index(a) < ids.index(feast)      # storm ranks above feast


def test_query_owner_filter_excludes_other_owner():
    s = KeywordSQLiteStore(_kw)
    s.add("storm wrecked the ship", {"bob"})
    hits = s.query("storm ship", owner="alice", top_k=5)
    assert hits == []                               # alice owns nothing matching


def test_query_global_excludes_self():
    s = KeywordSQLiteStore(_kw)
    m = s.add("storm wrecked the ship", {"alice"})
    n = s.add("storm at sea", {"bob"})
    hits = s.query_global("storm wrecked the ship", top_k=5, exclude_id=m)
    ids = [i for i, _ in hits]
    assert m not in ids
    assert n in ids


def test_empty_keywords_returns_no_hits():
    s = KeywordSQLiteStore(lambda text: [])          # extractor yields nothing
    s.add("storm wrecked the ship", {"alice"})
    assert s.query("storm", owner="alice", top_k=5) == []


def test_all_entries_and_restore_roundtrip_uses_saved_keywords():
    s = KeywordSQLiteStore(_kw)
    i = s.add("storm wrecked the ship", {"alice"})
    entries = s.all_entries()
    assert entries[0]["keywords"]                    # keywords persisted

    calls = {"n": 0}

    def counting_kw(text):
        calls["n"] += 1
        return _kw(text)

    s2 = KeywordSQLiteStore(counting_kw)
    s2.restore(entries)                              # should NOT recompute
    assert calls["n"] == 0
    assert s2.get(i) == "storm wrecked the ship"
    assert s2.owners_of(i) == {"alice"}
    assert [i2 for i2, _ in s2.query("storm ship", owner="alice", top_k=5)] == [i]
