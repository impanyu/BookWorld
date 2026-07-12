"""Tests for RoleFIFOCache — first-in-first-out per-role id cache."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.lru_chroma_memory import RoleFIFOCache


def test_insert_new_id_goes_to_front():
    cache = RoleFIFOCache(capacity=3)
    cache.insert("a")
    cache.insert("b")
    assert cache.ids() == ["b", "a"]


def test_reinsert_existing_id_does_not_reorder():
    cache = RoleFIFOCache(capacity=3)
    cache.insert("a")
    cache.insert("b")
    cache.insert("a")  # FIFO: no refresh on re-access
    assert cache.ids() == ["b", "a"]


def test_eviction_drops_oldest_arrival_tail():
    cache = RoleFIFOCache(capacity=2)
    cache.insert("a")
    cache.insert("b")
    cache.insert("c")  # evicts "a" (oldest)
    assert cache.ids() == ["c", "b"]
    assert not cache.contains("a")


def test_replace_and_remove_match_lru():
    cache = RoleFIFOCache(capacity=3)
    cache.insert("a")
    cache.insert("b")
    cache.insert("c")  # ["c", "b", "a"]
    cache.replace("b", "B")
    assert cache.ids() == ["c", "B", "a"]
    cache.remove("c")
    assert cache.ids() == ["B", "a"]
