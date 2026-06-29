"""Tests for RoleLRUCache — a per-role LRU cache that stores only memory ids."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.lru_chroma_memory import RoleLRUCache


def test_insert_new_id_goes_to_front():
    cache = RoleLRUCache(capacity=3)
    cache.insert("a")
    cache.insert("b")
    assert cache.ids() == ["b", "a"]


def test_insert_existing_id_moves_to_front():
    cache = RoleLRUCache(capacity=3)
    cache.insert("a")
    cache.insert("b")
    cache.insert("a")
    assert cache.ids() == ["a", "b"]


def test_eviction_drops_least_recently_used_tail():
    cache = RoleLRUCache(capacity=2)
    cache.insert("a")
    cache.insert("b")
    cache.insert("c")  # evicts "a"
    assert cache.ids() == ["c", "b"]
    assert not cache.contains("a")


def test_replace_preserves_position():
    cache = RoleLRUCache(capacity=3)
    cache.insert("a")
    cache.insert("b")
    cache.insert("c")  # ["c", "b", "a"]
    cache.replace("b", "B")
    assert cache.ids() == ["c", "B", "a"]


def test_replace_missing_id_is_noop():
    cache = RoleLRUCache(capacity=3)
    cache.insert("a")
    cache.replace("zzz", "B")
    assert cache.ids() == ["a"]


def test_remove_deletes_id():
    cache = RoleLRUCache(capacity=3)
    cache.insert("a")
    cache.insert("b")
    cache.remove("a")
    assert cache.ids() == ["b"]
    assert not cache.contains("a")


def test_contains_reports_membership():
    cache = RoleLRUCache(capacity=3)
    cache.insert("a")
    assert cache.contains("a")
    assert not cache.contains("b")
