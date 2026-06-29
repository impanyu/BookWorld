"""Tests for LRUChromaMemoryManager — read flow (cache hit / miss / empty)."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.lru_chroma_memory import LRUChromaMemoryManager


_VECS = {
    "a storm wrecked the ship": [0.99, 0.01, 0.0],
    "thunder rolled at sea": [0.90, 0.10, 0.0],
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
        persist_dir=None, consensus_threshold=999,  # disable consensus here
        miss_retrieve_k=5, cache_capacity=20, **kw,
    )


def test_empty_cache_is_immediate_miss_no_judge_call():
    llm = FakeLLM()  # no responses; judge must NOT be called
    m = _manager(llm)
    # seed the store directly (does not touch the cache)
    m.store.add("a storm wrecked the ship", {"alice"})

    res = m.retrieve_for_role("alice", "storm")

    assert "a storm wrecked the ship" in res
    assert llm.prompts == []  # empty cache short-circuits the LLM judge
    assert m.cache_ids("alice")  # cache now populated


def test_hit_returns_cache_and_does_not_query_store():
    llm = FakeLLM(["yes"])  # judge says sufficient
    m = _manager(llm)
    m.add_memory("alice", "a feast in the great hall")  # in alice cache + store
    m.store.add("a storm wrecked the ship", {"alice"})  # in store, NOT in cache

    res = m.retrieve_for_role("alice", "storm")

    assert "a feast in the great hall" in res
    assert "a storm wrecked the ship" not in res  # hit -> store not consulted
    assert len(llm.prompts) == 1


def test_miss_pulls_owner_filtered_and_updates_cache_in_reverse_similarity():
    llm = FakeLLM(["no"])  # judge says insufficient
    m = _manager(llm)
    m.add_memory("alice", "a feast in the great hall")  # cache: [feast]
    storm1 = m.store.add("a storm wrecked the ship", {"alice"})  # closest
    storm2 = m.store.add("thunder rolled at sea", {"alice"})     # second

    res = m.retrieve_for_role("alice", "storm")

    # most-similar ends at the front; the pre-existing feast sinks toward the tail
    ids = m.cache_ids("alice")
    assert ids[0] == storm1
    assert ids[1] == storm2
    assert "a storm wrecked the ship" in res
    assert "thunder rolled at sea" in res
    assert "a feast in the great hall" in res


def test_current_cache_text_returns_cache_without_llm_or_store_query():
    llm = FakeLLM()  # must not be called
    m = _manager(llm)
    m.add_memory("alice", "a feast in the great hall")
    m.store.add("a storm wrecked the ship", {"alice"})  # in store, not cache

    res = m.current_cache_text("alice")

    assert "a feast in the great hall" in res
    assert "a storm wrecked the ship" not in res
    assert llm.prompts == []


def test_miss_respects_owner_filter():
    llm = FakeLLM(["no"])
    m = _manager(llm)
    m.add_memory("alice", "a feast in the great hall")  # alice cache
    m.store.add("a storm wrecked the ship", {"bob"})    # owned by bob only

    res = m.retrieve_for_role("alice", "storm")

    assert "a storm wrecked the ship" not in res  # alice cannot see bob's memory
