# Memory Ablation Knobs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the `lru_chroma` memory mechanism configurable along three orthogonal ablation axes — consensus on/off, cache policy (lru/fifo/none), and index backend (chroma/keyword) — without touching the `consensus`/`gmemory` baselines.

**Architecture:** Refactor `LRUChromaMemoryManager` ([modules/lru_chroma_memory.py](../../../modules/lru_chroma_memory.py)) into a strategy-pluggable orchestrator. A cache interface gains a FIFO implementation and a "no cache" manager flag; a store interface gains an LLM-keyword SQLite backend alongside the existing ChromaDB one; consensus becomes a gated step. Persistence stays "in-memory store + holographic JSON sidecar" — the sidecar now stores derived data (embeddings / keywords) so loads never recompute.

**Tech Stack:** Python 3.9, `sqlite3` (stdlib), `chromadb`, `pytest`. Tests run with `venv/bin/python -m pytest` from the `BookWorld/` directory.

**Spec:** [docs/superpowers/specs/2026-06-29-memory-ablation-knobs-design.md](../specs/2026-06-29-memory-ablation-knobs-design.md)

## Global Constraints

- All work is inside the `memory_type == "lru_chroma"` pathway. Do **not** change the `consensus` or `gmemory` mechanisms.
- Defaults preserve today's behavior: `consensus_enabled=True`, `cache_strategy="lru"`, `index_backend="chroma"`. The existing 18 tests MUST stay green after every task.
- No new third-party dependency. `sqlite3` is stdlib; keyword extraction reuses the injected `llm`.
- No true on-disk persistence: store is in-memory; persistence is the JSON sidecar only.
- Store-interface method names/signatures are shared by both backends: `add(text, owners)->id`, `delete(id)`, `get(id)->str|None`, `owners_of(id)->set[str]`, `query(text, owner, top_k)->list[(id,text)]`, `query_global(text, top_k, exclude_id=None)->list[(id,text)]`, `count()->int`, `all_entries()->list[dict]`, `restore(entries)->None`.
- **This repo is not currently under git.** Each task ends with a **Checkpoint** that runs the test suite green. If you `git init`, you may additionally commit at each checkpoint; otherwise treat the checkpoint as the gate.
- Run tests from `BookWorld/`: `venv/bin/python -m pytest tests/ -q`.

---

### Task 1: FIFO cache class

**Files:**
- Modify: `modules/lru_chroma_memory.py` (add `RoleFIFOCache` after `RoleLRUCache`, ~line 57)
- Test: `tests/test_role_fifo_cache.py` (create)

**Interfaces:**
- Consumes: `RoleLRUCache` (existing) — front=MRU list, methods `ids/insert/replace/remove/contains/load`.
- Produces: `RoleFIFOCache(capacity=20)` — same interface as `RoleLRUCache`; the **only** behavioral difference is `insert` of an already-present id is a no-op (keeps arrival position). New ids go to the front; eviction drops the tail (oldest arrival).

- [ ] **Step 1: Write the failing test**

Create `tests/test_role_fifo_cache.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_role_fifo_cache.py -q`
Expected: FAIL with `ImportError: cannot import name 'RoleFIFOCache'`.

- [ ] **Step 3: Write minimal implementation**

In `modules/lru_chroma_memory.py`, immediately after the `RoleLRUCache` class (before `_owner_flag`), add:

```python
class RoleFIFOCache(RoleLRUCache):
    """Per-role FIFO cache. Same id-only storage as RoleLRUCache, but a
    re-inserted id keeps its arrival position (no refresh-on-access); only the
    oldest arrival (tail) is evicted past capacity.
    """

    def insert(self, mem_id: str):
        if mem_id in self._ids:
            return  # FIFO: do not reorder on re-access
        self._ids.insert(0, mem_id)
        if len(self._ids) > self.capacity:
            self._ids = self._ids[: self.capacity]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/python -m pytest tests/test_role_fifo_cache.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Checkpoint**

Run: `venv/bin/python -m pytest tests/ -q`
Expected: all green (22 passed). Optionally commit.

---

### Task 2: Consensus toggle

**Files:**
- Modify: `modules/lru_chroma_memory.py` — `LRUChromaMemoryManager.__init__` (~line 242), `add_memory` (~line 279), `run_consensus` (~line 289)
- Test: `tests/test_consensus_toggle.py` (create)

**Interfaces:**
- Consumes: existing `LRUChromaMemoryManager.add_memory(role, text)`, `run_consensus()`, `pending_count()`.
- Produces: `LRUChromaMemoryManager(..., consensus_enabled: bool = True)`. When `False`: `add_memory` does not append to `_pending` and never triggers consensus; `run_consensus()` returns immediately.

- [ ] **Step 1: Write the failing test**

Create `tests/test_consensus_toggle.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_consensus_toggle.py -q`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'consensus_enabled'`.

- [ ] **Step 3: Write minimal implementation**

In `LRUChromaMemoryManager.__init__`, add the parameter (extend the signature) and store it. Change the signature line and add the assignment:

```python
    def __init__(self, llm, embedding_fn, role_codes: List[str],
                 collection_name: str = "lru_chroma_memory",
                 persist_dir: Optional[str] = None,
                 cache_capacity: int = 20, miss_retrieve_k: int = 5,
                 consensus_top_k: int = 5, consensus_threshold: int = 10,
                 language: str = "zh", consensus_enabled: bool = True):
        ...
        self.consensus_enabled = consensus_enabled
```

Add the assignment right after `self.consensus_threshold = consensus_threshold`.

In `add_memory`, gate the enqueue:

```python
    def add_memory(self, role_code: str, text: str) -> str:
        mem_id = self.store.add(text, owners={role_code})
        self._cache_for(role_code).insert(mem_id)
        if self.consensus_enabled:
            self._pending.append(mem_id)
            if len(self._pending) >= self.consensus_threshold:
                self.run_consensus()
        return mem_id
```

In `run_consensus`, add an early guard as the first line of the body:

```python
    def run_consensus(self):
        if not self.consensus_enabled:
            return
        queue = self._pending
        ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/python -m pytest tests/test_consensus_toggle.py -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Checkpoint**

Run: `venv/bin/python -m pytest tests/ -q`
Expected: all green (23 passed). Optionally commit.

---

### Task 3: Cache strategy selection + `none` read flow

**Files:**
- Modify: `modules/lru_chroma_memory.py` — `__init__` (cache construction), `retrieve_for_role` (~line 260), `add_memory` (~line 279), `current_cache_text` (~line 368), `_cache_for` (~line 374)
- Test: `tests/test_cache_strategy.py` (create)

**Interfaces:**
- Consumes: `RoleLRUCache`, `RoleFIFOCache` (Task 1), existing read/write flow.
- Produces: `LRUChromaMemoryManager(..., cache_strategy: str = "lru")`. `cache_strategy` ∈ `{"lru","fifo","none"}`. Sets `self.cache_enabled = (cache_strategy != "none")`. When `"none"`: no caches are built, `retrieve_for_role` goes straight to `store.query` (no judge LLM), `add_memory` does no cache insert, `current_cache_text` returns `""`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_cache_strategy.py`:

```python
"""Tests for the cache_strategy knob (lru / fifo / none)."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.lru_chroma_memory import (
    LRUChromaMemoryManager, RoleLRUCache, RoleFIFOCache,
)

_VECS = {
    "a storm wrecked the ship": [0.99, 0.01, 0.0],
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
        persist_dir=None, consensus_threshold=999, miss_retrieve_k=5, **kw,
    )


def test_fifo_strategy_uses_fifo_cache():
    m = _manager(FakeLLM(), cache_strategy="fifo")
    assert isinstance(m._cache_for("alice"), RoleFIFOCache)


def test_lru_strategy_is_default():
    m = _manager(FakeLLM())
    assert isinstance(m._cache_for("alice"), RoleLRUCache)


def test_none_strategy_goes_straight_to_store_no_judge():
    llm = FakeLLM()  # no responses; judge must NOT be called
    m = _manager(llm, cache_strategy="none")
    m.add_memory("alice", "a storm wrecked the ship")

    res = m.retrieve_for_role("alice", "storm")

    assert "a storm wrecked the ship" in res
    assert llm.prompts == []              # no cache => no sufficiency judge
    assert m.cache_ids("alice") == []     # no cache state retained
    assert m.current_cache_text("alice") == ""


def test_none_strategy_respects_owner_filter():
    m = _manager(FakeLLM(), cache_strategy="none")
    m.add_memory("bob", "a storm wrecked the ship")  # owned by bob

    res = m.retrieve_for_role("alice", "storm")
    assert "a storm wrecked the ship" not in res
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_cache_strategy.py -q`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'cache_strategy'`.

- [ ] **Step 3: Write minimal implementation**

Extend the `__init__` signature with `cache_strategy: str = "lru"` (add after `consensus_enabled`). Replace the cache-construction line. The current line is:

```python
        self._caches = {rc: RoleLRUCache(cache_capacity) for rc in role_codes}
```

Replace with:

```python
        self.cache_strategy = cache_strategy
        self.cache_enabled = cache_strategy != "none"
        cache_cls = RoleFIFOCache if cache_strategy == "fifo" else RoleLRUCache
        self._cache_cls = cache_cls
        self._caches = (
            {rc: cache_cls(cache_capacity) for rc in role_codes}
            if self.cache_enabled else {}
        )
```

Update `_cache_for` to use the selected class:

```python
    def _cache_for(self, role_code: str) -> RoleLRUCache:
        if role_code not in self._caches:
            self._caches[role_code] = self._cache_cls(self.cache_capacity)
        return self._caches[role_code]
```

Add the no-cache branch at the very top of `retrieve_for_role`:

```python
    def retrieve_for_role(self, role_code: str, query: str) -> str:
        if not self.cache_enabled:
            hits = self.store.query(query, owner=role_code,
                                    top_k=self.miss_retrieve_k)
            return self._format([t for _, t in hits])

        cache = self._cache_for(role_code)
        ...  # unchanged below
```

Gate the cache insert in `add_memory`:

```python
    def add_memory(self, role_code: str, text: str) -> str:
        mem_id = self.store.add(text, owners={role_code})
        if self.cache_enabled:
            self._cache_for(role_code).insert(mem_id)
        if self.consensus_enabled:
            self._pending.append(mem_id)
            if len(self._pending) >= self.consensus_threshold:
                self.run_consensus()
        return mem_id
```

Guard `current_cache_text` for the no-cache case (add as first line of the body):

```python
    def current_cache_text(self, role_code: str) -> str:
        if not self.cache_enabled:
            return ""
        texts = [t for t in (self.store.get(i)
                             for i in self._cache_for(role_code).ids()) if t]
        return self._format(texts)
```

(The consensus cache-repointing loop already iterates `self._caches.values()`, which is `{}` when cache is disabled, so it is a safe no-op — no change needed there.)

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/python -m pytest tests/test_cache_strategy.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Checkpoint**

Run: `venv/bin/python -m pytest tests/ -q`
Expected: all green (27 passed). Optionally commit.

---

### Task 4: KeywordSQLiteStore (no-embedding index backend)

**Files:**
- Modify: `modules/lru_chroma_memory.py` — add `KeywordSQLiteStore` class (after `SharedChromaStore`, ~line 177) and a module-level `make_llm_keyword_fn` helper (near the prompts)
- Test: `tests/test_keyword_sqlite_store.py` (create)

**Interfaces:**
- Consumes: a `keyword_fn(text: str) -> list[str]` injected at construction; module helper `_parse_json_list` (existing).
- Produces:
  - `KeywordSQLiteStore(keyword_fn, db_path=":memory:")` implementing the full store interface (see Global Constraints). `add` accepts optional `mem_id`/`keywords` for restore. `all_entries()` returns `{id, text, owners, keywords}`. Ranking is by count of shared keywords, descending, ties by insertion order.
  - `make_llm_keyword_fn(llm, top_k=5, language="zh") -> (str -> list[str])` — builds a keyword extractor over `llm.chat`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_keyword_sqlite_store.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_keyword_sqlite_store.py -q`
Expected: FAIL with `ImportError: cannot import name 'KeywordSQLiteStore'`.

- [ ] **Step 3: Write minimal implementation**

Add the keyword-extraction prompt and helper near the other prompts in `modules/lru_chroma_memory.py`:

```python
KEYWORD_PROMPT = """Extract up to {k} short, representative keywords from the text \
below (single words or very short phrases). Cover the key entities, actions, and topics.

## Text:
{text}

Return ONLY a JSON array of strings parsable by json.loads(). No code fences.
"""


def make_llm_keyword_fn(llm, top_k: int = 5, language: str = "zh"):
    """Build a keyword_fn(text)->list[str] backed by the LLM."""

    def keyword_fn(text: str) -> List[str]:
        if not text or not text.strip():
            return []
        prompt = KEYWORD_PROMPT.format(k=top_k, text=text)
        try:
            response = llm.chat(prompt)
        except Exception:
            return []
        parsed = _parse_json_list(response) or []
        out, seen = [], set()
        for kw in parsed:
            norm = kw.strip().lower()
            if norm and norm not in seen:
                seen.add(norm)
                out.append(norm)
        return out[:top_k]

    return keyword_fn
```

Add the store class after `SharedChromaStore`:

```python
class KeywordSQLiteStore:
    """A no-embedding global memory store backed by in-process SQLite.

    Similarity = count of shared keywords (extracted via *keyword_fn*). Mirrors
    SharedChromaStore's interface so consensus + the manager work unchanged.
    """

    def __init__(self, keyword_fn, db_path: str = ":memory:"):
        import sqlite3

        self._keyword_fn = keyword_fn
        self._conn = sqlite3.connect(db_path)
        c = self._conn
        c.execute("CREATE TABLE memories (id TEXT PRIMARY KEY, text TEXT)")
        c.execute("CREATE TABLE owners (mem_id TEXT, role TEXT)")
        c.execute("CREATE TABLE keywords (mem_id TEXT, keyword TEXT)")
        c.execute("CREATE INDEX idx_owners_role ON owners(role)")
        c.execute("CREATE INDEX idx_owners_mem ON owners(mem_id)")
        c.execute("CREATE INDEX idx_kw_keyword ON keywords(keyword)")
        c.execute("CREATE INDEX idx_kw_mem ON keywords(mem_id)")
        c.commit()

    # ---- write ----

    def add(self, text: str, owners: Set[str], mem_id: Optional[str] = None,
            keywords: Optional[List[str]] = None) -> str:
        mem_id = mem_id or str(uuid.uuid4())
        kws = keywords if keywords is not None else self._keyword_fn(text)
        c = self._conn
        c.execute("INSERT INTO memories VALUES (?, ?)", (mem_id, text))
        c.executemany("INSERT INTO owners VALUES (?, ?)",
                      [(mem_id, o) for o in owners])
        c.executemany("INSERT INTO keywords VALUES (?, ?)",
                      [(mem_id, k) for k in kws])
        c.commit()
        return mem_id

    def delete(self, mem_id: str):
        c = self._conn
        c.execute("DELETE FROM memories WHERE id = ?", (mem_id,))
        c.execute("DELETE FROM owners WHERE mem_id = ?", (mem_id,))
        c.execute("DELETE FROM keywords WHERE mem_id = ?", (mem_id,))
        c.commit()

    # ---- read ----

    def get(self, mem_id: str) -> Optional[str]:
        row = self._conn.execute(
            "SELECT text FROM memories WHERE id = ?", (mem_id,)).fetchone()
        return row[0] if row else None

    def owners_of(self, mem_id: str) -> Set[str]:
        rows = self._conn.execute(
            "SELECT role FROM owners WHERE mem_id = ?", (mem_id,)).fetchall()
        return {r[0] for r in rows}

    def query(self, text: str, owner: str, top_k: int) -> List[Tuple[str, str]]:
        return self._match(text, top_k, owner=owner, exclude_id=None)

    def query_global(self, text: str, top_k: int,
                     exclude_id: Optional[str] = None) -> List[Tuple[str, str]]:
        return self._match(text, top_k, owner=None, exclude_id=exclude_id)

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    def all_entries(self) -> List[dict]:
        out = []
        for mem_id, text in self._conn.execute(
                "SELECT id, text FROM memories").fetchall():
            kws = [r[0] for r in self._conn.execute(
                "SELECT keyword FROM keywords WHERE mem_id = ?", (mem_id,))]
            out.append({"id": mem_id, "text": text,
                        "owners": sorted(self.owners_of(mem_id)),
                        "keywords": kws})
        return out

    def restore(self, entries: List[dict]):
        for e in entries:
            self.add(e["text"], set(e.get("owners", [])),
                     mem_id=e["id"], keywords=e.get("keywords"))

    # ---- internal ----

    def _match(self, text: str, top_k: int, owner: Optional[str],
               exclude_id: Optional[str]) -> List[Tuple[str, str]]:
        if top_k < 1:
            return []
        kws = self._keyword_fn(text)
        if not kws:
            return []
        placeholders = ",".join("?" for _ in kws)
        owner_join = "JOIN owners o ON o.mem_id = k.mem_id" if owner else ""
        owner_where = "AND o.role = ?" if owner else ""
        exclude_where = "AND k.mem_id != ?" if exclude_id else ""
        sql = (
            "SELECT m.id, m.text, COUNT(DISTINCT k.keyword) AS score "
            "FROM keywords k JOIN memories m ON m.id = k.mem_id "
            f"{owner_join} "
            f"WHERE k.keyword IN ({placeholders}) {owner_where} {exclude_where} "
            "GROUP BY m.id ORDER BY score DESC, m.rowid ASC LIMIT ?"
        )
        params = list(kws)
        if owner:
            params.append(owner)
        if exclude_id:
            params.append(exclude_id)
        params.append(top_k)
        rows = self._conn.execute(sql, params).fetchall()
        return [(r[0], r[1]) for r in rows]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/python -m pytest tests/test_keyword_sqlite_store.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Checkpoint**

Run: `venv/bin/python -m pytest tests/ -q`
Expected: all green (33 passed). Optionally commit.

---

### Task 5: index_backend selection in the manager

**Files:**
- Modify: `modules/lru_chroma_memory.py` — `LRUChromaMemoryManager.__init__` (store construction, ~line 249)
- Test: `tests/test_index_backend.py` (create)

**Interfaces:**
- Consumes: `SharedChromaStore`, `KeywordSQLiteStore`, `make_llm_keyword_fn` (Task 4).
- Produces: `LRUChromaMemoryManager(..., index_backend: str = "chroma", keyword_top_k: int = 5)`. When `index_backend == "keyword"`, `self.store` is a `KeywordSQLiteStore` built with `make_llm_keyword_fn(llm, keyword_top_k, language)`; otherwise the existing `SharedChromaStore`. Retrieval + consensus run over either, unchanged.

- [ ] **Step 1: Write the failing test**

Create `tests/test_index_backend.py`:

```python
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
    return LRUChromaMemoryManager(
        llm=KwLLM(), embedding_fn=_embed, role_codes=["alice", "bob"],
        persist_dir=None, consensus_threshold=999, miss_retrieve_k=5, **kw,
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_index_backend.py -q`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'index_backend'`.

- [ ] **Step 3: Write minimal implementation**

Extend the `__init__` signature with `index_backend: str = "chroma"` and `keyword_top_k: int = 5` (after `cache_strategy`). Replace the store-construction line. The current line is:

```python
        self.store = SharedChromaStore(embedding_fn, collection_name, persist_dir)
```

Replace with:

```python
        self.index_backend = index_backend
        if index_backend == "keyword":
            self.keyword_fn = make_llm_keyword_fn(llm, keyword_top_k, language)
            self.store = KeywordSQLiteStore(self.keyword_fn)
        else:
            self.keyword_fn = None
            self.store = SharedChromaStore(embedding_fn, collection_name,
                                           persist_dir)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/python -m pytest tests/test_index_backend.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Checkpoint**

Run: `venv/bin/python -m pytest tests/ -q`
Expected: all green (37 passed). Optionally commit.

---

### Task 6: Holographic chroma sidecar (store embeddings, no recompute on load)

**Files:**
- Modify: `modules/lru_chroma_memory.py` — `SharedChromaStore.all_entries` (~line 133) and `SharedChromaStore.restore` (~line 146)
- Test: `tests/test_holographic_sidecar.py` (create)

**Interfaces:**
- Consumes: existing `SharedChromaStore` / `LRUChromaMemoryManager.save_to_file` / `load_from_file`.
- Produces: chroma `all_entries()` now includes an `embedding` field (list of float); `restore()` reuses it and only recomputes via `embedding_fn` when `embedding` is absent (back-compat). (KeywordSQLiteStore already persists `keywords` from Task 4.)

- [ ] **Step 1: Write the failing test**

Create `tests/test_holographic_sidecar.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_holographic_sidecar.py -q`
Expected: FAIL on `test_chroma_sidecar_stores_embedding_and_skips_recompute` — either `entries[0].get("embedding")` is `None`, or `calls["n"]` increased on load.

- [ ] **Step 3: Write minimal implementation**

In `SharedChromaStore.all_entries`, include embeddings. Replace the current method body with:

```python
    def all_entries(self) -> List[dict]:
        """Export every entry as {id, text, owners, embedding}."""
        res = self._collection.get(include=["documents", "metadatas", "embeddings"])
        ids = res.get("ids") or []
        docs = res.get("documents") or []
        metas = res.get("metadatas") or []
        embs = res.get("embeddings")
        embs = embs if embs is not None else []
        out = []
        for i, mem_id in enumerate(ids):
            meta = metas[i] if i < len(metas) else {}
            owners = json.loads(meta.get("owners", "[]")) if meta else []
            emb = embs[i] if i < len(embs) else None
            entry = {"id": mem_id, "text": docs[i], "owners": owners}
            if emb is not None:
                entry["embedding"] = [float(x) for x in emb]
            out.append(entry)
        return out
```

In `SharedChromaStore.restore`, reuse the saved embedding when present:

```python
    def restore(self, entries: List[dict]):
        """Re-import entries (preserving ids); reuse stored embeddings, else recompute."""
        for e in entries:
            text, owners = e["text"], set(e.get("owners", []))
            emb = e.get("embedding")
            if emb is None:
                emb = self._embedding_fn([text])[0]
            self._collection.add(
                ids=[e["id"]], embeddings=[emb], documents=[text],
                metadatas=[self._owner_metadata(owners)],
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/python -m pytest tests/test_holographic_sidecar.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Checkpoint**

Run: `venv/bin/python -m pytest tests/ -q`
Expected: all green (39 passed). The existing `test_lru_chroma_persistence.py` still passes (reused embeddings still answer the query). Optionally commit.

---

### Task 7: Wire the three knobs through BookWorld config

**Files:**
- Modify: `BookWorld.py:164-172` (the `LRUChromaMemoryManager(...)` construction inside `init_world_agent_from_file`)
- Test: `tests/test_bookworld_memory_config.py` (create) — a focused unit test of the config-to-manager mapping (no full simulation).

**Interfaces:**
- Consumes: `self.config` (already loaded at `BookWorld.py:52`), `LRUChromaMemoryManager` (Tasks 2/3/5 params).
- Produces: the manager is constructed with `consensus_enabled`, `cache_strategy`, `index_backend` read from `config` with defaults `True` / `"lru"` / `"chroma"`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_bookworld_memory_config.py`:

```python
"""The three ablation knobs read from config reach LRUChromaMemoryManager."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.lru_chroma_memory import LRUChromaMemoryManager


def _build(config):
    """Mirror BookWorld's construction of the lru_chroma manager from config."""
    return LRUChromaMemoryManager(
        llm=_NullLLM(), embedding_fn=lambda ts: [[0.0, 0.0, 1.0] for _ in ts],
        role_codes=["alice"], consensus_threshold=10, miss_retrieve_k=5,
        cache_capacity=20, consensus_top_k=5, language="zh",
        consensus_enabled=config.get("consensus_enabled", True),
        cache_strategy=config.get("cache_strategy", "lru"),
        index_backend=config.get("index_backend", "chroma"),
    )


class _NullLLM:
    def chat(self, prompt):
        return ""


def test_defaults_preserve_current_behavior():
    m = _build({})
    assert m.consensus_enabled is True
    assert m.cache_strategy == "lru"
    assert m.index_backend == "chroma"


def test_config_overrides_all_three_knobs():
    m = _build({"consensus_enabled": False,
                "cache_strategy": "fifo",
                "index_backend": "keyword"})
    assert m.consensus_enabled is False
    assert m.cache_strategy == "fifo"
    assert m.index_backend == "keyword"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_bookworld_memory_config.py -q`
Expected: PASS already for `_build` (it constructs the manager directly) — but it asserts the manager exposes `consensus_enabled`, `cache_strategy`, `index_backend` attributes set in Tasks 2/3/5. If any attribute is missing, this fails. (If Tasks 2/3/5 are complete, this passes; the real deliverable of this task is the `BookWorld.py` edit, verified in Step 4.)

- [ ] **Step 3: Modify `BookWorld.py`**

Replace the manager construction at `BookWorld.py:164-173`:

```python
            self.lru_memory = LRUChromaMemoryManager(
                llm=self.world_llm if hasattr(self, 'world_llm') else llm,
                embedding_fn=embedding,
                role_codes=list(self.role_agents.keys()),
                consensus_threshold=self.consensus_threshold,
                miss_retrieve_k=5,
                cache_capacity=20,
                consensus_top_k=5,
                language=self.language,
                consensus_enabled=self.config.get("consensus_enabled", True),
                cache_strategy=self.config.get("cache_strategy", "lru"),
                index_backend=self.config.get("index_backend", "chroma"),
            )
```

- [ ] **Step 4: Verify the wiring**

Run: `venv/bin/python -m pytest tests/test_bookworld_memory_config.py -q`
Expected: PASS (2 passed).

Then confirm the edit compiles and reads config:

Run: `venv/bin/python -c "import ast; ast.parse(open('BookWorld.py').read()); print('ok')"`
Expected: `ok`.

- [ ] **Step 5: Checkpoint**

Run: `venv/bin/python -m pytest tests/ -q`
Expected: all green (41 passed). Optionally commit.

---

## Final verification

- [ ] Run the whole suite: `venv/bin/python -m pytest tests/ -q` → all green.
- [ ] Spot-check an ablation combo end to end by config: set `"index_backend": "keyword"`, `"cache_strategy": "none"`, `"consensus_enabled": false` in a test config and confirm BookWorld initializes (manual smoke; real LLM not required for construction).
- [ ] Update `config.json` (or the experiment configs) with the three keys documented, defaulting to current behavior, so the ablation grid is reproducible.
