# Memory Ablation Knobs — Design Spec

**Date:** 2026-06-29
**Status:** Draft (pending user review)
**Scope:** Make the `memory_type = "lru_chroma"` memory mechanism configurable along three
orthogonal axes for an ablation study, by refactoring `LRUChromaMemoryManager`
([modules/lru_chroma_memory.py](../../../modules/lru_chroma_memory.py)) into a strategy-pluggable
orchestrator. Builds on [2026-06-29-lru-chroma-memory-design.md](2026-06-29-lru-chroma-memory-design.md).

---

## 1. Motivation

The `lru_chroma` mechanism couples three independent design choices into one fixed pipeline:
a **consensus** merge step, an **LRU cache**, and a **ChromaDB embedding index**. To ablate
each contribution we expose three orthogonal switches. The `"consensus"` and `"gmemory"`
baselines are untouched; all changes live in the `lru_chroma` pathway.

---

## 2. The Three Knobs (config)

New optional keys in `config.json`, threaded through `BookWorld.__init__`
([BookWorld.py:162](../../../BookWorld.py)) into `LRUChromaMemoryManager`:

| Config key | Values | Default | Effect |
|---|---|---|---|
| `consensus_enabled` | `true` / `false` | `true` | Toggle the global consensus merge. |
| `cache_strategy` | `"lru"` / `"fifo"` / `"none"` | `"lru"` | Per-role cache policy (or no cache). |
| `index_backend` | `"chroma"` / `"keyword"` | `"chroma"` | Memory index: embedding similarity vs. LLM-keyword matching. |

All three are independent and compose freely (e.g. `cache_strategy="none"`,
`index_backend="keyword"`, `consensus_enabled=false`). The class name
`LRUChromaMemoryManager` is retained to minimize churn; it now means "the modular
cache + shared-store + optional-consensus manager".

---

## 3. Knob 1 — Consensus toggle

`LRUChromaMemoryManager(..., consensus_enabled: bool = True)`.

- When `True`: unchanged. `add_memory` enqueues into `pending`; reaching
  `consensus_threshold` triggers `run_consensus()`.
- When `False`: `add_memory` does **not** append to `pending` and never triggers consensus;
  `run_consensus()` is a no-op. Store + cache writes are unchanged. The `pending` list stays
  empty.

No other code path changes. Persistence still serializes `pending` (empty when disabled).

---

## 4. Knob 2 — Cache strategy (cache interface)

Introduce a minimal cache interface; the existing `RoleLRUCache` is one implementation.
All caches expose: `ids()`, `insert(id)`, `replace(old, new)`, `remove(id)`,
`contains(id)`, `load(ids)`.

### 4.1 `lru` (default) — unchanged
`RoleLRUCache` as today. Front = MRU, tail evicted. Re-inserting an existing id moves it to
the front.

### 4.2 `fifo` — new `RoleFIFOCache`
First-in-first-out by arrival order. Same list representation (front = newest arrival, tail
= oldest), same eviction (drop the tail past capacity). The **only** difference from LRU:
`insert(id)` on an **already-present** id is a no-op (it keeps its arrival position; no
"refresh on access"). New ids are inserted at the front. `replace`/`remove`/`contains`/`load`
are identical to LRU and can be shared via a common base.

### 4.3 `none` — no cache
No per-role cache state is maintained. Implemented with a manager-level `cache_enabled` flag
(set `False` when `cache_strategy == "none"`) that changes the read flow (§6), rather than a
null cache object. Write side: `add_memory` still writes to the store but performs **no**
cache insert. Consensus cache-repointing loops become no-ops (no caches to update).

The cache class for `lru`/`fifo` is selected at construction; `none` sets `cache_enabled =
False` and constructs no caches.

---

## 5. Knob 3 — Index backend (store interface)

The existing `SharedChromaStore` is one implementation of a store interface used by the
manager and consensus. The interface (already the de-facto surface of `SharedChromaStore`):

```
add(text, owners) -> id
delete(id)
get(id) -> text | None
owners_of(id) -> set[str]
query(text, owner, top_k) -> [(id, text)]          # owner-filtered nearest
query_global(text, top_k, exclude_id=None) -> [(id, text)]   # unfiltered nearest
count() -> int
all_entries() -> [dict]        # holographic export (see §7)
restore(entries) -> None       # holographic import (see §7)
```

Because consensus only uses these methods, **consensus works unchanged over either backend**.

### 5.1 `chroma` (default) — unchanged
`SharedChromaStore`, embedding similarity. Embeddings supplied by BookWorld's
`embedding_fn`. In-memory (`EphemeralClient`) as currently wired (no `persist_dir`).

### 5.2 `keyword` — new `KeywordSQLiteStore`
No embeddings. In-process SQLite (`sqlite3`, `:memory:` database). "Similarity" = count of
shared LLM-extracted keywords.

**Keyword extraction (`keyword_fn`).** A `keyword_fn(text) -> list[str]` is injected into the
store (symmetric to `embedding_fn` for chroma). The manager builds it from `self.llm` with a
keyword-extraction prompt: "extract up to K (default 5) representative keywords from this
text; return a JSON array of strings." Keywords are normalized (stripped, lowercased,
de-duplicated). Works for zh + en; no whitespace tokenization assumption.

**Schema (three tables):**
- `memories(id TEXT PRIMARY KEY, text TEXT)`
- `owners(mem_id TEXT, role TEXT)` — index on `role`, on `mem_id`
- `keywords(mem_id TEXT, keyword TEXT)` — index on `keyword`, on `mem_id`

**Methods:**
- `add(text, owners)`: `id = uuid4`; `kws = keyword_fn(text)`; insert one `memories` row,
  one `owners` row per owner, one `keywords` row per keyword. Return `id`.
- `query(text, owner, top_k)`: `qk = keyword_fn(text)`; if `qk` empty → `[]`. SQL:
  ```sql
  SELECT k.mem_id, COUNT(DISTINCT k.keyword) AS score
  FROM keywords k JOIN owners o ON o.mem_id = k.mem_id AND o.role = :owner
  WHERE k.keyword IN (:qk)
  GROUP BY k.mem_id ORDER BY score DESC LIMIT :top_k
  ```
  Join `memories` for text. Return `[(id, text)]` highest-overlap first. Ties broken by
  insertion order (rowid).
- `query_global(text, top_k, exclude_id=None)`: same without the owner join, with
  `k.mem_id != :exclude_id`.
- `delete(id)`: delete from all three tables.
- `get`, `owners_of`, `count`, `all_entries`, `restore`: per §7.

**Empty-result semantics.** A query with no keyword overlap returns `[]` — a legitimate
"miss with empty retrieval" (returns empty memory block). Same as chroma returning nothing
on an empty collection.

---

## 6. Read flow with the cache knob — `retrieve_for_role(role_code, query)`

```
if not cache_enabled:                       # cache_strategy == "none"
    hits = store.query(query, owner=role_code, top_k=miss_retrieve_k)
    return format([t for _, t in hits])     # straight to index; no judge, no state

# cache_enabled (lru or fifo):  unchanged from today
cache = cache_for(role_code)
cached = [get(i) for i in cache.ids()]
if cached and judge_sufficient(cached, query):   # LLM judge
    return format(cached)                        # hit
hits = store.query(query, owner=role_code, top_k=miss_retrieve_k)   # miss
for id in reversed(hits): cache.insert(id)       # most-similar ends at front
return format([get(i) for i in cache.ids()])
```

The `none` path skips the sufficiency-judge LLM call entirely and never touches cache state.

---

## 7. Holographic sidecar persistence

Keep the current model: **store lives in memory; persistence is a JSON sidecar**
(`lru_chroma_state.json`) written by `save_to_file` / read by `load_from_file`. **No** true
on-disk DB (no chroma `persist_dir`, no SQLite file). Per the decision, the sidecar stores
**holographic derived data** so loads require zero recomputation.

`all_entries()` per backend returns one dict per entry:
- chroma: `{id, text, owners, embedding}` — `embedding` is the stored vector (list of float).
- keyword: `{id, text, owners, keywords}` — `keywords` is the stored list of strings.

`restore(entries)` uses the derived field directly:
- chroma: re-add with the saved `embedding` (no `embedding_fn` call). Fall back to recompute
  only if `embedding` is absent (back-compat with old sidecars).
- keyword: re-insert rows using saved `keywords` (no `keyword_fn` / LLM call). Fall back to
  recompute only if `keywords` is absent.

The manager sidecar still also stores each role's cache id list and the `pending` queue
(unchanged). `caches` is empty/absent when `cache_strategy == "none"`.

---

## 8. Wiring (`BookWorld.py`)

`BookWorld.__init__` reads the three config keys (with defaults) and passes them to
`LRUChromaMemoryManager`:

```python
self.lru_memory = LRUChromaMemoryManager(
    llm=..., embedding_fn=embedding, role_codes=...,
    consensus_threshold=self.consensus_threshold,
    consensus_enabled=config.get("consensus_enabled", True),
    cache_strategy=config.get("cache_strategy", "lru"),
    index_backend=config.get("index_backend", "chroma"),
    ...
)
```

The manager constructs `self.store` (chroma or keyword) and, when `index_backend ==
"keyword"`, builds `keyword_fn` from `self.llm`. `embedding_fn` is ignored by the keyword
backend (BookWorld still builds an embedding model upstream; harmless).

---

## 9. Testing

Existing 18 tests stay green (defaults preserve current behavior). New tests:

- **FIFO cache** (`test_role_fifo_cache.py`): re-insert-no-reorder, new-id-to-front,
  tail eviction, replace/remove parity with LRU.
- **`none` read flow**: `retrieve_for_role` returns store hits directly, no judge call
  (assert the fake LLM judge prompt is never sent), no cache state retained.
- **Consensus toggle off**: `add_memory` leaves `pending` empty and never calls the
  consensus LLM; `run_consensus()` is a no-op.
- **`KeywordSQLiteStore`**: add/get/owners_of/delete; `query` owner-filtering + overlap
  ranking; `query_global` exclude_id; empty-keyword → `[]`; using a fake `keyword_fn`.
- **Keyword backend end-to-end**: manager with `index_backend="keyword"` does a
  retrieve + a consensus merge (fake llm + fake keyword_fn).
- **Holographic sidecar**: save then load with no `embedding_fn` / `keyword_fn` recompute
  (assert recompute hook not called when derived data present); back-compat load when the
  derived field is absent (recompute path).

---

## 10. Out of scope / non-goals

- No change to `"consensus"` / `"gmemory"` mechanisms.
- No true on-disk persistence (in-memory + sidecar only).
- No new third-party dependency (sqlite3 is stdlib; keyword extraction reuses the LLM).
- MongoDB backend not implemented (SQLite chosen).
- World agent still has no cache and issues no queries.
- Residual/consensus entries still never re-trigger consensus.

---

## 11. Open risks

- **LLM keyword quality / cost.** Keyword extraction costs one LLM call per `add_memory` and
  per retrieval miss. Mitigated by holographic sidecar (no recompute on load) and by the
  same `consensus_threshold` batching that already bounds consensus cost.
- **Keyword recall.** Pure keyword overlap can miss paraphrases an embedding index would
  catch — this is precisely the ablation signal being measured, not a bug.
- **Sidecar size.** Storing embeddings inflates the JSON; acceptable for BookWorld-scale
  casts and the in-memory model.
