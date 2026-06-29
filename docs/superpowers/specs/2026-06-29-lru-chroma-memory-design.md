# LRU-Cache + Shared ChromaDB Global Memory — Design Spec

**Date:** 2026-06-29
**Status:** Approved (pending implementation plan)
**Scope:** A new multi-agent memory mechanism for BookWorld, selectable as `memory_type = "lru_chroma"`, alongside the existing `"consensus"` and `"gmemory"` mechanisms.

---

## 1. Motivation

BookWorld runs a multi-agent role-play simulation where each `RoleAgent` keeps its own
flat `HistoryManager` and the `WorldAgent` keeps a global consensus memory. The existing
`"consensus"` mechanism periodically summarizes overlapping role memories into a shared
world memory and compresses per-role memories.

This new mechanism replaces that design with a **cache + shared vector store** architecture:

- Each role has a small **LRU cache** that holds only **ids** (not memory content).
- A single **shared ChromaDB** collection stores all memory content for all roles and the
  world, with explicit **owner sets** per memory entry.
- A role first checks whether its cache is sufficient to answer the current query
  (an LLM judgment); on a cache miss it pulls the most semantically relevant memories it
  is allowed to see from the shared store and LRU-updates its cache.
- A **consensus mechanism** merges semantically-close memories globally to bound total
  storage: each new memory is compared against its global nearest neighbors, and the
  most compact "shared consensus + per-memory residual" rewrite is committed.

---

## 2. Integration with BookWorld

A new value `memory_type = "lru_chroma"` is added (config + `BookWorld.__init__`).

Two touch points in the live simulation loop:

1. **Write side** — `BookWorld.record()` ([BookWorld.py:762](../../../BookWorld.py)):
   matching the original dispatch (`for code in group: role_agents[code].record(record)`),
   every in-scene role independently stores its own copy of the turn memory via
   `manager.add_memory(role_code, text)` with `owners = {role_code}`. This per-role
   duplication is intentional: it is what produces semantically-close, different-owner
   memories for the consensus step to merge.

2. **Read side** — `RoleAgent.retrieve_history()`
   ([main_role_agent.py:471](../../../modules/main_role_agent.py)): in the `lru_chroma`
   branch, retrieval is delegated to `manager.retrieve_for_role(role_code, query)`.

**Memory text vs. query.** Consistent with BookWorld's uniform record dispatch, the stored
memory `text` for a turn is the turn's record `detail` (the produced utterance/action) —
every in-scene role stores the same `detail` under its own ownership. The `query` passed to
`retrieve_for_role` is the role's current action context (the situation it is responding
to), used only for the cache-sufficiency judgment and miss-retrieval; it is not stored. For
the acting role this matches the user's "query + reply" intent (it has just produced the
reply and is responding to the query); observers store the observed `detail` only.

The `WorldAgent` does **not** get its own cache. World may be an owner of memories
(union ownership after consensus), but it has no LRU cache and does not act on queries.

---

## 3. Components

All new code lives in `modules/lru_chroma_memory.py`. Three classes plus a top-level
manager.

### 3.1 `SharedChromaStore`

A single global ChromaDB collection shared by all roles and the world. Built directly on
`chromadb` (the existing `modules/db/ChromaDB.py` wrapper is text-keyed and supports
neither metadata, id-based ops, nor filtered queries, so it is not reused). The embedding
function is the same `get_embedding_model(...)` used elsewhere in BookWorld (default
`bge-small`), and persistence uses the same ChromaDB persistent-client pattern.

**Per-entry schema:**
- `id`: uuid4 string.
- `document`: the memory text.
- `metadata`:
  - one boolean flag per owning role: `owner_<role_code> = True`. This encodes the
    owner set in a way ChromaDB `where` filters support natively (membership test =
    `where={"owner_<role>": True}`).
  - `owners`: JSON-encoded list of role codes (for readable owner reads + union ops).

**Methods:**
- `add(text, owners) -> id` — insert a new entry; set per-role flags + `owners`.
- `query(text, owner, top_k) -> [(id, text)]` — owner-filtered nearest-neighbor search
  (`where={"owner_<owner>": True}`).
- `query_global(text, top_k, exclude_id=None) -> [(id, text)]` — **unfiltered** global
  nearest-neighbor search (used by consensus); excludes `exclude_id` and any tombstoned
  id.
- `get(id) -> text | None`.
- `owners_of(id) -> set[str]`.
- `delete(id)`.

### 3.2 `RoleLRUCache`

One per role. **Stores only ids**, never memory content.

- Capacity: **20**.
- Ordering: front = most-recently-used (MRU), tail = least-recently-used (LRU).
- Methods:
  - `ids() -> list[str]`.
  - `insert(id)` — if present, move to front; else add at front, evicting the tail when
    size exceeds 20.
  - `replace(old_id, new_id)` — repoint a cache slot in place (preserve position).
  - `remove(id)`.
  - `contains(id) -> bool`.

### 3.3 `LRUChromaMemoryManager`

Top-level orchestrator. Holds:
- `store: SharedChromaStore`.
- `caches: {role_code: RoleLRUCache}` (created lazily / from the role roster).
- `llm`, `embedding`.
- `pending: list[str]` — ids of newly-added turn memories awaiting consensus.
- `consensus_threshold: int` (default 10).
- `cache_capacity = 20`, `miss_retrieve_k = 5`, `consensus_top_k = 5`.

---

## 4. Read Flow — `retrieve_for_role(role_code, query) -> str`

1. If the role's cache is **empty** → immediate cache miss (skip the LLM judgment).
2. Otherwise build the cache content (fetch each cached id's text from the store) and ask
   the LLM: *given these cached memories + this query, are they sufficient to answer?*
   → `yes` / `no`.
3. **Hit (`yes`)** → return the full cache content.
4. **Miss (`no`)**:
   - `store.query(query, owner=role_code, top_k=5)` — owner-filtered top-5.
   - Insert the 5 results into the cache by LRU rules, in **reverse similarity order**
     (top5 first … top1 last) so the most-similar result (top1) ends at the front and is
     least likely to be evicted. Duplicates are moved to the front; new ids are added to
     the front; capacity 20 evicts from the tail.
   - Return the **updated** full cache content.

Return format is a newline-joined block of the cached memory texts (label consistent with
the existing `[个人记忆]` / `[Personal Memory]` convention).

---

## 5. Write + Consensus Flow

### 5.1 `add_memory(role_code, text) -> id`
- `id = store.add(text, owners={role_code})`.
- `caches[role_code].insert(id)`.
- `pending.append(id)`.
- If `len(pending) >= consensus_threshold` → `run_consensus()`.

### 5.2 `run_consensus()`
Process each id in `pending` (skip ids already tombstoned/deleted by an earlier item in
the same batch), then clear the queue. For a new memory `m`:

1. `neighbors = store.query_global(m_text, top_k=5, exclude_id=m)` — global nearest 5,
   no owner filter. (May be < 5 early on.) Let `N = len(neighbors)`. If `N == 0`, skip `m`.
2. For `n in 1..N`:
   - Input array (length `n+1`): `[m_text, neighbor_1_text, …, neighbor_n_text]`.
   - LLM produces an output array (length `n+2`):
     `[consensus, residual_0, residual_1, …, residual_n]`, where:
     - `consensus` = the most concise statement of what all `n+1` memories agree on.
     - `residual_i` = the part of input memory `i` **not** covered by the consensus,
       written as concisely as possible; **empty string** if memory `i` is fully covered.
   - The prompt **hard-requires**: `len(consensus) + Σ len(residual_i)` ≤
     `Σ len(input_i)` (total character count must not grow).
   - Record `total_chars(n) = len(consensus) + Σ len(residual_i)`.
3. Pick `n* = argmin_n total_chars(n)`.
4. **Safety fallback:** if `total_chars(n*)` ≥ `Σ len(input_i for n*)` (consensus failed to
   compress, e.g. LLM violated the constraint), **skip** `m` — leave all entries unchanged.
5. Otherwise commit the `n*` solution:
   - **Consensus entry:** `store.add(consensus, owners = ⋃ owners of the n*+1 memories)`.
     (World may appear in this union if it owned any input.)
   - **Residual entries:** for each non-empty `residual_i`,
     `new_id_i = store.add(residual_i, owners = owners_of(original_i))`. Empty residual →
     no entry (record `new_id_i = None`).
   - **Delete originals:** `store.delete(original_i)` for all `n*+1` inputs (including `m`).
   - **Cache repointing:** for each original id → its `new_id_i`, iterate every role cache:
     - if `new_id_i` is not None → `cache.replace(original_i, new_id_i)`.
     - if `new_id_i` is None → `cache.remove(original_i)`.
   - The **consensus entry is not inserted into any cache**; it surfaces later via
     owner-filtered miss retrieval.

### 5.3 Loop-safety invariants
- Consensus and residual entries are **never** re-enqueued into `pending` — only fresh
  turn memories from `add_memory` trigger consensus. This prevents infinite re-merging.
- Within one `run_consensus()` batch, a later pending id may have been deleted by an
  earlier item's merge; such ids are skipped.

---

## 6. Persistence

- ChromaDB persists its own collection on disk (persistent client).
- `LRUChromaMemoryManager.save_to_file(root_dir)` additionally writes a JSON sidecar with:
  each role's cache id list (order preserved) + the `pending` queue + config.
- `load_from_file(root_dir)` reconstructs caches and pending from the sidecar and re-binds
  the existing ChromaDB collection. The embedding function is re-injected on load (not
  serialized), consistent with `HistoryManager.__setstate__`.

---

## 7. Defaults (configurable)

| Parameter | Default | Meaning |
|---|---|---|
| `cache_capacity` | 20 | per-role LRU cache size (ids only) |
| `miss_retrieve_k` | 5 | entries pulled from the store on a cache miss |
| `consensus_top_k` | 5 | global nearest neighbors examined per new memory |
| `consensus_threshold` | 10 | pending new memories that trigger a consensus batch |
| owner encoding | per-role boolean metadata flags | enables native ChromaDB filtering |

---

## 8. Out of Scope / Non-Goals

- No change to the `"consensus"` or `"gmemory"` mechanisms; this is a parallel third option.
- No ANN index tuning beyond what ChromaDB provides by default.
- World agent gets no cache and does not query memory.
- Residual/consensus entries do not recursively trigger further consensus.

---

## 9. Open Risks

- **LLM compliance with the char-count constraint** is not guaranteed; mitigated by the
  measured argmin + the safety fallback that skips non-compressing merges.
- **Per-insert cost** is bounded by batching consensus at `consensus_threshold`; each
  processed memory still costs up to `consensus_top_k` LLM calls.
- **ChromaDB `where` on many per-role flags**: number of metadata keys grows with the
  number of distinct roles; acceptable for BookWorld's small casts.
