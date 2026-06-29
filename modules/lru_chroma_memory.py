"""LRU-Cache + shared ChromaDB global memory mechanism for BookWorld.

Selectable as ``memory_type = "lru_chroma"``. See
``docs/superpowers/specs/2026-06-29-lru-chroma-memory-design.md``.

Components:
  - RoleLRUCache          — per-role LRU cache holding only memory ids.
  - SharedChromaStore     — single global vector store with owner-set metadata.
  - LRUChromaMemoryManager — orchestrates retrieval + consensus.
"""

import json
import os
import uuid
from typing import List, Optional, Set, Tuple


class RoleLRUCache:
    """Per-role LRU cache. Stores only memory ids, never content.

    Front of the list is most-recently-used (MRU); the tail is evicted first.
    """

    def __init__(self, capacity: int = 20):
        self.capacity = capacity
        self._ids: List[str] = []  # index 0 == most recently used

    def ids(self) -> List[str]:
        return list(self._ids)

    def contains(self, mem_id: str) -> bool:
        return mem_id in self._ids

    def insert(self, mem_id: str):
        """Insert/refresh an id at the front; evict the LRU tail past capacity."""
        if mem_id in self._ids:
            self._ids.remove(mem_id)
        self._ids.insert(0, mem_id)
        if len(self._ids) > self.capacity:
            self._ids = self._ids[: self.capacity]

    def replace(self, old_id: str, new_id: str):
        """Repoint a slot in place, preserving its position. No-op if absent."""
        try:
            idx = self._ids.index(old_id)
        except ValueError:
            return
        self._ids[idx] = new_id

    def remove(self, mem_id: str):
        if mem_id in self._ids:
            self._ids.remove(mem_id)

    def load(self, ids: List[str]):
        """Restore cache contents (front == most recently used)."""
        self._ids = list(ids)[: self.capacity]


def _owner_flag(role_code: str) -> str:
    return f"owner_{role_code}"


class SharedChromaStore:
    """A single global ChromaDB collection shared by all roles and the world.

    Each entry stores the memory text as the document, with owner-set metadata
    encoded as one boolean flag per owning role (``owner_<role>=True``) so that
    membership filtering is a native ChromaDB ``where`` clause. Embeddings are
    supplied explicitly via *embedding_fn* (``list[str] -> list[vector]``), so the
    store does not depend on ChromaDB's EmbeddingFunction interface.
    """

    def __init__(self, embedding_fn, collection_name: str = "lru_chroma_memory",
                 persist_dir: Optional[str] = None):
        import chromadb

        self._embedding_fn = embedding_fn
        if persist_dir:
            self._client = chromadb.PersistentClient(path=persist_dir)
        else:
            # EphemeralClient shares one in-memory system per process, so make
            # the collection name unique to isolate each in-memory store.
            self._client = chromadb.EphemeralClient()
            collection_name = f"{collection_name}_{uuid.uuid4().hex[:8]}"
        # We pass embeddings explicitly; no collection-level embedding function.
        self._collection = self._client.get_or_create_collection(name=collection_name)

    # ---- write ----

    def add(self, text: str, owners: Set[str]) -> str:
        mem_id = str(uuid.uuid4())
        emb = self._embedding_fn([text])[0]
        self._collection.add(
            ids=[mem_id],
            embeddings=[emb],
            documents=[text],
            metadatas=[self._owner_metadata(owners)],
        )
        return mem_id

    def delete(self, mem_id: str):
        self._collection.delete(ids=[mem_id])

    # ---- read ----

    def get(self, mem_id: str) -> Optional[str]:
        res = self._collection.get(ids=[mem_id], include=["documents"])
        docs = res.get("documents") or []
        return docs[0] if docs else None

    def owners_of(self, mem_id: str) -> Set[str]:
        res = self._collection.get(ids=[mem_id], include=["metadatas"])
        metas = res.get("metadatas") or []
        if not metas:
            return set()
        return set(json.loads(metas[0].get("owners", "[]")))

    def query(self, text: str, owner: str, top_k: int) -> List[Tuple[str, str]]:
        """Owner-filtered nearest-neighbor search. Returns [(id, text), ...]."""
        return self._query(text, top_k, where={_owner_flag(owner): True})

    def query_global(self, text: str, top_k: int,
                     exclude_id: Optional[str] = None) -> List[Tuple[str, str]]:
        """Unfiltered nearest-neighbor search, excluding *exclude_id*."""
        # Over-fetch by one so excluding self still yields up to top_k results.
        raw = self._query(text, top_k + 1 if exclude_id else top_k, where=None)
        out = [(i, t) for i, t in raw if i != exclude_id]
        return out[:top_k]

    def count(self) -> int:
        return self._collection.count()

    def all_entries(self) -> List[dict]:
        """Export every entry as {id, text, owners} (owners as a sorted list)."""
        res = self._collection.get(include=["documents", "metadatas"])
        ids = res.get("ids") or []
        docs = res.get("documents") or []
        metas = res.get("metadatas") or []
        out = []
        for i, mem_id in enumerate(ids):
            meta = metas[i] if i < len(metas) else {}
            owners = json.loads(meta.get("owners", "[]")) if meta else []
            out.append({"id": mem_id, "text": docs[i], "owners": owners})
        return out

    def restore(self, entries: List[dict]):
        """Re-import entries (preserving ids), recomputing embeddings."""
        for e in entries:
            text, owners = e["text"], set(e.get("owners", []))
            emb = self._embedding_fn([text])[0]
            self._collection.add(
                ids=[e["id"]], embeddings=[emb], documents=[text],
                metadatas=[self._owner_metadata(owners)],
            )

    # ---- internal ----

    def _query(self, text: str, top_k: int, where) -> List[Tuple[str, str]]:
        n = self._collection.count()
        if n < 1 or top_k < 1:
            return []
        emb = self._embedding_fn([text])[0]
        kwargs = dict(query_embeddings=[emb], n_results=min(top_k, n),
                      include=["documents"])
        if where is not None:
            kwargs["where"] = where
        res = self._collection.query(**kwargs)
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        return list(zip(ids, docs))

    @staticmethod
    def _owner_metadata(owners: Set[str]) -> dict:
        meta = {_owner_flag(o): True for o in owners}
        meta["owners"] = json.dumps(sorted(owners))
        return meta


JUDGE_SUFFICIENT_PROMPT = """You are deciding whether a character's cached memories are \
enough to respond to the current situation.

## Cached memories:
{memories}

## Current situation / query:
{query}

Are the cached memories sufficient to respond to the current situation without \
retrieving more memories? Answer with a single word: "yes" or "no".
"""


CONSENSUS_PROMPT = """You are compressing a group of overlapping memories from a \
multi-character role-play simulation.

You are given an ordered array of {count} memories:
{memories}

Produce a JSON array of exactly {out_count} strings:
  - Element 0: the most CONCISE statement of the consensus — the shared facts that \
all {count} memories agree on.
  - Elements 1..{count}: for each input memory in order, the part of that memory NOT \
already covered by the consensus, written as concisely as possible. If an input memory \
is ENTIRELY covered by the consensus, output an empty string "" at its position.

HARD REQUIREMENT: the total number of characters in your output (the consensus plus all \
residuals) MUST be less than or equal to the total number of characters in the input \
memories. Be ruthless about removing redundancy.

Return ONLY the JSON array, parsable by json.loads(). Do not include ```json fences.
"""


def _parse_json_list(text: str) -> Optional[list]:
    """Parse a JSON array of strings from LLM output, tolerating markdown fences."""
    import re
    if not text:
        return None
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return None
    try:
        result = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    if not isinstance(result, list):
        return None
    return [str(x) for x in result]


class LRUChromaMemoryManager:
    """Orchestrates per-role LRU caches over a shared ChromaDB store.

    Read:  retrieve_for_role(role, query) — judge cache sufficiency, fall back to
           an owner-filtered store query on a miss, LRU-update the cache.
    Write: add_memory(role, text) — store + cache + enqueue for consensus.
    """

    def __init__(self, llm, embedding_fn, role_codes: List[str],
                 collection_name: str = "lru_chroma_memory",
                 persist_dir: Optional[str] = None,
                 cache_capacity: int = 20, miss_retrieve_k: int = 5,
                 consensus_top_k: int = 5, consensus_threshold: int = 10,
                 language: str = "zh"):
        self.llm = llm
        self.store = SharedChromaStore(embedding_fn, collection_name, persist_dir)
        self.cache_capacity = cache_capacity
        self.miss_retrieve_k = miss_retrieve_k
        self.consensus_top_k = consensus_top_k
        self.consensus_threshold = consensus_threshold
        self.language = language
        self._caches = {rc: RoleLRUCache(cache_capacity) for rc in role_codes}
        self._pending: List[str] = []

    # ---- read ----

    def retrieve_for_role(self, role_code: str, query: str) -> str:
        cache = self._cache_for(role_code)

        cached_texts = [t for t in (self.store.get(i) for i in cache.ids())
                        if t]
        if cached_texts and self._judge_sufficient(cached_texts, query):
            return self._format(cached_texts)

        # cache miss (empty cache short-circuits the judge above)
        hits = self.store.query(query, owner=role_code, top_k=self.miss_retrieve_k)
        # insert in reverse similarity so the most-similar (top1) ends at the front
        for mem_id, _ in reversed(hits):
            cache.insert(mem_id)

        final_texts = [t for t in (self.store.get(i) for i in cache.ids()) if t]
        return self._format(final_texts)

    # ---- write ----

    def add_memory(self, role_code: str, text: str) -> str:
        mem_id = self.store.add(text, owners={role_code})
        self._cache_for(role_code).insert(mem_id)
        self._pending.append(mem_id)
        if len(self._pending) >= self.consensus_threshold:
            self.run_consensus()
        return mem_id

    # ---- consensus ----

    def run_consensus(self):
        """Process every pending new memory through the global consensus merge.

        Consensus/residual entries are never re-enqueued, so this terminates.
        """
        queue = self._pending
        self._pending = []
        for mem_id in queue:
            self._consensus_one(mem_id)

    def _consensus_one(self, mem_id: str):
        m_text = self.store.get(mem_id)
        if m_text is None:  # already consumed by an earlier merge this batch
            return
        neighbors = [(i, t) for i, t in
                     self.store.query_global(m_text, top_k=self.consensus_top_k,
                                             exclude_id=mem_id)
                     if t is not None]
        if not neighbors:
            return

        best = None  # (total_chars, n, consensus, residuals)
        for n in range(1, len(neighbors) + 1):
            inputs = [m_text] + [neighbors[j][1] for j in range(n)]
            parsed = self._consensus_llm(inputs)
            if parsed is None:
                continue
            consensus, residuals = parsed[0], parsed[1:]
            total = len(consensus) + sum(len(r) for r in residuals)
            if best is None or total < best[0]:
                best = (total, n, consensus, residuals)

        if best is None:
            return
        total, n, consensus, residuals = best
        input_ids = [mem_id] + [neighbors[j][0] for j in range(n)]
        inputs_total = sum(len(t) for t in
                           [m_text] + [neighbors[j][1] for j in range(n)])
        if total >= inputs_total:  # safety fallback: no real compression
            return

        # gather owners before any deletion
        input_owners = [self.store.owners_of(iid) for iid in input_ids]
        union: Set[str] = set()
        for o in input_owners:
            union |= o

        self.store.add(consensus, owners=union)  # consensus entry (not cached)
        for k, iid in enumerate(input_ids):
            resid = residuals[k].strip()
            new_id = self.store.add(resid, owners=input_owners[k]) if resid else None
            self.store.delete(iid)
            for cache in self._caches.values():
                if new_id is not None:
                    cache.replace(iid, new_id)
                else:
                    cache.remove(iid)

    def _consensus_llm(self, inputs: List[str]) -> Optional[list]:
        listing = "\n".join(f"[{i}] {t}" for i, t in enumerate(inputs))
        prompt = CONSENSUS_PROMPT.format(
            count=len(inputs), out_count=len(inputs) + 1, memories=listing)
        try:
            response = self.llm.chat(prompt)
        except Exception:
            return None
        parsed = _parse_json_list(response)
        if parsed is None or len(parsed) != len(inputs) + 1:
            return None
        return parsed

    # ---- helpers ----

    def pending_count(self) -> int:
        return len(self._pending)

    def cache_ids(self, role_code: str) -> List[str]:
        return self._cache_for(role_code).ids()

    def current_cache_text(self, role_code: str) -> str:
        """Format the role's current cache contents without a judge or store query."""
        texts = [t for t in (self.store.get(i)
                             for i in self._cache_for(role_code).ids()) if t]
        return self._format(texts)

    def _cache_for(self, role_code: str) -> RoleLRUCache:
        if role_code not in self._caches:
            self._caches[role_code] = RoleLRUCache(self.cache_capacity)
        return self._caches[role_code]

    def _judge_sufficient(self, cached_texts: List[str], query: str) -> bool:
        prompt = JUDGE_SUFFICIENT_PROMPT.format(
            memories="\n".join(f"- {t}" for t in cached_texts), query=query)
        try:
            response = self.llm.chat(prompt)
        except Exception:
            return False
        return "yes" in (response or "").strip().lower()[:5]

    def _format(self, texts: List[str]) -> str:
        if not texts:
            return ""
        label = "[个人记忆]" if self.language == "zh" else "[Personal Memory]"
        return "\n" + label + "\n" + "\n".join(texts) + "\n"

    # ---- persistence ----

    def save_to_file(self, root_dir: str, filename: str = "lru_chroma_state.json"):
        os.makedirs(root_dir, exist_ok=True)
        data = {
            "entries": self.store.all_entries(),
            "caches": {rc: cache.ids() for rc, cache in self._caches.items()},
            "pending": list(self._pending),
        }
        with open(os.path.join(root_dir, filename), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_from_file(self, root_dir: str, filename: str = "lru_chroma_state.json"):
        path = os.path.join(root_dir, filename)
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.store.restore(data.get("entries", []))
        for rc, ids in data.get("caches", {}).items():
            self._cache_for(rc).load(ids)
        self._pending = list(data.get("pending", []))
