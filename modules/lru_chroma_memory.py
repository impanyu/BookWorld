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


KEYWORD_PROMPT = """Extract up to {k} short, representative keywords from the text \
below (single words or very short phrases). Cover the key entities, actions, and topics.

## Text:
{text}

Return ONLY a JSON array of strings parsable by json.loads(). No code fences.
"""


def make_llm_keyword_fn(llm, top_k: int = 5):
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
                 language: str = "zh", consensus_enabled: bool = True,
                 cache_strategy: str = "lru", index_backend: str = "chroma",
                 keyword_top_k: int = 5):
        self.llm = llm
        self.index_backend = index_backend
        if index_backend == "keyword":
            self.keyword_fn = make_llm_keyword_fn(llm, keyword_top_k)
            self.store = KeywordSQLiteStore(self.keyword_fn)
        else:
            self.keyword_fn = None
            self.store = SharedChromaStore(embedding_fn, collection_name,
                                           persist_dir)
        self.cache_capacity = cache_capacity
        self.miss_retrieve_k = miss_retrieve_k
        self.consensus_top_k = consensus_top_k
        self.consensus_threshold = consensus_threshold
        self.consensus_enabled = consensus_enabled
        self.language = language
        self.cache_strategy = cache_strategy
        self.cache_enabled = cache_strategy != "none"
        cache_cls = RoleFIFOCache if cache_strategy == "fifo" else RoleLRUCache
        self._cache_cls = cache_cls
        self._caches = (
            {rc: cache_cls(cache_capacity) for rc in role_codes}
            if self.cache_enabled else {}
        )
        self._pending: List[str] = []

    # ---- read ----

    def retrieve_for_role(self, role_code: str, query: str) -> str:
        if not self.cache_enabled:
            hits = self.store.query(query, owner=role_code,
                                    top_k=self.miss_retrieve_k)
            return self._format([t for _, t in hits])

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
        if self.cache_enabled:
            self._cache_for(role_code).insert(mem_id)
        if self.consensus_enabled:
            self._pending.append(mem_id)
            if len(self._pending) >= self.consensus_threshold:
                self.run_consensus()
        return mem_id

    # ---- consensus ----

    def run_consensus(self):
        """Process every pending new memory through the global consensus merge.

        Consensus/residual entries are never re-enqueued, so this terminates.
        """
        if not self.consensus_enabled:
            return
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

        consensus_id = self.store.add(consensus, owners=union)  # consensus entry

        # Caches that held any of the merged inputs are the ones whose old
        # entries get superseded: repoint each old id to its residual (or drop
        # it when the residual is empty), then LRU-insert the consensus id so
        # the old entries are replaced by the consensus + residual entries.
        touched = [cache for cache in self._caches.values()
                   if any(cache.contains(iid) for iid in input_ids)]
        for k, iid in enumerate(input_ids):
            resid = residuals[k].strip()
            new_id = self.store.add(resid, owners=input_owners[k]) if resid else None
            self.store.delete(iid)
            for cache in self._caches.values():
                if new_id is not None:
                    cache.replace(iid, new_id)
                else:
                    cache.remove(iid)
        for cache in touched:
            cache.insert(consensus_id)

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
        if not self.cache_enabled:
            return []
        return self._cache_for(role_code).ids()

    def current_cache_text(self, role_code: str) -> str:
        """Format the role's current cache contents without a judge or store query."""
        if not self.cache_enabled:
            return ""
        texts = [t for t in (self.store.get(i)
                             for i in self._cache_for(role_code).ids()) if t]
        return self._format(texts)

    def _cache_for(self, role_code: str) -> RoleLRUCache:
        if role_code not in self._caches:
            self._caches[role_code] = self._cache_cls(self.cache_capacity)
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
