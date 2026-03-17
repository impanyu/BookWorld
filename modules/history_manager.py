from typing import Any, Dict, List, Optional, Literal
from datetime import datetime
from bw_utils import load_json_file, save_json_file
import os
import math


class HistoryManager:
    """Embedding-aware memory store used by both role agents and world agent.

    Each memory item is a dict with at least:
        - detail (str): the textual content
        - timestamp (str): ISO-format add time
        - is_consensused (bool): whether consensus has processed this item
    Role-agent items carry additional metadata (cur_round, role_code, …).
    """

    def __init__(self, embedding_fn=None):
        self.detailed_history: List[Dict] = []
        self.total_word_count: int = 0
        self._embedding_fn = embedding_fn
        self._embeddings: List = []          # parallel cache of embedding vectors
        self._new_record_start_idx: int = 0  # backward-compat cursor

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def set_embedding_fn(self, embedding_fn):
        """Set / update the embedding function (e.g. after deserialization)."""
        self._embedding_fn = embedding_fn
        self._rebuild_embeddings()

    def _rebuild_embeddings(self):
        """Batch-recompute embeddings for all items that are missing them."""
        if not self._embedding_fn:
            return
        details = [r.get("detail", "") for r in self.detailed_history]
        non_empty_indices = [i for i, d in enumerate(details) if d]
        non_empty_texts = [details[i] for i in non_empty_indices]

        self._embeddings = [None] * len(self.detailed_history)
        if not non_empty_texts:
            return
        try:
            all_embs = self._embedding_fn(non_empty_texts)
            for j, idx in enumerate(non_empty_indices):
                self._embeddings[idx] = all_embs[j]
        except Exception:
            for idx in non_empty_indices:
                try:
                    self._embeddings[idx] = self._embedding_fn([details[idx]])[0]
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Add / modify / remove
    # ------------------------------------------------------------------

    @staticmethod
    def count_words(text: str) -> int:
        if not text:
            return 0
        return len(text.split())

    def add_record(self, record: dict):
        """Add a record dict. Automatically stamps *timestamp* and *is_consensused*."""
        record.setdefault("timestamp", datetime.now().isoformat())
        record.setdefault("is_consensused", False)

        if "detail" in record:
            self.total_word_count += self.count_words(record["detail"])
        self.detailed_history.append(record)

        detail = record.get("detail", "")
        if detail and self._embedding_fn:
            try:
                emb = self._embedding_fn([detail])[0]
                self._embeddings.append(emb)
            except Exception:
                self._embeddings.append(None)
        else:
            self._embeddings.append(None)

    def add_memory(self, text: str, metadata: Optional[dict] = None):
        """Convenience wrapper: add a simple text memory item (e.g. consensus items)."""
        record = {
            "detail": text,
            "timestamp": datetime.now().isoformat(),
            "is_consensused": True,
            **(metadata or {})
        }
        self.add_record(record)

    def replace_record_detail(self, idx: int, new_detail: str):
        """Replace a record's detail and update cached embedding."""
        if not (0 <= idx < len(self.detailed_history)):
            return
        self.detailed_history[idx]["detail"] = new_detail
        if idx < len(self._embeddings):
            if self._embedding_fn and new_detail:
                try:
                    self._embeddings[idx] = self._embedding_fn([new_detail])[0]
                except Exception:
                    self._embeddings[idx] = None
            else:
                self._embeddings[idx] = None

    def remove_record(self, idx: int):
        """Mark a record as removed by clearing its detail and embedding."""
        if not (0 <= idx < len(self.detailed_history)):
            return
        self.detailed_history[idx]["detail"] = ""
        if idx < len(self._embeddings):
            self._embeddings[idx] = None

    # ------------------------------------------------------------------
    # Consensus tracking
    # ------------------------------------------------------------------

    def get_unconsensused_items(self) -> List[Dict]:
        """Return items not yet processed by consensus, with their indices."""
        return [
            {"idx": i, "detail": r["detail"], "record": r}
            for i, r in enumerate(self.detailed_history)
            if r.get("detail", "") and not r.get("is_consensused", False)
        ]

    def get_unconsensused_count(self) -> int:
        return sum(
            1 for r in self.detailed_history
            if r.get("detail", "") and not r.get("is_consensused", False)
        )

    def mark_consensused(self, indices: List[int]):
        for idx in indices:
            if 0 <= idx < len(self.detailed_history):
                self.detailed_history[idx]["is_consensused"] = True

    def mark_all_consensused(self):
        for r in self.detailed_history:
            r["is_consensused"] = True

    # backward-compat cursor (used by Server's own HistoryManager)
    def get_new_records(self) -> List[Dict]:
        return [
            {"idx": i, "detail": r["detail"], "record": r}
            for i, r in enumerate(
                self.detailed_history[self._new_record_start_idx:],
                start=self._new_record_start_idx,
            )
            if r.get("detail", "")
        ]

    def get_new_record_count(self) -> int:
        return sum(
            1 for r in self.detailed_history[self._new_record_start_idx:]
            if r.get("detail", "")
        )

    def reset_new_record_cursor(self):
        self._new_record_start_idx = len(self.detailed_history)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_all_details(self) -> List[str]:
        """Return all non-empty details from the full history."""
        return [r["detail"] for r in self.detailed_history if r.get("detail", "")]

    def retrieve_by_similarity(self, query: str, top_k: int = 5) -> List[str]:
        """Return the *top_k* most similar memory items to *query*.

        Uses cached embeddings; falls back to recent history when no
        embedding function is available.
        """
        if not self._embedding_fn:
            return self.get_recent_history(top_k)

        if len(self._embeddings) < len(self.detailed_history):
            self._rebuild_embeddings()

        try:
            query_emb = self._embedding_fn([query])[0]
        except Exception:
            return self.get_recent_history(top_k)

        similarities = []
        for i, record in enumerate(self.detailed_history):
            detail = record.get("detail", "")
            emb = self._embeddings[i] if i < len(self._embeddings) else None
            if not detail or emb is None:
                continue
            sim = self._cosine_similarity(query_emb, emb)
            similarities.append((sim, detail))

        similarities.sort(key=lambda x: x[0], reverse=True)
        return [detail for _, detail in similarities[:top_k]]

    @staticmethod
    def _cosine_similarity(vec_a, vec_b) -> float:
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        na = math.sqrt(sum(a * a for a in vec_a))
        nb = math.sqrt(sum(b * b for b in vec_b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    # ------------------------------------------------------------------
    # Record access helpers
    # ------------------------------------------------------------------

    def modify_record(self, record_id: str, detail: str):
        """修改特定记录"""
        for i, record in enumerate(self.detailed_history):
            if record.get("record_id") == record_id:
                record["detail"] = detail
                if i < len(self._embeddings) and self._embedding_fn and detail:
                    try:
                        self._embeddings[i] = self._embedding_fn([detail])[0]
                    except Exception:
                        pass
                print(f"Record {record_id} has been modified.")
                return record.get("group")

    def search_record_detail(self, record_id: str):
        for record in self.detailed_history[::-1]:
            if record.get("record_id") == record_id:
                return record["detail"]
        return None

    def get_recent_history(self, recent_k=5):
        return [record["detail"] for record in self.detailed_history[-recent_k:]]

    def get_subsequent_history(self, start_idx):
        return [record["detail"] for record in self.detailed_history[start_idx:]]

    def get_complete_history(self):
        return [record["detail"] for record in self.detailed_history[:]]

    def __len__(self):
        return len(self.detailed_history)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def __getstate__(self):
        states = {}
        for key, value in self.__dict__.items():
            if key in ("_embedding_fn", "_embeddings"):
                continue
            if isinstance(value, (str, int, list, dict, bool, type(None))):
                states[key] = value
        return states

    def __setstate__(self, states):
        states.pop("lru_cache", None)
        states.pop("lru_size", None)
        self.__dict__.update(states)
        self._embedding_fn = None
        self._embeddings = []

    def save_to_file(self, root_dir, filename="simulation_history.json"):
        filepath = os.path.join(root_dir, filename)
        save_json_file(filepath, self.__getstate__())

    def load_from_file(self, root_dir, filename="simulation_history.json"):
        filepath = os.path.join(root_dir, filename)
        if os.path.exists(filepath):
            states = load_json_file(filepath)
            self.__setstate__(states)
