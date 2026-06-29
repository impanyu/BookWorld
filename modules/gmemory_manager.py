"""
G-Memory adaptation for BookWorld.

Implements the three-tier hierarchical memory from:
  "G-Memory: Tracing Hierarchical Memory for Multi-Agent Systems"
  (Zhang et al., 2025)

Tiers:
  1. Interaction Store  — condensed round trajectories (embedding-indexed)
  2. Event Graph        — similarity-connected event graph with k-hop retrieval
  3. Insight Manager    — generalizable rules with score-based pruning
"""

import os
import re
import json
import random
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple

from modules import gmemory_prompts as prompts


# ---------------------------------------------------------------------------
# Tier 1 — Interaction Store
# ---------------------------------------------------------------------------

class InteractionStore:
    """Stores condensed round trajectories, indexed by embedding for retrieval."""

    def __init__(self, embedding_fn=None):
        self._records: List[Dict[str, Any]] = []
        self._embeddings: List = []
        self._embedding_fn = embedding_fn

    def add_round(self, event: str, trajectory: str,
                  key_moments: str, round_num: int,
                  metadata: Optional[dict] = None):
        record = {
            "event": event,
            "trajectory": trajectory,
            "key_moments": key_moments,
            "round": round_num,
            "timestamp": datetime.now().isoformat(),
            **(metadata or {}),
        }
        self._records.append(record)

        emb = None
        if self._embedding_fn and event:
            try:
                emb = self._embedding_fn([event])[0]
            except Exception:
                pass
        self._embeddings.append(emb)

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        if not self._records:
            return []
        if not self._embedding_fn or not query:
            return self._records[-top_k:]

        try:
            q_emb = self._embedding_fn([query])[0]
        except Exception:
            return self._records[-top_k:]

        scored = []
        for i, emb in enumerate(self._embeddings):
            if emb is None:
                continue
            sim = _cosine_sim(q_emb, emb)
            scored.append((sim, i))
        scored.sort(reverse=True)
        return [self._records[i] for _, i in scored[:top_k]]

    def get_recent(self, n: int = 5) -> List[Dict]:
        return self._records[-n:]

    def __len__(self):
        return len(self._records)

    def to_dict(self) -> list:
        return self._records

    def load_from_list(self, records: list, embedding_fn=None):
        self._records = records
        self._embedding_fn = embedding_fn or self._embedding_fn
        self._embeddings = []
        for r in self._records:
            emb = None
            if self._embedding_fn and r.get("event"):
                try:
                    emb = self._embedding_fn([r["event"]])[0]
                except Exception:
                    pass
            self._embeddings.append(emb)


# ---------------------------------------------------------------------------
# Tier 2 — Event Graph  (lightweight, no networkx dependency)
# ---------------------------------------------------------------------------

class EventGraph:
    """Similarity-connected event graph with k-hop neighborhood retrieval.

    Nodes are event descriptions; edges connect events whose embedding
    similarity exceeds a threshold.
    """

    def __init__(self, embedding_fn=None, sim_threshold: float = 0.7):
        self._nodes: List[str] = []
        self._edges: Dict[int, List[Tuple[int, float]]] = {}  # adjacency list
        self._embeddings: List = []
        self._embedding_fn = embedding_fn
        self._sim_threshold = sim_threshold

    def add_event(self, event: str):
        if event in self._nodes:
            return
        idx = len(self._nodes)
        self._nodes.append(event)

        emb = None
        if self._embedding_fn:
            try:
                emb = self._embedding_fn([event])[0]
            except Exception:
                pass
        self._embeddings.append(emb)
        self._edges[idx] = []

        if emb is None:
            return
        for j in range(idx):
            other_emb = self._embeddings[j]
            if other_emb is None:
                continue
            sim = _cosine_sim(emb, other_emb)
            if sim >= self._sim_threshold:
                self._edges[idx].append((j, sim))
                self._edges.setdefault(j, []).append((idx, sim))

    def retrieve_related(self, query: str, top_k: int = 3, hop: int = 1) -> List[str]:
        if not self._nodes or not self._embedding_fn:
            return self._nodes[-top_k:]

        try:
            q_emb = self._embedding_fn([query])[0]
        except Exception:
            return self._nodes[-top_k:]

        scored = []
        for i, emb in enumerate(self._embeddings):
            if emb is None:
                continue
            scored.append((_cosine_sim(q_emb, emb), i))
        scored.sort(reverse=True)
        seed_indices = [i for _, i in scored[:top_k]]

        visited = set(seed_indices)
        frontier = list(seed_indices)
        for _ in range(hop):
            next_frontier = []
            for n in frontier:
                for neighbor, _ in self._edges.get(n, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.append(neighbor)
            frontier = next_frontier

        return [self._nodes[i] for i in visited]

    def __len__(self):
        return len(self._nodes)

    def to_dict(self) -> dict:
        edges_ser = {str(k): [(n, w) for n, w in v] for k, v in self._edges.items()}
        return {"nodes": self._nodes, "edges": edges_ser}

    def load_from_dict(self, data: dict, embedding_fn=None):
        self._nodes = data.get("nodes", [])
        self._edges = {int(k): v for k, v in data.get("edges", {}).items()}
        self._embedding_fn = embedding_fn or self._embedding_fn
        self._embeddings = []
        for node in self._nodes:
            emb = None
            if self._embedding_fn:
                try:
                    emb = self._embedding_fn([node])[0]
                except Exception:
                    pass
            self._embeddings.append(emb)


# ---------------------------------------------------------------------------
# Tier 3 — Insight Manager
# ---------------------------------------------------------------------------

class InsightManager:
    """Manages generalizable insights (rules) extracted from simulation rounds.

    Each insight: {rule: str, score: int, related_events: [str]}
    Score increases on AGREE, decreases on REMOVE; pruned when <= 0.
    """

    OPERATIONS_RE = re.compile(
        r"((?:REMOVE|EDIT|ADD|AGREE)(?: \d+|)):\s*(.*)", re.MULTILINE
    )

    def __init__(self, llm=None):
        self.insights: List[Dict[str, Any]] = []
        self._llm = llm

    def set_llm(self, llm):
        self._llm = llm

    def get_insights_text(self) -> str:
        if not self.insights:
            return "(none)"
        return "\n".join(
            f"{i+1}. {ins['rule']}" for i, ins in enumerate(self.insights)
        )

    def query_relevant(self, event: str, top_k: int = 10) -> List[str]:
        """Return the top-k insights whose related events overlap with *event*.

        Falls back to highest-scored insights if no overlap found.
        """
        scored = []
        for ins in self.insights:
            overlap = sum(1 for e in ins.get("related_events", []) if e in event or event in e)
            scored.append((overlap + ins.get("score", 0) * 0.1, ins["rule"]))
        scored.sort(reverse=True)
        return [rule for _, rule in scored[:top_k]]

    def finetune_from_rounds(self, interaction_store: 'InteractionStore',
                             sample_count: int = 3):
        """Sample rounds and derive / update insights."""
        if not self._llm or len(interaction_store) < 2:
            return

        records = interaction_store.to_dict()
        if len(records) >= 2:
            chunks = _random_chunks(records, max(1, len(records) // 3))
            for chunk in chunks[:sample_count]:
                self._finetune_from_success_group(chunk)

        self._prune()

    def _finetune_from_success_group(self, rounds: List[Dict]):
        rounds_text = ""
        for i, r in enumerate(rounds):
            rounds_text += (
                f"\n### Round {i+1} (event: {r.get('event', 'N/A')}):\n"
                f"{r.get('key_moments', r.get('trajectory', ''))}\n"
            )
        existing = self.get_insights_text()
        prompt = prompts.EXTRACT_INSIGHTS_SUCCESS_USER.format(
            success_rounds=rounds_text, existing_insights=existing
        )
        try:
            response = self._llm.chat(prompt)
            ops = self._parse_operations(response)
            self._apply_operations(ops, [r.get("event", "") for r in rounds])
        except Exception as e:
            print(f"[GMemory] Insight finetune failed: {e}")

    def merge_insights(self, limit: int = 10):
        if not self._llm or len(self.insights) <= limit:
            return
        rules_text = "\n".join(ins["rule"] for ins in self.insights)
        prompt = prompts.MERGE_INSIGHTS_USER.format(
            insights=rules_text, limit=limit
        )
        try:
            system_msg = prompts.MERGE_INSIGHTS_SYSTEM
            full_prompt = system_msg + "\n\n" + prompt
            response = self._llm.chat(full_prompt)
            merged = _parse_numbered_list(response)
            if merged:
                self.insights = [
                    {"rule": r, "score": 2, "related_events": []}
                    for r in merged
                ]
        except Exception as e:
            print(f"[GMemory] Insight merge failed: {e}")

    def backward(self, used_insights: List[str], reward: bool):
        delta = 1 if reward else -2
        for ins in self.insights:
            if ins["rule"] in used_insights:
                ins["score"] += delta
        self._prune()

    def project_for_role(self, insights: List[str],
                         role_name: str, role_profile: str,
                         context: str = "") -> List[str]:
        if not self._llm or not insights:
            return insights
        insights_text = "\n".join(f"{i+1}. {r}" for i, r in enumerate(insights))
        if context:
            prompt = prompts.PROJECT_INSIGHTS_WITH_CONTEXT_USER.format(
                context=context, role_name=role_name,
                role_profile=role_profile, insights=insights_text
            )
        else:
            prompt = prompts.PROJECT_INSIGHTS_USER.format(
                role_name=role_name, role_profile=role_profile,
                insights=insights_text
            )
        try:
            full_prompt = prompts.PROJECT_INSIGHTS_SYSTEM + "\n\n" + prompt
            response = self._llm.chat(full_prompt)
            result = _parse_numbered_list(response)
            return result if result else insights
        except Exception:
            return insights

    # ---- internal ----

    def _parse_operations(self, text: str) -> List[Tuple[str, str]]:
        matches = self.OPERATIONS_RE.findall(text)
        ops = []
        for operation, rule_text in matches:
            rule_text = rule_text.strip()
            if not rule_text:
                continue
            ops.append((operation.strip(), rule_text))
        return ops

    def _apply_operations(self, ops: List[Tuple[str, str]],
                          related_events: List[str]):
        for operation, rule_text in ops:
            parts = operation.split()
            op_type = parts[0]
            op_num = int(parts[1]) - 1 if len(parts) > 1 else None

            if op_type == "ADD":
                self.insights.append({
                    "rule": rule_text, "score": 2,
                    "related_events": list(related_events),
                })
            elif op_type == "AGREE" and op_num is not None and 0 <= op_num < len(self.insights):
                self.insights[op_num]["score"] += 1
                self.insights[op_num]["related_events"] = list(
                    set(self.insights[op_num].get("related_events", []) + related_events)
                )
            elif op_type == "EDIT" and op_num is not None and 0 <= op_num < len(self.insights):
                self.insights[op_num]["rule"] = rule_text
                self.insights[op_num]["score"] += 1
            elif op_type == "REMOVE" and op_num is not None and 0 <= op_num < len(self.insights):
                self.insights[op_num]["score"] -= 3

    def _prune(self):
        self.insights = [ins for ins in self.insights if ins.get("score", 0) > 0]

    def to_list(self) -> list:
        return self.insights

    def load_from_list(self, data: list):
        self.insights = data


# ---------------------------------------------------------------------------
# Top-level G-Memory Manager
# ---------------------------------------------------------------------------

class GMemoryManager:
    """Orchestrates the three-tier G-Memory for BookWorld.

    Usage:
        gm = GMemoryManager(llm=..., embedding_fn=...)
        # after each round:
        gm.add_round_memory(event, trajectory, round_num, role_agents)
        # when an agent needs memory:
        memory_text = gm.retrieve_for_role(query, role_name, role_profile)
    """

    def __init__(self, llm=None, embedding_fn=None,
                 insight_start_threshold: int = 3,
                 insight_interval: int = 3,
                 insight_merge_interval: int = 15,
                 hop: int = 1):
        self.interaction_store = InteractionStore(embedding_fn=embedding_fn)
        self.event_graph = EventGraph(embedding_fn=embedding_fn, sim_threshold=0.6)
        self.insight_manager = InsightManager(llm=llm)

        self._llm = llm
        self._embedding_fn = embedding_fn
        self._insight_start = insight_start_threshold
        self._insight_interval = insight_interval
        self._merge_interval = insight_merge_interval
        self._hop = hop
        self._insights_cache: List[str] = []

    # ---- Memory Write (after each round) ----

    def add_round_memory(self, event: str, trajectory: str,
                         round_num: int):
        key_moments = self._condense_trajectory(event, trajectory)

        self.interaction_store.add_round(
            event=event, trajectory=trajectory,
            key_moments=key_moments, round_num=round_num,
        )
        self.event_graph.add_event(event)

        n = len(self.interaction_store)
        if n >= self._insight_start and n % self._insight_interval == 0:
            self.insight_manager.finetune_from_rounds(
                self.interaction_store, sample_count=3
            )
        if n > 0 and n % self._merge_interval == 0:
            self.insight_manager.merge_insights()

    # ---- Memory Read ----

    def retrieve_for_role(self, query: str, role_name: str,
                          role_profile: str, top_k: int = 3) -> str:
        """Retrieve G-Memory context for a role agent's decision-making.

        Returns a formatted text block with:
          - relevant past round key moments
          - role-projected insights
        """
        related_events = self.event_graph.retrieve_related(
            query, top_k=top_k, hop=self._hop
        )

        past_rounds = self.interaction_store.search(query, top_k=top_k)

        raw_insights = self.insight_manager.query_relevant(query, top_k=8)
        self._insights_cache = list(raw_insights)

        role_insights = self.insight_manager.project_for_role(
            raw_insights, role_name, role_profile, context=query
        )

        parts = []
        if past_rounds:
            moments = []
            for r in past_rounds:
                km = r.get("key_moments", "")
                if km:
                    moments.append(f"(Round {r.get('round', '?')}) {km}")
            if moments:
                parts.append(
                    "[Past Experience]\n" + "\n".join(moments[-top_k:])
                )

        if role_insights:
            parts.append(
                "[Insights]\n" + "\n".join(f"- {ins}" for ins in role_insights)
            )

        if not parts:
            return ""
        return "\n" + "\n\n".join(parts) + "\n"

    def backward(self, reward: bool):
        """Update insight scores based on round outcome."""
        if self._insights_cache:
            self.insight_manager.backward(self._insights_cache, reward)
            self._insights_cache = []

    # ---- Persistence ----

    def save_to_file(self, root_dir: str):
        os.makedirs(root_dir, exist_ok=True)
        data = {
            "interactions": self.interaction_store.to_dict(),
            "event_graph": self.event_graph.to_dict(),
            "insights": self.insight_manager.to_list(),
        }
        path = os.path.join(root_dir, "gmemory_state.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_from_file(self, root_dir: str):
        path = os.path.join(root_dir, "gmemory_state.json")
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.interaction_store.load_from_list(
            data.get("interactions", []), self._embedding_fn
        )
        self.event_graph.load_from_dict(
            data.get("event_graph", {}), self._embedding_fn
        )
        self.insight_manager.load_from_list(data.get("insights", []))

    # ---- Internal ----

    def _condense_trajectory(self, event: str, trajectory: str) -> str:
        if not self._llm:
            return trajectory[:500]
        prompt = prompts.CONDENSE_TRAJECTORY_USER.format(
            event=event, trajectory=trajectory
        )
        try:
            full_prompt = prompts.CONDENSE_TRAJECTORY_SYSTEM + "\n\n" + prompt
            return self._llm.chat(full_prompt)
        except Exception:
            return trajectory[:500]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _cosine_sim(a, b) -> float:
    import numpy as np
    a, b = np.array(a), np.array(b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _parse_numbered_list(text: str) -> List[str]:
    pattern = r'\d+\.\s+(.*?)(?=\n\d+\.|\Z)'
    items = re.findall(pattern, text.strip(), flags=re.DOTALL)
    return [item.strip() for item in items if item.strip()]


def _random_chunks(lst: list, chunk_size: int) -> List[list]:
    shuffled = list(lst)
    random.shuffle(shuffled)
    return [shuffled[i:i+chunk_size] for i in range(0, len(shuffled), chunk_size)]
