#!/usr/bin/env python3
"""
Memory evaluation: compare three memory mechanisms in BookWorld.

  1. No Mechanism  — flat personal memory, no shared/global memory, no consensus
  2. Consensus     — personal memory + world consensus memory
  3. G-Memory      — personal memory + hierarchical graph memory

Methodology:
  - Run THREE independent LLM simulations, one per mechanism.
  - Each run produces its own event stream (different due to LLM non-determinism
    and different memory retrieval influencing agent behavior).
  - Compare memory sizes, timing, and quality independently.

Usage:
    python3 eval_memory.py
    python3 eval_memory.py --rounds 10 --threshold 10
    python3 eval_memory.py --rounds 10 --save_dir ./eval_memory_results_10rounds
"""

import sys, os, json, math, argparse, time, re
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# Memory counting — unit = BYTES (UTF-8)
# ═══════════════════════════════════════════════════════════════════════════

def _text_bytes(text: str) -> int:
    return len(text.encode("utf-8")) if text else 0


def memory_bytes_active(hm) -> int:
    """Bytes of non-empty detail fields (actual usable memory)."""
    return sum(_text_bytes(r.get("detail", "")) for r in hm.detailed_history)


def take_snapshot(server, round_num, memory_type):
    """Unified snapshot for any memory type."""
    snap = {"round": round_num}

    for code in server.role_codes:
        hm = server.role_agents[code].history_manager
        snap[f"{code}_bytes"] = memory_bytes_active(hm)

    if memory_type == "consensus":
        whm = server.world_agent.memory
        snap["shared_bytes"] = memory_bytes_active(whm)
    elif memory_type == "gmemory":
        gm = server.gmemory
        gm_bytes = 0
        if gm:
            for rec in gm.interaction_store.to_dict():
                gm_bytes += _text_bytes(rec.get("key_moments", ""))
                gm_bytes += _text_bytes(rec.get("event", ""))
            for ins in gm.insight_manager.insights:
                gm_bytes += _text_bytes(ins.get("rule", ""))
        snap["shared_bytes"] = gm_bytes
    else:
        snap["shared_bytes"] = 0

    snap["total_bytes"] = sum(snap[f"{c}_bytes"] for c in server.role_codes) + snap["shared_bytes"]
    return snap


# ═══════════════════════════════════════════════════════════════════════════
# Per-round dialogue & memory logging
# ═══════════════════════════════════════════════════════════════════════════

def dump_round_log(server, round_0idx, memory_type, log_dir):
    """Write per-round dialogue and every agent's full memory state to files.

    Directory layout:
        log_dir/<mechanism>/round_NNN/dialogue.txt
        log_dir/<mechanism>/round_NNN/memory_<RoleCode>.txt
        log_dir/<mechanism>/round_NNN/memory_world.txt
    """
    round_1idx = round_0idx + 1
    rdir = os.path.join(log_dir, memory_type, f"round_{round_1idx:03d}")
    os.makedirs(rdir, exist_ok=True)

    # --- dialogue ---
    round_records = [
        r for r in server.history_manager.detailed_history
        if r.get("cur_round") == round_0idx and r.get("detail")
    ]
    with open(os.path.join(rdir, "dialogue.txt"), "w", encoding="utf-8") as f:
        f.write(f"=== Round {round_1idx} Dialogue ===\n\n")
        for rec in round_records:
            actor = rec.get("actor", "?")
            act_type = rec.get("act_type", "")
            detail = rec.get("detail", "")
            f.write(f"[{act_type}] {actor}:\n{detail}\n\n")

    # --- per-agent memory ---
    for code in server.role_codes:
        agent_hm = server.role_agents[code].history_manager
        records = agent_hm.detailed_history
        nickname = server.role_agents[code].nickname
        with open(os.path.join(rdir, f"memory_{code}.txt"), "w", encoding="utf-8") as f:
            f.write(f"=== {nickname} ({code}) — Memory after Round {round_1idx} ===\n")
            f.write(f"=== Total records: {len(records)}, "
                    f"Active bytes: {memory_bytes_active(agent_hm)} ===\n\n")
            for i, rec in enumerate(records):
                detail = rec.get("detail", "")
                consensused = rec.get("is_consensused", False)
                rd = rec.get("cur_round", "?")
                ts = rec.get("timestamp", "")
                status = " [consensused]" if consensused else ""
                if not detail:
                    status += " [empty]"
                f.write(f"--- record {i} (round {rd}, {ts}){status} ---\n")
                f.write(f"{detail}\n\n")

    # --- world / shared memory ---
    with open(os.path.join(rdir, "memory_world.txt"), "w", encoding="utf-8") as f:
        if memory_type == "consensus":
            whm = server.world_agent.memory
            records = whm.detailed_history
            f.write(f"=== World Consensus Memory after Round {round_1idx} ===\n")
            f.write(f"=== Total records: {len(records)}, "
                    f"Active bytes: {memory_bytes_active(whm)} ===\n\n")
            for i, rec in enumerate(records):
                detail = rec.get("detail", "")
                src = rec.get("source", "")
                ts = rec.get("timestamp", "")
                f.write(f"--- record {i} (source={src}, {ts}) ---\n")
                f.write(f"{detail}\n\n")
        elif memory_type == "gmemory":
            gm = server.gmemory
            f.write(f"=== G-Memory State after Round {round_1idx} ===\n\n")
            if gm:
                interactions = gm.interaction_store.to_dict()
                f.write(f"--- Interactions ({len(interactions)}) ---\n\n")
                for ix in interactions:
                    f.write(f"[Round {ix.get('round', '?')}] event: {ix.get('event', '')[:200]}\n")
                    f.write(f"  key_moments: {ix.get('key_moments', '')[:500]}\n\n")
                insights = gm.insight_manager.to_list()
                f.write(f"--- Insights ({len(insights)}) ---\n\n")
                for ins in insights:
                    f.write(f"  [{ins.get('score', 0)}] {ins.get('rule', '')}\n")
            else:
                f.write("(gmemory is None)\n")
        else:
            f.write(f"=== No shared memory for mechanism '{memory_type}' ===\n")


# ═══════════════════════════════════════════════════════════════════════════
# Timing
# ═══════════════════════════════════════════════════════════════════════════

def benchmark_retrieval(server, n_queries=10):
    sample_queries = []
    for r in server.history_manager.detailed_history[-20:]:
        d = r.get("detail", "")
        if d:
            sample_queries.append(d[:120])
    if not sample_queries:
        sample_queries = ["What happened recently?"] * n_queries
    while len(sample_queries) < n_queries:
        sample_queries = sample_queries * 2
    sample_queries = sample_queries[:n_queries]

    times = []
    for code in server.role_codes:
        agent = server.role_agents[code]
        for q in sample_queries:
            t0 = time.perf_counter()
            agent.retrieve_history(q, top_k=5, retrieve=True)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
    return times


# ═══════════════════════════════════════════════════════════════════════════
# Simulation runner
# ═══════════════════════════════════════════════════════════════════════════

def run_simulation(preset_path, llm_name, embedding_name,
                   rounds, consensus_threshold, memory_type, log_dir=None):
    from BookWorld import BookWorld

    print(f"\n  Initialising BookWorld (memory_type={memory_type}) …")
    bw = BookWorld(
        preset_path=preset_path,
        world_llm_name=llm_name,
        role_llm_name=llm_name,
        embedding_name=embedding_name,
        memory_top_k=5,
        consensus_threshold=consensus_threshold,
        memory_type=memory_type,
    )
    bw.set_generator(
        rounds=rounds, save_dir="", if_save=0,
        mode="free", scene_mode=0,
    )

    server = bw.server
    role_codes = list(server.role_codes)
    snapshots = []
    round_times = []
    prev_round = -1
    round_start = time.perf_counter()

    print(f"  Running {rounds} rounds ({memory_type}, "
          f"agents: {len(role_codes)}, threshold: {consensus_threshold}) …\n")

    try:
        for msg in bw.generator:
            cur = server.cur_round
            if cur != prev_round:
                if prev_round >= 0:
                    elapsed = time.perf_counter() - round_start
                    round_times.append(elapsed)
                    snap = take_snapshot(server, prev_round + 1, memory_type)
                    snapshots.append(snap)
                    if log_dir:
                        dump_round_log(server, prev_round, memory_type, log_dir)
                    print(f"    Round {prev_round+1:>2}  │  time: {elapsed:.1f}s")
                prev_round = cur
                round_start = time.perf_counter()
    except Exception as e:
        print(f"\n  ERROR at round {prev_round+1}: {e}")
        import traceback; traceback.print_exc()

    if prev_round >= 0:
        elapsed = time.perf_counter() - round_start
        round_times.append(elapsed)
        snap = take_snapshot(server, prev_round + 1, memory_type)
        snapshots.append(snap)
        if log_dir:
            dump_round_log(server, prev_round, memory_type, log_dir)

    print(f"  Finished {memory_type} — {len(snapshots)} rounds.")

    retrieval_times = benchmark_retrieval(server, n_queries=10)
    print(f"  Retrieval benchmark: avg {np.mean(retrieval_times)*1000:.1f}ms")

    transcript = collect_transcript(server)
    return snapshots, role_codes, round_times, retrieval_times, server, transcript


def collect_transcript(server):
    history_texts = server.history_manager.get_complete_history()
    roles_info = {}
    for code in server.role_codes:
        agent = server.role_agents[code]
        roles_info[code] = {
            "nickname": agent.nickname,
            "profile": agent.role_profile,
        }
    return {
        "history_texts": history_texts,
        "roles_info": roles_info,
        "summary": server.script if server.script else server.intervention,
        "source": server.source,
        "language": server.language,
        "num_roles": len(server.role_codes),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Quality evaluation (BookWorld LLM-as-judge)
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_quality(transcript, eval_llm_name, role_llm, mode="free"):
    from modules.eval_agent import EvalAgent
    from bw_utils import merge_text_with_limit

    history_texts = transcript["history_texts"]
    roles_info = transcript["roles_info"]
    language = transcript.get("language", "en")

    eval_agent = EvalAgent(
        roles_info=roles_info,
        summary=transcript["summary"],
        source=transcript["source"],
        llm_name=eval_llm_name,
        role_llm=role_llm,
    )

    start_idx = 2 * transcript["num_roles"]
    max_words = 5000 if language == "zh" else 3000
    text = merge_text_with_limit(
        text_list=history_texts[start_idx:],
        max_words=max_words, language=language,
    )
    if not text:
        print("    Warning: not enough transcript text.")
        return {}

    num_records = len(history_texts) - start_idx
    print(f"    Scoring {num_records} records ({len(text)} chars) …")
    eval_agent.save_generated_text("bookworld", text)
    scores = eval_agent.naive_score(text, method="bookworld", mode=mode)
    return scores or {}


# ═══════════════════════════════════════════════════════════════════════════
# Build unified series from THREE independent runs
# ═══════════════════════════════════════════════════════════════════════════

def build_series(all_snaps, all_round_times, all_retr_times, role_codes):
    """Build unified series dict from three independent runs.
    all_snaps/all_round_times/all_retr_times are dicts keyed by mechanism name.
    """
    min_rounds = min(len(snaps) for snaps in all_snaps.values())
    if min_rounds == 0:
        return None
    rounds = list(range(1, min_rounds + 1))

    series = {"rounds": rounds}
    for key in KEYS:
        snaps = all_snaps[key][:min_rounds]
        series[key] = {
            "total":  [s["total_bytes"] for s in snaps],
            "shared": [s["shared_bytes"] for s in snaps],
            "agents": {c: [s[f"{c}_bytes"] for s in snaps] for c in role_codes},
        }

    series["timing"] = {
        "round_times": {k: all_round_times[k][:min_rounds] for k in KEYS},
        "retrieval":   {k: all_retr_times[k] for k in KEYS},
    }
    return series


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

COLORS  = {"no_consensus": "#e74c3c", "consensus": "#2ecc71", "gmemory": "#3498db"}
LABELS  = {"no_consensus": "No Mechanism", "consensus": "Consensus", "gmemory": "G-Memory"}
MARKERS = {"no_consensus": "o", "consensus": "s", "gmemory": "^"}
KEYS    = ["no_consensus", "consensus", "gmemory"]


def _kb(b):
    return b / 1024


def plot_total_memory(series, save_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    for key in KEYS:
        ax.plot(series["rounds"], [_kb(v) for v in series[key]["total"]],
                f"{MARKERS[key]}-", color=COLORS[key],
                label=LABELS[key], linewidth=2, markersize=5)
    ax.set_xlabel("Round", fontsize=13)
    ax.set_ylabel("Total Memory Size (KB)", fontsize=13)
    ax.set_title("Total Memory Size Across All Agents", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12); ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3); fig.tight_layout()
    path = os.path.join(save_dir, "total_memory_comparison.png")
    fig.savefig(path, dpi=150); plt.close(fig); print(f"  Saved: {path}")


def plot_shared_memory(series, save_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    for key in KEYS:
        ax.plot(series["rounds"], [_kb(v) for v in series[key]["shared"]],
                f"{MARKERS[key]}-", color=COLORS[key],
                label=LABELS[key], linewidth=2, markersize=5)
    ax.set_xlabel("Round", fontsize=13)
    ax.set_ylabel("Shared Memory Size (KB)", fontsize=13)
    ax.set_title("Shared / Global Memory Size", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12); ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3); fig.tight_layout()
    path = os.path.join(save_dir, "shared_memory_comparison.png")
    fig.savefig(path, dpi=150); plt.close(fig); print(f"  Saved: {path}")


def plot_per_agent(series, role_codes, save_dir):
    n = len(role_codes) + 1
    cols = min(4, n); rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    all_keys = list(role_codes) + ["shared"]
    for i, code in enumerate(all_keys):
        r, c = divmod(i, cols); ax = axes[r][c]
        for key in KEYS:
            if code == "shared":
                vals = [_kb(v) for v in series[key]["shared"]]
            else:
                vals = [_kb(v) for v in series[key]["agents"][code]]
            ax.plot(series["rounds"], vals, f"{MARKERS[key]}-",
                    color=COLORS[key], label=LABELS[key], linewidth=1.5, markersize=3)
        label = "Shared / World" if code == "shared" else (code.split("-")[0] if "-" in code else code)
        ax.set_title(label.replace("_", " "), fontsize=11, fontweight="bold")
        ax.set_xlabel("Round", fontsize=9); ax.set_ylabel("Memory (KB)", fontsize=9)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True)); ax.grid(True, alpha=0.3)
        if i == 0: ax.legend(fontsize=7)
    for j in range(i + 1, rows * cols):
        r, c = divmod(j, cols); axes[r][c].set_visible(False)
    fig.suptitle("Per-Agent Memory Size Over Rounds\n"
                 "(Each mechanism runs independently)",
                 fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    path = os.path.join(save_dir, "per_agent_memory_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"  Saved: {path}")


def plot_final_distribution(series, role_codes, save_dir):
    labels = [c.split("-")[0].replace("_", " ") for c in role_codes] + ["Shared/World"]
    x = np.arange(len(labels)); width = 0.25
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 2), 5))
    for idx, key in enumerate(KEYS):
        vals = [_kb(series[key]["agents"][c][-1]) for c in role_codes] + [_kb(series[key]["shared"][-1])]
        offset = (idx - 1) * width
        bars = ax.bar(x + offset, vals, width, label=LABELS[key], color=COLORS[key], alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            if h > 0.01:
                ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Agent", fontsize=13); ax.set_ylabel("Final Memory Size (KB)", fontsize=13)
    ax.set_title("Final Memory Size Distribution by Agent", fontsize=15, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend(fontsize=12); ax.grid(True, axis="y", alpha=0.3); fig.tight_layout()
    path = os.path.join(save_dir, "final_memory_distribution.png")
    fig.savefig(path, dpi=150); plt.close(fig); print(f"  Saved: {path}")


def plot_round_times(series, save_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    for key in KEYS:
        ax.plot(series["rounds"], series["timing"]["round_times"][key],
                f"{MARKERS[key]}-", color=COLORS[key],
                label=LABELS[key], linewidth=2, markersize=5)
    ax.set_xlabel("Round", fontsize=13); ax.set_ylabel("Round Time (seconds)", fontsize=13)
    ax.set_title("Per-Round Wall Time (Independent Runs)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=12); ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3); fig.tight_layout()
    path = os.path.join(save_dir, "round_times_comparison.png")
    fig.savefig(path, dpi=150); plt.close(fig); print(f"  Saved: {path}")


def plot_retrieval_times(series, save_dir):
    data_ms = {k: np.array(series["timing"]["retrieval"][k]) * 1000 for k in KEYS}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    bp = ax.boxplot([data_ms[k] for k in KEYS],
                    labels=[LABELS[k] for k in KEYS], patch_artist=True)
    for i, key in enumerate(KEYS):
        bp["boxes"][i].set_facecolor(COLORS[key]); bp["boxes"][i].set_alpha(0.7)
    ax.set_ylabel("Retrieval Time (ms)", fontsize=13)
    ax.set_title("Retrieval Time Distribution", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    means = [np.mean(data_ms[k]) for k in KEYS]
    stds  = [np.std(data_ms[k])  for k in KEYS]
    bars = ax2.bar([LABELS[k] for k in KEYS], means, yerr=stds, capsize=5,
                   color=[COLORS[k] for k in KEYS], alpha=0.8)
    for bar, m in zip(bars, means):
        ax2.annotate(f"{m:.1f}ms", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     xytext=(0, 5), textcoords="offset points", ha="center", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Avg Retrieval Time (ms)", fontsize=13)
    ax2.set_title("Average Retrieval Time", fontsize=13, fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Memory Retrieval Efficiency (Independent Runs)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(save_dir, "retrieval_time_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"  Saved: {path}")


def plot_quality_scores(quality_scores, save_dir):
    dims = None
    for key in KEYS:
        if quality_scores.get(key):
            dims = list(quality_scores[key].keys())
            break
    if not dims:
        print("  Skipping quality plot — no scores.")
        return

    available_keys = [k for k in KEYS if quality_scores.get(k)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    x = np.arange(len(dims)); width = 0.25
    for idx, key in enumerate(available_keys):
        scores = quality_scores[key]
        vals = [scores.get(d, 0) for d in dims]
        offset = (idx - 1) * width
        bars = ax.bar(x + offset, vals, width, label=LABELS[key], color=COLORS[key], alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f"{h}", xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace(" & ", "\n& ") for d in dims], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Score (1-7)", fontsize=12)
    ax.set_title("Quality Scores by Dimension", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 7.5); ax.legend(fontsize=10); ax.grid(True, axis="y", alpha=0.3)

    ax2 = axes[1]; ax2.remove()
    ax2 = fig.add_subplot(1, 2, 2, polar=True)
    N = len(dims)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]
    for key in available_keys:
        scores = quality_scores[key]
        vals = [scores.get(d, 0) for d in dims] + [scores.get(dims[0], 0)]
        ax2.plot(angles, vals, f"{MARKERS[key]}-", color=COLORS[key],
                 linewidth=2, label=LABELS[key], markersize=4)
        ax2.fill(angles, vals, color=COLORS[key], alpha=0.1)
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels([d[:15] for d in dims], fontsize=8)
    ax2.set_ylim(0, 7); ax2.set_title("Quality Radar", fontsize=13, fontweight="bold", pad=20)
    ax2.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

    fig.suptitle("Narrative Quality Comparison (LLM-as-Judge, 1-7 scale)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(save_dir, "quality_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════

def print_summary(series, role_codes, rounds_count, quality_scores):
    print(f"\n{'=' * 94}")
    print(f"  MEMORY SIZE SUMMARY (KB)  —  after {rounds_count} rounds")
    print(f"  Each mechanism ran as an INDEPENDENT simulation")
    print(f"{'=' * 94}")
    print(f"  {'Agent':<22} {'No Mechanism':>16} {'Consensus':>16} {'G-Memory':>16}")
    print(f"  {'-' * 70}")
    for code in role_codes:
        name = code.split("-")[0].replace("_", " ")
        vals = [_kb(series[k]["agents"][code][-1]) for k in KEYS]
        print(f"  {name:<22} {vals[0]:>13.1f}KB {vals[1]:>13.1f}KB {vals[2]:>13.1f}KB")
    vals_s = [_kb(series[k]["shared"][-1]) for k in KEYS]
    print(f"  {'Shared/World':<22} {vals_s[0]:>13.1f}KB {vals_s[1]:>13.1f}KB {vals_s[2]:>13.1f}KB")
    print(f"  {'-' * 70}")
    vals_t = [_kb(series[k]["total"][-1]) for k in KEYS]
    print(f"  {'TOTAL':<22} {vals_t[0]:>13.1f}KB {vals_t[1]:>13.1f}KB {vals_t[2]:>13.1f}KB")

    nc_agent_total = sum(_kb(series["no_consensus"]["agents"][c][-1]) for c in role_codes)
    con_agent_total = sum(_kb(series["consensus"]["agents"][c][-1]) for c in role_codes)
    diff_agent = nc_agent_total - con_agent_total
    pct_agent = (diff_agent / nc_agent_total * 100) if nc_agent_total > 0 else 0
    print(f"  Per-agent total: No Mech={nc_agent_total:.1f}KB, "
          f"Consensus={con_agent_total:.1f}KB "
          f"(diff: {diff_agent:+.1f}KB / {pct_agent:+.1f}%)")

    diff_total = vals_t[0] - vals_t[1]
    pct_total = (diff_total / vals_t[0] * 100) if vals_t[0] > 0 else 0
    print(f"  Overall total:   No Mech={vals_t[0]:.1f}KB, "
          f"Consensus={vals_t[1]:.1f}KB "
          f"(diff: {diff_total:+.1f}KB / {pct_total:+.1f}%)")

    timing = series["timing"]
    print(f"\n  TIMING")
    print(f"  {'-' * 70}")
    rt = timing["round_times"]
    print(f"  {'Avg round time':<22} {np.mean(rt['no_consensus']):>13.1f}s "
          f"{np.mean(rt['consensus']):>13.1f}s  {np.mean(rt['gmemory']):>13.1f}s")
    retr = timing["retrieval"]
    print(f"  {'Avg retrieval time':<22} {np.mean(retr['no_consensus'])*1000:>12.1f}ms "
          f"{np.mean(retr['consensus'])*1000:>12.1f}ms {np.mean(retr['gmemory'])*1000:>12.1f}ms")

    if any(quality_scores.get(k) for k in KEYS):
        dims = list(next(v for v in quality_scores.values() if v).keys())
        print(f"\n  QUALITY SCORES (1-7)")
        print(f"  {'-' * 70}")
        print(f"  {'Dimension':<22} {'No Mechanism':>16} {'Consensus':>16} {'G-Memory':>16}")
        print(f"  {'-' * 70}")
        for d in dims:
            vals = [quality_scores.get(k, {}).get(d, "-") for k in KEYS]
            row = f"  {d:<22}"
            for v in vals:
                row += f" {v:>16}" if isinstance(v, (int, float)) else f" {str(v):>16}"
            print(row)
    print(f"{'=' * 94}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

RUN_CONFIGS = [
    ("no_consensus", "none"),
    ("consensus",    "consensus"),
    ("gmemory",      "gmemory"),
]

def main():
    parser = argparse.ArgumentParser(
        description="3-way memory mechanism comparison (independent runs)")
    parser.add_argument("--preset", type=str, default="./experiment_presets/experiment_alice.json")
    parser.add_argument("--llm", type=str, default="gpt-4o-mini")
    parser.add_argument("--eval_llm", type=str, default="gpt-4o-mini")
    parser.add_argument("--embedding", type=str, default="openai")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--threshold", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="./eval_memory_results")
    parser.add_argument("--skip_quality", action="store_true")
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        for key in ["OPENAI_API_KEY", "DASHSCOPE_API_KEY", "DEEPSEEK_API_KEY",
                     "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY",
                     "ARK_API_KEY"]:
            val = cfg.get(key, "")
            if val:
                os.environ[key] = val

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"\n{'─' * 72}")
    print(f"  Memory Evaluation — 3-Way Comparison (INDEPENDENT runs)")
    print(f"  Each mechanism runs as a separate simulation")
    print(f"  Preset   : {args.preset}")
    print(f"  LLM      : {args.llm}")
    print(f"  Rounds   : {args.rounds}")
    print(f"  Threshold: {args.threshold}")
    print(f"{'─' * 72}")

    all_snaps = {}
    all_round_times = {}
    all_retr_times = {}
    all_transcripts = {}
    all_servers = {}
    role_codes = None

    for run_idx, (label, mem_type) in enumerate(RUN_CONFIGS, 1):
        print(f"\n{'━' * 54}")
        print(f"  RUN {run_idx} / {len(RUN_CONFIGS)} :  {label.upper()}")
        print(f"{'━' * 54}")
        log_dir = os.path.join(args.save_dir, "logs")
        snaps, codes, rt, retr, server, transcript = run_simulation(
            preset_path=os.path.abspath(args.preset),
            llm_name=args.llm, embedding_name=args.embedding,
            rounds=args.rounds, consensus_threshold=args.threshold,
            memory_type=mem_type, log_dir=log_dir,
        )
        all_snaps[label] = snaps
        all_round_times[label] = rt
        all_retr_times[label] = retr
        all_transcripts[label] = transcript
        all_servers[label] = server
        if role_codes is None:
            role_codes = codes

    if any(len(s) == 0 for s in all_snaps.values()):
        print("\n  Insufficient data from one or more runs. Exiting.")
        return

    series = build_series(all_snaps, all_round_times, all_retr_times, role_codes)
    if series is None:
        print("\n  No overlapping rounds. Exiting.")
        return

    # Quality evaluation
    quality_scores = {}
    if not args.skip_quality:
        print(f"\n{'━' * 54}")
        print(f"  QUALITY EVALUATION (LLM-as-judge)")
        print(f"{'━' * 54}")
        for label in [lbl for lbl, _ in RUN_CONFIGS]:
            print(f"\n  Evaluating {label} …")
            try:
                scores = evaluate_quality(
                    all_transcripts[label], args.eval_llm,
                    role_llm=all_servers[label].role_llm, mode="free")
                quality_scores[label] = scores
                print(f"    Scores: {scores}")
            except Exception as e:
                print(f"    Failed: {e}")
                import traceback; traceback.print_exc()
                quality_scores[label] = {}
    else:
        quality_scores = {k: {} for k in KEYS}

    rounds_count = len(series["rounds"])
    print_summary(series, role_codes, rounds_count, quality_scores)

    print(f"\nGenerating plots → {args.save_dir}/")
    plot_total_memory(series, args.save_dir)
    plot_shared_memory(series, args.save_dir)
    plot_per_agent(series, role_codes, args.save_dir)
    plot_final_distribution(series, role_codes, args.save_dir)
    plot_round_times(series, args.save_dir)
    plot_retrieval_times(series, args.save_dir)
    if any(quality_scores.get(k) for k in KEYS):
        plot_quality_scores(quality_scores, args.save_dir)

    summary = {
        "preset": args.preset, "llm": args.llm, "rounds": rounds_count,
        "consensus_threshold": args.threshold, "role_codes": role_codes,
        "methodology": "3 fully independent simulation runs",
        "series": {k: v for k, v in series.items() if k != "timing"},
        "quality_scores": quality_scores,
        "timestamp": datetime.now().isoformat(),
    }
    summary_path = os.path.join(args.save_dir, "eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {summary_path}")
    print(f"\n  Evaluation complete. All outputs in: {args.save_dir}/\n")


if __name__ == "__main__":
    main()
