#!/usr/bin/env python3
"""
Memory-size evaluation: run actual BookWorld simulations and compare
memory dynamics with consensus vs. without consensus.

Uses a SINGLE simulation run with consensus enabled, tracking both:
  - actual memory sizes (with consensus filtering)
  - cumulative add counts (what sizes would be without consensus removing anything)

This ensures both conditions share exactly the same events and actions,
giving a fair comparison.

Usage (use the venv with all dependencies):
    python3 eval_memory.py
    python3 eval_memory.py --rounds 5 --threshold 10
    python3 eval_memory.py --preset ./experiment_presets/experiment_alice.json
"""

import sys, os, json, math, argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Memory counting
# ─────────────────────────────────────────────────────────────────────────────

def active_memory_size(history_manager):
    """Count items with non-empty detail (actual usable memory)."""
    return sum(1 for r in history_manager.detailed_history if r.get("detail", ""))


def total_ever_added(history_manager):
    """Count all items ever added (including removed ones) — the no-consensus baseline."""
    return len(history_manager.detailed_history)


def take_snapshot(server, round_num):
    """Capture memory sizes for every agent at the current point."""
    snap = {"round": round_num}
    for code in server.role_codes:
        hm = server.role_agents[code].history_manager
        snap[f"{code}_active"] = active_memory_size(hm)
        snap[f"{code}_total"] = total_ever_added(hm)
    whm = server.world_agent.memory
    snap["world_active"] = active_memory_size(whm)
    snap["world_total"] = total_ever_added(whm)

    snap["sum_active"] = sum(snap[f"{c}_active"] for c in server.role_codes) + snap["world_active"]
    snap["sum_total"] = sum(snap[f"{c}_total"] for c in server.role_codes)
    return snap


# ─────────────────────────────────────────────────────────────────────────────
# Run simulation
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(preset_path, llm_name, embedding_name,
                   rounds, consensus_threshold):
    from BookWorld import BookWorld

    print(f"\n  Initialising BookWorld …")
    bw = BookWorld(
        preset_path=preset_path,
        world_llm_name=llm_name,
        role_llm_name=llm_name,
        embedding_name=embedding_name,
        memory_top_k=5,
        consensus_threshold=consensus_threshold,
    )
    bw.set_generator(
        rounds=rounds,
        save_dir="",
        if_save=0,
        mode="free",
        scene_mode=0,
    )

    server = bw.server
    role_codes = list(server.role_codes)
    snapshots = []
    prev_round = -1

    print(f"  Running {rounds} rounds  "
          f"(agents: {len(role_codes)}, consensus_threshold: {consensus_threshold}) …\n")

    try:
        for msg in bw.generator:
            cur = server.cur_round
            if cur != prev_round:
                if prev_round >= 0:
                    snap = take_snapshot(server, prev_round + 1)
                    snapshots.append(snap)
                    a_act = [snap[f"{c}_active"] for c in role_codes]
                    a_tot = [snap[f"{c}_total"] for c in role_codes]
                    print(f"    Round {prev_round+1:>2}  │  "
                          f"active(role): {a_act}  world: {snap['world_active']}  "
                          f"sum: {snap['sum_active']}  │  "
                          f"no-consensus-equiv: {snap['sum_total']}")
                prev_round = cur
    except Exception as e:
        print(f"\n  ERROR during simulation at round {prev_round+1}: {e}")
        print(f"  Continuing with {len(snapshots)} complete rounds of data.\n")

    if prev_round >= 0:
        snap = take_snapshot(server, prev_round + 1)
        snapshots.append(snap)

    print(f"\n  Finished — {len(snapshots)} rounds recorded.")
    return snapshots, role_codes


def snapshots_to_series(snapshots, role_codes):
    """Convert list of snapshot dicts into series for plotting."""
    return {
        "rounds": [s["round"] for s in snapshots],
        "with_consensus": {
            "total": [s["sum_active"] for s in snapshots],
            "world": [s["world_active"] for s in snapshots],
            "agents": {c: [s[f"{c}_active"] for s in snapshots] for c in role_codes},
        },
        "without_consensus": {
            "total": [s["sum_total"] for s in snapshots],
            "world": [0] * len(snapshots),     # no world memory without consensus
            "agents": {c: [s[f"{c}_total"] for s in snapshots] for c in role_codes},
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_total_memory(series, save_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    rounds = series["rounds"]
    ax.plot(rounds, series["without_consensus"]["total"],
            "o-", color="#e74c3c", label="Without Consensus", linewidth=2, markersize=5)
    ax.plot(rounds, series["with_consensus"]["total"],
            "s-", color="#2ecc71", label="With Consensus", linewidth=2, markersize=5)
    ax.set_xlabel("Round", fontsize=13)
    ax.set_ylabel("Total Memory Size (items)", fontsize=13)
    ax.set_title("Total Memory Size Across All Agents", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, "total_memory_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_world_memory(series, save_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    rounds = series["rounds"]
    ax.plot(rounds, series["without_consensus"]["world"],
            "o-", color="#e74c3c", label="Without Consensus (no world memory)",
            linewidth=2, markersize=5)
    ax.plot(rounds, series["with_consensus"]["world"],
            "s-", color="#2ecc71", label="With Consensus",
            linewidth=2, markersize=5)
    ax.set_xlabel("Round", fontsize=13)
    ax.set_ylabel("World Memory Size (items)", fontsize=13)
    ax.set_title("World Agent Global Memory Size", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, "world_memory_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_per_agent(series, role_codes, save_dir):
    n = len(role_codes) + 1
    cols = min(4, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    all_keys = list(role_codes) + ["world"]
    rounds = series["rounds"]
    for i, code in enumerate(all_keys):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        if code != "world":
            no_vals = series["without_consensus"]["agents"][code]
            yes_vals = series["with_consensus"]["agents"][code]
        else:
            no_vals = series["without_consensus"]["world"]
            yes_vals = series["with_consensus"]["world"]
        label = code.split("-")[0] if "-" in code else code
        ax.plot(rounds, no_vals,
                "o-", color="#e74c3c", label="Without Consensus", linewidth=1.5, markersize=3)
        ax.plot(rounds, yes_vals,
                "s-", color="#2ecc71", label="With Consensus", linewidth=1.5, markersize=3)
        ax.set_title(label.replace("_", " "), fontsize=11, fontweight="bold")
        ax.set_xlabel("Round", fontsize=9)
        ax.set_ylabel("Memory Size", fontsize=9)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)

    for j in range(i + 1, rows * cols):
        r, c = divmod(j, cols)
        axes[r][c].set_visible(False)

    fig.suptitle("Per-Agent Memory Size Over Rounds",
                 fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = os.path.join(save_dir, "per_agent_memory_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_final_distribution(series, role_codes, save_dir):
    labels = [c.split("-")[0].replace("_", " ") for c in role_codes] + ["World"]
    no_vals = [series["without_consensus"]["agents"][c][-1] for c in role_codes] \
              + [series["without_consensus"]["world"][-1]]
    yes_vals = [series["with_consensus"]["agents"][c][-1] for c in role_codes] \
               + [series["with_consensus"]["world"][-1]]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.5), 5))
    bars1 = ax.bar(x - width / 2, no_vals, width,
                   label="Without Consensus", color="#e74c3c", alpha=0.85)
    bars2 = ax.bar(x + width / 2, yes_vals, width,
                   label="With Consensus", color="#2ecc71", alpha=0.85)
    ax.set_xlabel("Agent", fontsize=13)
    ax.set_ylabel("Final Memory Size (items)", fontsize=13)
    ax.set_title("Final Memory Size Distribution by Agent",
                 fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend(fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f"{int(h)}", xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    path = os.path.join(save_dir, "final_memory_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def print_summary(series, role_codes, rounds_count):
    w = series["with_consensus"]
    wo = series["without_consensus"]
    print(f"\n{'=' * 76}")
    print(f"  MEMORY SIZE SUMMARY  (after {rounds_count} rounds)")
    print(f"  'Without Consensus' = total items ever added (no filtering/removal)")
    print(f"  'With Consensus'    = actual active items (after consensus filtering)")
    print(f"{'=' * 76}")
    print(f"  {'Agent':<28} {'Without Consensus':>18} {'With Consensus':>16} {'Reduction':>14}")
    print(f"  {'-' * 72}")

    for code in role_codes:
        nc = wo["agents"][code][-1]
        wc = w["agents"][code][-1]
        diff = nc - wc
        pct = (diff / nc * 100) if nc > 0 else 0
        name = code.split("-")[0].replace("_", " ")
        print(f"  {name:<28} {nc:>18} {wc:>16} {diff:>7} ({pct:.1f}%)")

    nc_w = wo["world"][-1]
    wc_w = w["world"][-1]
    print(f"  {'World (global)':<28} {nc_w:>18} {wc_w:>16}     +{wc_w} (new)")
    print(f"  {'-' * 72}")
    nc_t = wo["total"][-1]
    wc_t = w["total"][-1]
    diff_t = nc_t - wc_t
    pct_t = (diff_t / nc_t * 100) if nc_t > 0 else 0
    print(f"  {'TOTAL':<28} {nc_t:>18} {wc_t:>16} {diff_t:>7} ({pct_t:.1f}%)")
    print(f"{'=' * 76}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Memory evaluation: consensus effect analysis (real LLM simulation)")
    parser.add_argument("--preset", type=str,
                        default="./experiment_presets/experiment_alice.json",
                        help="Preset JSON path")
    parser.add_argument("--llm", type=str, default="gpt-4o-mini",
                        help="LLM model name for both world and role agents")
    parser.add_argument("--embedding", type=str, default="openai",
                        help="Embedding model name")
    parser.add_argument("--rounds", type=int, default=5,
                        help="Number of simulation rounds")
    parser.add_argument("--threshold", type=int, default=10,
                        help="Consensus trigger threshold")
    parser.add_argument("--save_dir", type=str, default="./eval_memory_results",
                        help="Output directory for plots & data")
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

    print(f"\n{'─' * 68}")
    print(f"  Memory Evaluation  —  Real LLM Simulation (single-run method)")
    print(f"  Preset   : {args.preset}")
    print(f"  LLM      : {args.llm}")
    print(f"  Embed    : {args.embedding}")
    print(f"  Rounds   : {args.rounds}")
    print(f"  Threshold: {args.threshold}")
    print(f"{'─' * 68}")

    snapshots, role_codes = run_simulation(
        preset_path=os.path.abspath(args.preset),
        llm_name=args.llm,
        embedding_name=args.embedding,
        rounds=args.rounds,
        consensus_threshold=args.threshold,
    )

    if not snapshots:
        print("\n  No data collected. Exiting.")
        return

    series = snapshots_to_series(snapshots, role_codes)

    print_summary(series, role_codes, len(snapshots))

    print(f"\nGenerating plots → {args.save_dir}/")
    plot_total_memory(series, args.save_dir)
    plot_world_memory(series, args.save_dir)
    plot_per_agent(series, role_codes, args.save_dir)
    plot_final_distribution(series, role_codes, args.save_dir)

    summary = {
        "preset": args.preset,
        "llm": args.llm,
        "rounds": len(snapshots),
        "consensus_threshold": args.threshold,
        "role_codes": role_codes,
        "series": series,
        "timestamp": datetime.now().isoformat(),
    }
    summary_path = os.path.join(args.save_dir, "eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {summary_path}")

    print(f"\n  Evaluation complete. All outputs in: {args.save_dir}/\n")


if __name__ == "__main__":
    main()
