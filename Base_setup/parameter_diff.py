"""
parameter_diff.py — export a ranked, deduplicated parameter table to JSON for Typst import.

Reads a saved bounds_search.json or single-run .json, filters out near-duplicate
solutions (runs whose parameters are all within `threshold` of an already-accepted
run), and writes the surviving results to a compact JSON file.

The output is designed for `json("file.json")` in Typst:

    #let data = json("parameter_diff.json")
    #for run in data.runs { ... run.rank, run.cost, run.params.L ... }

Usage (from Base_setup/):
    python parameter_diff.py plots/20260501_123456_run/bounds_search.json
    python parameter_diff.py plots/.../bounds_search.json --top_n 5
    python parameter_diff.py plots/.../bounds_search.json --threshold 0.02
    python parameter_diff.py plots/.../bounds_search.json --threshold 0 --top_n 10
    python parameter_diff.py plots/.../bounds_search.json --out my_table.json
"""

import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_results(json_path: Path) -> tuple[list[dict], list[str]]:
    """Return (results, param_names) from a bounds_search or single-run JSON.

    Results are sorted by ascending cost. Each result dict has keys:
        rank (int), label (str), cost (float), params (dict[str, float])
    """
    raw = json.loads(json_path.read_text())
    results_raw = raw["results"]

    if isinstance(results_raw, list):
        # bounds_search format — already ranked by cost
        entries = results_raw
        first_p = entries[0]["params"]
        param_names = list(first_p.keys()) if isinstance(first_p, dict) else None
        results = []
        for e in entries:
            p = e["params"]
            params = dict(p) if isinstance(p, dict) else dict(zip(param_names or [], p))
            results.append({"rank": e["rank"], "label": e["label"],
                            "cost": e["cost"], "r2": e.get("r2"), "params": params})
    else:
        # single-run format — dict of optimizer_name -> result
        param_names = None
        results = []
        for label, res in results_raw.items():
            p = res["params"]
            if isinstance(p, dict):
                param_names = list(p.keys())
                params = dict(p)
            else:
                params = {str(i): v for i, v in enumerate(p)}
            results.append({"rank": 0, "label": label, "cost": res["cost"], "r2": res.get("r2"), "params": params})
        results.sort(key=lambda r: r["cost"])
        for i, r in enumerate(results, 1):
            r["rank"] = i

    if param_names is None:
        param_names = list(results[0]["params"].keys())

    return results, param_names


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def _param_ranges(results: list[dict], param_names: list[str]) -> dict[str, float]:
    """Max − min of each parameter across all results; returns 1.0 if zero span."""
    ranges = {}
    for name in param_names:
        vals = [r["params"][name] for r in results]
        span = max(vals) - min(vals)
        ranges[name] = span if span > 0 else 1.0
    return ranges


def _max_norm_diff(a: dict, b: dict, param_names: list[str],
                   ranges: dict[str, float]) -> float:
    """Largest normalised absolute parameter difference between two results."""
    return max(abs(a["params"][n] - b["params"][n]) / ranges[n] for n in param_names)


def filter_by_difference(
    results: list[dict],
    param_names: list[str],
    threshold: float,
    top_n: int | None,
) -> list[dict]:
    """Return results that are each sufficiently different from all accepted ones.

    A candidate is kept if, for every already-accepted run, at least one
    parameter differs by more than `threshold` (normalised by the observed
    range across all results). Setting threshold=0 disables filtering.

    Args:
        results:     Runs sorted by ascending cost.
        param_names: Ordered list of parameter names.
        threshold:   Minimum normalised difference required (0 = keep all).
        top_n:       Stop once this many runs are accepted (None = no limit).

    Returns:
        Accepted runs in cost order.
    """
    if threshold == 0:
        return results[:top_n] if top_n is not None else list(results)

    ranges = _param_ranges(results, param_names)
    accepted: list[dict] = []

    for r in results:
        is_dup = any(
            _max_norm_diff(r, prev, param_names, ranges) < threshold
            for prev in accepted
        )
        if not is_dup:
            accepted.append(r)
        if top_n is not None and len(accepted) >= top_n:
            break

    return accepted


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def build_output(
    accepted: list[dict],
    param_names: list[str],
    source_path: Path,
    threshold: float,
    total_runs: int,
) -> dict:
    """Build the JSON dict ready for Typst import."""
    runs = []
    for i, r in enumerate(accepted, 1):
        runs.append({
            "rank": r["rank"],
            "label": r["label"],
            "cost": round(r["cost"], 8),
            "r2": round(r["r2"], 6) if r.get("r2") is not None else None,
            "params": {k: round(v, 6) for k, v in r["params"].items()},
        })
    return {
        "param_names": param_names,
        "runs": runs,
        "meta": {
            "source": source_path.name,
            "threshold": threshold,
            "total_runs": total_runs,
            "exported_runs": len(runs),
        },
    }


# ---------------------------------------------------------------------------
# Interactive runner (mirrors example_replot.py pattern)
# ---------------------------------------------------------------------------

PLOTS_DIR = Path("plots")


def list_runs(plots_dir: Path) -> list[Path]:
    """Return all .json files found recursively under plots_dir, sorted by name."""
    return sorted(plots_dir.rglob("*.json"))


def _prompt_float(prompt: str, default: float) -> float:
    raw = input(f"{prompt} (Enter for {default}): ").strip()
    return float(raw) if raw else default


def _prompt_int_or_none(prompt: str) -> int | None:
    raw = input(f"{prompt} (Enter for all): ").strip()
    return int(raw) if raw else None


if __name__ == "__main__":
    import sys

    if not PLOTS_DIR.exists():
        print("No plots folder found — run main.py or single_run.py first.")
        sys.exit(1)

    runs = list_runs(PLOTS_DIR)
    if not runs:
        print("No saved runs found in 'plots/' — run main.py or single_run.py first.")
        sys.exit(1)

    print("Available saved runs:")
    for i, p in enumerate(runs):
        print(f"  [{i}] {p.relative_to(PLOTS_DIR)}")

    raw = input("\nEnter number to load (or paste a full path): ").strip()
    try:
        json_path = runs[int(raw)]
    except (ValueError, IndexError):
        json_path = Path(raw)

    if not json_path.exists():
        print(f"File not found: {json_path}")
        sys.exit(1)

    print(
        "\nDifference threshold: fraction of each parameter's observed range.\n"
        "  A run is treated as a near-duplicate and skipped if every parameter\n"
        "  differs by less than this fraction from an already-accepted run.\n"
        "  0.05 = 5% of range  |  0 = no filtering, export all\n"
        "  Example: if mu_s spans 1.0–4.0 across all runs, threshold 0.05 means\n"
        "           runs within 0.15 of each other on mu_s (and similar for every\n"
        "           other param) are considered duplicates.\n"
    )
    threshold = _prompt_float("Threshold", 0.05)
    top_n     = _prompt_int_or_none("Max runs to export")

    save_name = input("Output filename (Enter for 'parameter_diff'): ").strip()
    if not save_name:
        save_name = "parameter_diff"
    if not save_name.endswith(".json"):
        save_name += ".json"
    out_path = json_path.parent / save_name

    results, param_names = load_results(json_path)
    total = len(results)
    accepted = filter_by_difference(results, param_names, threshold, top_n)

    output = build_output(accepted, param_names, json_path, threshold, total)
    out_path.write_text(json.dumps(output, indent=2))

    print(f"\nLoaded {total} runs → kept {len(accepted)} (threshold={threshold})")
    print(f"Saved: {out_path}")
    for r in accepted:
        params_str = "  ".join(f"{k}={v:.4f}" for k, v in r["params"].items())
        print(f"  rank {r['rank']:>3}  cost={r['cost']:.6e}  {params_str}")
