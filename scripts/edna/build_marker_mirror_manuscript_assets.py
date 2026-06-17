#!/usr/bin/env python3
"""Build MarkerMirror manuscript-facing tables from source artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from progress_logging import ProgressLogger, default_log_path


ROOT = Path(__file__).resolve().parents[2]
PAPER1 = ROOT / "results" / "paper1_phylo_calibrated_assignment"
SOURCE = PAPER1 / "source_tables"
REMOTE = ROOT / "results" / "remote_runs" / "2026-06-03" / "rtx_pro_6000"
FULL_RUN = REMOTE / "marker_mirror_12s_production_v1" / "vast_full_all_queries_20260603"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE)
    parser.add_argument("--full-run-dir", type=Path, default=FULL_RUN)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PAPER1 / "manuscript_assets" / "marker_mirror",
    )
    parser.add_argument("--log-file", type=Path, default=None)
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def fmt_pct(value: Any) -> float:
    return round(float(value), 2)


def candidate_support(source_dir: Path) -> pd.DataFrame:
    blast = read_csv(source_dir / "marker_mirror_union_blast_candidate_support_summary.csv")
    vsearch = read_csv(source_dir / "marker_mirror_union_vsearch_candidate_support_summary.csv")
    combined = pd.concat([blast, vsearch], ignore_index=True)
    labels = {
        "marker_mirror_12s_to_16s": "MarkerMirror 12S->16S only",
        "same_marker_12s_blastn_local": "Same-marker 12S BLASTN",
        "union_marker_mirror_plus_blastn": "MarkerMirror + BLASTN union",
        "same_marker_12s_vsearch_global": "Same-marker 12S VSEARCH",
        "union_marker_mirror_plus_vsearch": "MarkerMirror + VSEARCH union",
    }
    combined = combined[combined["candidate_source"].isin(labels)].copy()
    combined["display_label"] = combined["candidate_source"].map(labels)
    combined = combined.drop_duplicates("display_label", keep="first")
    return combined[
        [
            "display_label",
            "top_k",
            "n_queries",
            "species_hit_pct",
            "genus_hit_pct",
            "family_hit_pct",
            "order_hit_pct",
        ]
    ].rename(
        columns={
            "display_label": "candidate_source",
            "species_hit_pct": "species_pct",
            "genus_hit_pct": "genus_pct",
            "family_hit_pct": "family_pct",
            "order_hit_pct": "order_pct",
        }
    )


def order_policies(source_dir: Path) -> pd.DataFrame:
    stable = read_csv(source_dir / "marker_mirror_stable_order_policy_summary.csv")
    stable = stable[stable["policy"].str.contains("max_repeat_threshold", na=False)].iloc[0]
    high = read_csv(source_dir / "marker_mirror_high_coverage_order_repair_assignment_summary.csv").iloc[0]
    high_nested = read_csv(source_dir / "marker_mirror_high_coverage_order_repair_summary.csv").iloc[0]
    rows = [
        {
            "mode": "stable_order",
            "status": "default conservative mode",
            "n_queries": int(stable["n_queries"]),
            "n_assigned": int(stable["n_assigned"]),
            "coverage_pct": fmt_pct(stable["coverage_pct"]),
            "diagnostic_precision_pct": fmt_pct(stable["assigned_precision_pct"]),
            "false_species_call_rate_pct": fmt_pct(stable.get("false_species_call_rate_pct", 0.0)),
            "nested_mean_heldout_coverage_pct": "",
            "nested_mean_heldout_precision_pct": "",
            "target_met_rate_pct": 100.0,
            "claim_boundary": "Order/no-call only; conservative default.",
        },
        {
            "mode": "high_coverage_order",
            "status": "explicit diagnostic/research mode",
            "n_queries": int(high["n_queries"]),
            "n_assigned": int(high["n_assigned"]),
            "coverage_pct": fmt_pct(high["coverage_pct"]),
            "diagnostic_precision_pct": fmt_pct(high["assigned_precision_pct"]),
            "false_species_call_rate_pct": 0.0,
            "nested_mean_heldout_coverage_pct": fmt_pct(high_nested["mean_coverage_pct"]),
            "nested_mean_heldout_precision_pct": fmt_pct(float(high_nested["mean_assigned_precision"]) * 100.0),
            "target_met_rate_pct": fmt_pct(high_nested["target_met_rate_pct"]),
            "claim_boundary": "Order/no-call only; not default; not species identification.",
        },
    ]
    return pd.DataFrame(rows)


def rank_boundaries(source_dir: Path) -> pd.DataFrame:
    repair = read_csv(source_dir / "marker_mirror_high_coverage_rank_repair_comparison.csv")
    sets = read_csv(source_dir / "marker_mirror_hierarchical_candidate_sets_policy_grid_summary.csv")
    rows: list[dict[str, Any]] = []
    for row in repair.itertuples(index=False):
        rows.append(
            {
                "rank": row.rank,
                "single_label_best_coverage_pct": fmt_pct(row.best_mean_coverage_pct),
                "single_label_best_precision_pct": fmt_pct(float(row.best_mean_assigned_precision) * 100.0),
                "single_label_target_met_rate_pct": fmt_pct(row.best_target_met_rate_pct),
                "single_label_recommendation": row.recommendation,
            }
        )
    frame = pd.DataFrame(rows)
    set_rows: list[dict[str, Any]] = []
    for rank, group in sets.groupby("rank", sort=False):
        best = group.sort_values(
            ["full_query_truth_coverage_pct", "mean_set_size"],
            ascending=[False, True],
        ).iloc[0]
        set_rows.append(
            {
                "rank": rank,
                "best_set_policy": best["policy"],
                "best_set_top_k": int(best["top_k"]),
                "best_set_full_query_truth_coverage_pct": fmt_pct(
                    best["full_query_truth_coverage_pct"]
                ),
                "best_set_mean_size": fmt_pct(best["mean_set_size"]),
                "best_set_p90_size": fmt_pct(best["p90_set_size"]),
            }
        )
    merged = frame.merge(pd.DataFrame(set_rows), on="rank", how="left")
    merged["manuscript_interpretation"] = merged["rank"].map(
        {
            "order": "Defensible order/no-call mode; high-coverage order remains explicit diagnostic.",
            "family": "Do not enable; single-label and set-valued outputs fail target-0.99.",
            "genus": "Do not enable; current evidence is insufficient for useful target-0.99 output.",
        }
    )
    return merged


def runtime_table(full_run_dir: Path) -> pd.DataFrame:
    manifest_path = full_run_dir / "marker_mirror_12s_production_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    timings = manifest.get("timings", {})
    labels = {
        "marker_mirror_seconds": "MarkerMirror candidate generation",
        "blast_seconds": "BLASTN same-marker search",
        "vsearch_seconds": "VSEARCH same-marker search",
        "stable_policy_seconds": "Stable order/no-call policy",
    }
    rows = [
        {"stage": labels[key], "seconds": round(float(value), 2)}
        for key, value in timings.items()
        if key in labels
    ]
    total = sum(row["seconds"] for row in rows)
    for row in rows:
        row["percent_runtime"] = round(100.0 * row["seconds"] / total, 2) if total else 0.0
    return pd.DataFrame(rows)


def figure_plan() -> pd.DataFrame:
    rows = [
        {
            "figure": "MarkerMirror Fig 1",
            "panel": "A",
            "title": "12S order/no-call pipeline",
            "source_table": "marker_mirror_runtime_table.csv",
            "visual": "pipeline schematic with runtime callouts",
            "message": "The tool is an evidence compiler, not a forced species classifier.",
        },
        {
            "figure": "MarkerMirror Fig 1",
            "panel": "B",
            "title": "Candidate support by source",
            "source_table": "marker_mirror_candidate_support_table.csv",
            "visual": "grouped bar chart for species/genus/family/order",
            "message": "BLASTN/VSEARCH same-marker evidence rescues high-rank support; species stays blocked.",
        },
        {
            "figure": "MarkerMirror Fig 2",
            "panel": "A",
            "title": "Stable versus high-coverage order mode",
            "source_table": "marker_mirror_order_policy_table.csv",
            "visual": "coverage-precision tradeoff plot",
            "message": "High-confidence order calls are possible with explicit abstention.",
        },
        {
            "figure": "MarkerMirror Fig 2",
            "panel": "B",
            "title": "Why family/genus are not enabled",
            "source_table": "marker_mirror_rank_boundary_table.csv",
            "visual": "rank boundary table or compact heatmap",
            "message": "The limitation is evidence-level, not merely thresholding.",
        },
    ]
    return pd.DataFrame(rows)


def methods_blurb() -> str:
    return """# MarkerMirror Methods Blurb

MarkerMirror evaluates 12S query fragments with a three-arm candidate generator:
a learned 12S-to-16S retrieval model, BLASTN same-marker 12S retrieval, and
VSEARCH same-marker 12S retrieval. Candidate lists are merged into
production-available evidence tables containing per-source top-k taxonomic
support. The default `stable_order` mode emits an order call only when
MarkerMirror, BLASTN, and VSEARCH agree on top-1 order and the learned
threshold is met; otherwise the query receives a no-call reason code. The
explicit `high_coverage_order` mode uses nested species-split calibration over
BLASTN/VSEARCH top-10 order agreement to increase order-call coverage while
remaining order/no-call only. Family and genus decisions are disabled because
neither single-label calibration nor set-valued candidate output met the
target-0.99 transfer criterion under repeated species-split validation.
"""


def main() -> None:
    args = parse_args()
    logger = ProgressLogger(args.log_file or default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).stem)
    logger.log(f"source_dir={rel(args.source_dir)}")
    logger.log(f"full_run_dir={rel(args.full_run_dir)}")
    logger.log(f"output_dir={rel(args.output_dir)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "candidate_support": args.output_dir / "marker_mirror_candidate_support_table.csv",
        "order_policy": args.output_dir / "marker_mirror_order_policy_table.csv",
        "rank_boundary": args.output_dir / "marker_mirror_rank_boundary_table.csv",
        "runtime": args.output_dir / "marker_mirror_runtime_table.csv",
        "figure_plan": args.output_dir / "marker_mirror_figure_plan.csv",
        "methods_blurb": args.output_dir / "marker_mirror_methods_blurb.md",
        "manifest": args.output_dir / "marker_mirror_manuscript_asset_manifest.json",
    }
    candidate = candidate_support(args.source_dir)
    candidate.to_csv(outputs["candidate_support"], index=False)
    logger.log(f"wrote {rel(outputs['candidate_support'])} rows={len(candidate)}")
    order = order_policies(args.source_dir)
    order.to_csv(outputs["order_policy"], index=False)
    logger.log(f"wrote {rel(outputs['order_policy'])} rows={len(order)}")
    ranks = rank_boundaries(args.source_dir)
    ranks.to_csv(outputs["rank_boundary"], index=False)
    logger.log(f"wrote {rel(outputs['rank_boundary'])} rows={len(ranks)}")
    runtime = runtime_table(args.full_run_dir)
    runtime.to_csv(outputs["runtime"], index=False)
    logger.log(f"wrote {rel(outputs['runtime'])} rows={len(runtime)}")
    figure_plan().to_csv(outputs["figure_plan"], index=False)
    logger.log(f"wrote {rel(outputs['figure_plan'])}")
    outputs["methods_blurb"].write_text(methods_blurb(), encoding="utf-8")
    logger.log(f"wrote {rel(outputs['methods_blurb'])}")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": rel(Path(__file__)),
        "inputs": {
            "source_dir": rel(args.source_dir),
            "full_run_dir": rel(args.full_run_dir),
        },
        "outputs": {key: rel(path) for key, path in outputs.items() if key != "manifest"},
        "claim_boundary": "Manuscript-facing planning assets only. These are not new metrics.",
    }
    outputs["manifest"].write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.log(f"wrote {rel(outputs['manifest'])}")
    logger.done(Path(__file__).stem)


if __name__ == "__main__":
    main()
