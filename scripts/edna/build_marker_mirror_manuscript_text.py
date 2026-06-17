#!/usr/bin/env python3
"""Build MarkerMirror manuscript captions and text snippets from assets."""

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
ASSETS = PAPER1 / "manuscript_assets" / "marker_mirror"
FIGURES = ASSETS / "figures"
SLIDES = ASSETS / "slide_tables"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", type=Path, default=ASSETS)
    parser.add_argument("--figure-dir", type=Path, default=FIGURES)
    parser.add_argument("--slide-dir", type=Path, default=SLIDES)
    parser.add_argument("--output-dir", type=Path, default=ASSETS / "text")
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


def pct(value: Any, digits: int = 1) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}%"


def row_by(df: pd.DataFrame, column: str, value: str) -> pd.Series:
    matches = df[df[column] == value]
    if matches.empty:
        raise ValueError(f"No row where {column}={value!r}")
    return matches.iloc[0]


def build_context(asset_dir: Path) -> dict[str, Any]:
    support = read_csv(asset_dir / "marker_mirror_candidate_support_table.csv")
    policy = read_csv(asset_dir / "marker_mirror_order_policy_table.csv")
    ranks = read_csv(asset_dir / "marker_mirror_rank_boundary_table.csv")
    runtime = read_csv(asset_dir / "marker_mirror_runtime_table.csv")
    mm = row_by(support, "candidate_source", "MarkerMirror 12S->16S only")
    blast = row_by(support, "candidate_source", "Same-marker 12S BLASTN")
    blast_union = row_by(support, "candidate_source", "MarkerMirror + BLASTN union")
    vsearch = row_by(support, "candidate_source", "Same-marker 12S VSEARCH")
    vsearch_union = row_by(support, "candidate_source", "MarkerMirror + VSEARCH union")
    stable = row_by(policy, "mode", "stable_order")
    high = row_by(policy, "mode", "high_coverage_order")
    order = row_by(ranks, "rank", "order")
    family = row_by(ranks, "rank", "family")
    genus = row_by(ranks, "rank", "genus")
    total_seconds = float(runtime["seconds"].sum())
    blast_runtime = row_by(runtime, "stage", "BLASTN same-marker search")
    vsearch_runtime = row_by(runtime, "stage", "VSEARCH same-marker search")
    mm_runtime = row_by(runtime, "stage", "MarkerMirror candidate generation")
    policy_runtime = row_by(runtime, "stage", "Stable order/no-call policy")
    return {
        "support": support,
        "policy": policy,
        "ranks": ranks,
        "runtime": runtime,
        "mm": mm,
        "blast": blast,
        "blast_union": blast_union,
        "vsearch": vsearch,
        "vsearch_union": vsearch_union,
        "stable": stable,
        "high": high,
        "order": order,
        "family": family,
        "genus": genus,
        "total_seconds": total_seconds,
        "blast_runtime": blast_runtime,
        "vsearch_runtime": vsearch_runtime,
        "mm_runtime": mm_runtime,
        "policy_runtime": policy_runtime,
    }


def figure_captions(ctx: dict[str, Any], figure_dir: Path) -> str:
    return f"""# MarkerMirror Figure Captions

## Candidate Support

**Figure MM1. Candidate support from learned cross-marker and classical
same-marker evidence.** Bar heights show whether the true taxon was present
among the top-50 candidates for 3,566 12S query sequences. MarkerMirror
12S->16S alone recovered species/genus/family/order support of
{pct(ctx['mm'].species_pct)}, {pct(ctx['mm'].genus_pct)},
{pct(ctx['mm'].family_pct)}, and {pct(ctx['mm'].order_pct)}, respectively.
Same-marker BLASTN recovered {pct(ctx['blast'].genus_pct)} genus,
{pct(ctx['blast'].family_pct)} family, and {pct(ctx['blast'].order_pct)}
order support, while the MarkerMirror + BLASTN union reached
{pct(ctx['blast_union'].species_pct)}, {pct(ctx['blast_union'].genus_pct)},
{pct(ctx['blast_union'].family_pct)}, and {pct(ctx['blast_union'].order_pct)}
for species/genus/family/order. VSEARCH independently confirmed the same
high-rank pattern. Species support remains low because the held-out query
species are absent from the current same-marker 12S reference by split design.

Source figure:
`{rel(figure_dir / "marker_mirror_candidate_support_bars.png")}`

## Order/No-Call Policy

**Figure MM2. Coverage-precision tradeoff for order/no-call output.** The
default `stable_order` policy emitted {int(ctx['stable'].n_assigned):,} order
calls from {int(ctx['stable'].n_queries):,} queries
({pct(ctx['stable'].coverage_pct)} coverage) at
{pct(ctx['stable'].diagnostic_precision_pct)} diagnostic precision. The
explicit `high_coverage_order` research mode emitted
{int(ctx['high'].n_assigned):,} order calls
({pct(ctx['high'].coverage_pct)} full-table coverage) while nested repeated
species-split validation averaged {pct(ctx['high'].nested_mean_heldout_coverage_pct)}
held-out coverage at {pct(ctx['high'].nested_mean_heldout_precision_pct)}
assigned precision. Both modes are order/no-call only.

Source figure:
`{rel(figure_dir / "marker_mirror_order_policy_tradeoff.png")}`

## Rank Boundary

**Figure MM3. Family and genus remain disabled under current evidence.** The
best order repair met target-0.99 in
{pct(ctx['order'].single_label_target_met_rate_pct)} of repeated species-split
runs, but the best family and genus repairs met the target in only
{pct(ctx['family'].single_label_target_met_rate_pct)} and
{pct(ctx['genus'].single_label_target_met_rate_pct)} of repeats, respectively.
Set-valued output did not solve the issue: the best family set covered
{pct(ctx['family'].best_set_full_query_truth_coverage_pct)} of full-query truth
labels but required a mean set size of {float(ctx['family'].best_set_mean_size):.1f}
families, and the best genus set covered
{pct(ctx['genus'].best_set_full_query_truth_coverage_pct)} with a mean set size
of {float(ctx['genus'].best_set_mean_size):.1f} genera. This supports an
order/no-call claim boundary.

Source figure:
`{rel(figure_dir / "marker_mirror_rank_boundary.png")}`

## Runtime

**Figure MM4. Runtime of the executable 12S wrapper on the Vast RTX host.** The
full run processed 3,566 12S queries in {ctx['total_seconds']:.1f} seconds.
MarkerMirror candidate generation took {float(ctx['mm_runtime'].seconds):.1f}
seconds, BLASTN took {float(ctx['blast_runtime'].seconds):.1f} seconds,
VSEARCH took {float(ctx['vsearch_runtime'].seconds):.1f} seconds, and the
stable order/no-call policy took {float(ctx['policy_runtime'].seconds):.1f}
seconds. BLASTN dominated total runtime.

Source figure:
`{rel(figure_dir / "marker_mirror_runtime_breakdown.png")}`
"""


def results_paragraph(ctx: dict[str, Any]) -> str:
    return f"""# MarkerMirror Results Paragraph

On the full 12S query set, MarkerMirror alone recovered the true
species/genus/family/order among the top-50 16S candidates at
{pct(ctx['mm'].species_pct)}, {pct(ctx['mm'].genus_pct)},
{pct(ctx['mm'].family_pct)}, and {pct(ctx['mm'].order_pct)}, respectively.
Classical same-marker evidence provided strong high-rank support: BLASTN
top-50 candidate support was {pct(ctx['blast'].species_pct)},
{pct(ctx['blast'].genus_pct)}, {pct(ctx['blast'].family_pct)}, and
{pct(ctx['blast'].order_pct)}, while VSEARCH support was
{pct(ctx['vsearch'].species_pct)}, {pct(ctx['vsearch'].genus_pct)},
{pct(ctx['vsearch'].family_pct)}, and {pct(ctx['vsearch'].order_pct)}. The
MarkerMirror + BLASTN union reached {pct(ctx['blast_union'].species_pct)},
{pct(ctx['blast_union'].genus_pct)}, {pct(ctx['blast_union'].family_pct)}, and
{pct(ctx['blast_union'].order_pct)} for species/genus/family/order. A
conservative all-source order/no-call policy assigned {int(ctx['stable'].n_assigned):,}
of {int(ctx['stable'].n_queries):,} queries ({pct(ctx['stable'].coverage_pct)}
coverage) at {pct(ctx['stable'].diagnostic_precision_pct)} diagnostic
precision. An explicit high-coverage order-only diagnostic assigned
{int(ctx['high'].n_assigned):,} queries ({pct(ctx['high'].coverage_pct)}
full-table coverage) and achieved {pct(ctx['high'].nested_mean_heldout_precision_pct)}
mean held-out precision in nested repeated species-split validation. Family,
genus, and species calls remain disabled because neither single-label nor
set-valued family/genus output met the target-0.99 transfer criterion under the
current evidence.
"""


def methods_paragraph(ctx: dict[str, Any]) -> str:
    return f"""# MarkerMirror Methods Paragraph

MarkerMirror was evaluated as a 12S order/no-call research pipeline rather than
a forced species classifier. For each 12S query, the wrapper generated three
candidate streams: learned cross-marker 12S->16S candidates, BLASTN same-marker
12S candidates, and VSEARCH same-marker 12S candidates. Candidate streams were
merged into production-available evidence tables containing per-source
taxonomic support and agreement features. The default `stable_order` policy
emitted an order call only when the evidence sources met the locked
all-source-agreement criterion; all other queries received a no-call with a
reason code. A separate `high_coverage_order` research mode used nested
species-split calibration over BLASTN/VSEARCH top-10 order agreement to
increase order-call coverage while remaining order/no-call only. Family and
genus modes were not enabled because repeated species-split validation did not
meet the target-0.99 criterion for those ranks, and set-valued family/genus
outputs required large candidate sets that were not useful as final
assignments.

The full wrapper run used {int(ctx['stable'].n_queries):,} 12S query sequences
and completed in {ctx['total_seconds']:.1f} seconds on the Vast RTX host. Stage
timings were {float(ctx['mm_runtime'].seconds):.1f} seconds for MarkerMirror
candidate generation, {float(ctx['blast_runtime'].seconds):.1f} seconds for
BLASTN, {float(ctx['vsearch_runtime'].seconds):.1f} seconds for VSEARCH, and
{float(ctx['policy_runtime'].seconds):.1f} seconds for the stable order/no-call
policy.
"""


def claim_boundary(ctx: dict[str, Any]) -> str:
    return f"""# MarkerMirror Claim Boundary Box

## Safe Claim

MarkerMirror combines learned cross-marker 12S->16S retrieval with BLASTN and
VSEARCH same-marker 12S evidence to make precise order/no-call decisions under
missing-reference conditions. The current conservative mode assigns
{int(ctx['stable'].n_assigned):,}/{int(ctx['stable'].n_queries):,} queries at
{pct(ctx['stable'].diagnostic_precision_pct)} diagnostic precision, and the
explicit high-coverage order diagnostic increases order-call coverage while
remaining order/no-call only.

## Do Not Claim

- Species-level 12S identification is solved.
- Family or genus calls are enabled.
- MarkerMirror replaces BLASTN or VSEARCH.
- These results are field-eDNA validation.
- The high-coverage mode is the conservative default.

## Why The Boundary Matters

The strongest result is not that a neural model beats classical search. The
strongest result is that learned cross-marker retrieval and classical alignment
evidence can be compiled into an auditable high-rank decision that abstains
when the evidence does not support a deeper taxonomic claim.
"""


def caption_inventory(output_dir: Path) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "figure": "MM1",
                "title": "Candidate support by evidence source",
                "text_file": rel(output_dir / "marker_mirror_figure_captions.md"),
                "figure_file": rel(FIGURES / "marker_mirror_candidate_support_bars.png"),
                "claim": "Union candidate generation recovers high-rank support.",
            },
            {
                "figure": "MM2",
                "title": "Order/no-call coverage-precision tradeoff",
                "text_file": rel(output_dir / "marker_mirror_figure_captions.md"),
                "figure_file": rel(FIGURES / "marker_mirror_order_policy_tradeoff.png"),
                "claim": "Order-level calls can be precise with explicit abstention.",
            },
            {
                "figure": "MM3",
                "title": "Rank boundary",
                "text_file": rel(output_dir / "marker_mirror_figure_captions.md"),
                "figure_file": rel(FIGURES / "marker_mirror_rank_boundary.png"),
                "claim": "Family/genus remain unsupported under current evidence.",
            },
            {
                "figure": "MM4",
                "title": "Runtime breakdown",
                "text_file": rel(output_dir / "marker_mirror_figure_captions.md"),
                "figure_file": rel(FIGURES / "marker_mirror_runtime_breakdown.png"),
                "claim": "The executable wrapper runs end to end; BLASTN dominates runtime.",
            },
        ]
    )


def write(path: Path, text: str, logger: ProgressLogger) -> None:
    path.write_text(text, encoding="utf-8")
    logger.log(f"wrote {rel(path)}")


def main() -> None:
    args = parse_args()
    logger = ProgressLogger(args.log_file or default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).stem)
    logger.log(f"asset_dir={rel(args.asset_dir)}")
    logger.log(f"figure_dir={rel(args.figure_dir)}")
    logger.log(f"slide_dir={rel(args.slide_dir)}")
    logger.log(f"output_dir={rel(args.output_dir)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ctx = build_context(args.asset_dir)
    outputs = {
        "figure_captions": args.output_dir / "marker_mirror_figure_captions.md",
        "results_paragraph": args.output_dir / "marker_mirror_results_paragraph.md",
        "methods_paragraph": args.output_dir / "marker_mirror_methods_paragraph.md",
        "claim_boundary": args.output_dir / "marker_mirror_claim_boundary_box.md",
        "caption_inventory": args.output_dir / "marker_mirror_caption_inventory.csv",
        "manifest": args.output_dir / "marker_mirror_manuscript_text_manifest.json",
    }
    write(outputs["figure_captions"], figure_captions(ctx, args.figure_dir), logger)
    write(outputs["results_paragraph"], results_paragraph(ctx), logger)
    write(outputs["methods_paragraph"], methods_paragraph(ctx), logger)
    write(outputs["claim_boundary"], claim_boundary(ctx), logger)
    inventory = caption_inventory(args.output_dir)
    inventory.to_csv(outputs["caption_inventory"], index=False)
    logger.log(f"wrote {rel(outputs['caption_inventory'])} rows={len(inventory)}")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": rel(Path(__file__)),
        "inputs": {
            "asset_dir": rel(args.asset_dir),
            "figure_dir": rel(args.figure_dir),
            "slide_dir": rel(args.slide_dir),
        },
        "outputs": {key: rel(path) for key, path in outputs.items() if key != "manifest"},
        "claim_boundary": "Manuscript text snippets only. They repackage existing MarkerMirror metrics and do not add new results.",
    }
    outputs["manifest"].write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.log(f"wrote {rel(outputs['manifest'])}")
    logger.done(Path(__file__).stem)


if __name__ == "__main__":
    main()
