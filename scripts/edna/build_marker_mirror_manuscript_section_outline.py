#!/usr/bin/env python3
"""Build a MarkerMirror manuscript section outline from text assets."""

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
TEXT = ASSETS / "text"
FIGURES = ASSETS / "figures"
SLIDES = ASSETS / "slide_tables"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", type=Path, default=ASSETS)
    parser.add_argument("--text-dir", type=Path, default=TEXT)
    parser.add_argument("--figure-dir", type=Path, default=FIGURES)
    parser.add_argument("--slide-dir", type=Path, default=SLIDES)
    parser.add_argument("--output-dir", type=Path, default=TEXT)
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


def pct(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.1f}%"


def row_by(df: pd.DataFrame, column: str, value: str) -> pd.Series:
    match = df[df[column] == value]
    if match.empty:
        raise ValueError(f"Missing row {column}={value!r}")
    return match.iloc[0]


def context(asset_dir: Path) -> dict[str, Any]:
    support = read_csv(asset_dir / "marker_mirror_candidate_support_table.csv")
    policy = read_csv(asset_dir / "marker_mirror_order_policy_table.csv")
    ranks = read_csv(asset_dir / "marker_mirror_rank_boundary_table.csv")
    runtime = read_csv(asset_dir / "marker_mirror_runtime_table.csv")
    return {
        "support": support,
        "policy": policy,
        "ranks": ranks,
        "runtime": runtime,
        "mm": row_by(support, "candidate_source", "MarkerMirror 12S->16S only"),
        "blast_union": row_by(support, "candidate_source", "MarkerMirror + BLASTN union"),
        "vsearch_union": row_by(support, "candidate_source", "MarkerMirror + VSEARCH union"),
        "stable": row_by(policy, "mode", "stable_order"),
        "high": row_by(policy, "mode", "high_coverage_order"),
        "order": row_by(ranks, "rank", "order"),
        "family": row_by(ranks, "rank", "family"),
        "genus": row_by(ranks, "rank", "genus"),
        "total_seconds": float(runtime["seconds"].sum()),
    }


def build_outline(ctx: dict[str, Any], text_dir: Path, figure_dir: Path, slide_dir: Path) -> str:
    return f"""# MarkerMirror Manuscript Section Outline

## Placement In Paper 1

Use this as a focused 12S/eDNA extension section after the COI/vector/retrieval
and strict missing-reference material. The section should not replace the core
COI benchmark. It strengthens the paper by showing that the same philosophy
extends to short ribosomal markers: build a candidate set first, then emit only
the deepest defensible rank.

## Section Thesis

MarkerMirror combines learned 12S->16S cross-marker retrieval with BLASTN and
VSEARCH same-marker 12S evidence to support order/no-call assignment for short
12S fragments under missing-reference conditions. The result is not species
identification. The contribution is an auditable evidence compiler that backs
off when species, genus, or family evidence does not transfer cleanly.

## Proposed Manuscript Structure

### Introduction Paragraph

Purpose:

- motivate why 12S/eDNA fragments need rank-aware inference;
- explain that species may be missing from reference tables or unresolved by
  the marker;
- introduce MarkerMirror as a cross-marker candidate generator plus classical
  evidence compiler.

Use:

- `marker_mirror_claim_boundary_box.md`
- `MARKER_MIRROR_COAUTHOR_ONE_PAGER.md`

Suggested lead sentence:

> We therefore evaluated a 12S-specific evidence compiler, MarkerMirror, that
> combines learned cross-marker retrieval with same-marker BLASTN/VSEARCH
> evidence and returns an order/no-call decision when deeper ranks are not
> supported.

### Methods Subsection

Purpose:

- describe the executable wrapper;
- separate candidate generation from final assignment;
- define both decision modes and how they should be interpreted.

Include:

- learned 12S->16S candidate generation;
- BLASTN same-marker 12S candidate generation;
- VSEARCH same-marker 12S candidate generation;
- shared feature table;
- `stable_order` default;
- `high_coverage_order` explicit research mode;
- repeated species-split validation for rank transfer.

Source text:

- `{rel(text_dir / "marker_mirror_methods_paragraph.md")}`
- CLI usage: `experiments/paper1_phylo_calibrated_assignment/MARKER_MIRROR_12S_CLI.md`

### Results Subsection 1: Candidate Support

Claim:

> Learned cross-marker evidence alone is useful but incomplete; the union with
> BLASTN/VSEARCH produces strong high-rank candidate support.

Numbers:

- MarkerMirror-only top-50 species/genus/family/order:
  {pct(ctx['mm'].species_pct)} / {pct(ctx['mm'].genus_pct)} /
  {pct(ctx['mm'].family_pct)} / {pct(ctx['mm'].order_pct)}.
- MarkerMirror + BLASTN union top-50 species/genus/family/order:
  {pct(ctx['blast_union'].species_pct)} / {pct(ctx['blast_union'].genus_pct)} /
  {pct(ctx['blast_union'].family_pct)} / {pct(ctx['blast_union'].order_pct)}.
- MarkerMirror + VSEARCH union top-50 species/genus/family/order:
  {pct(ctx['vsearch_union'].species_pct)} / {pct(ctx['vsearch_union'].genus_pct)} /
  {pct(ctx['vsearch_union'].family_pct)} / {pct(ctx['vsearch_union'].order_pct)}.

Figure/table:

- `{rel(figure_dir / "marker_mirror_candidate_support_bars.png")}`
- `{rel(slide_dir / "marker_mirror_candidate_support_slide_table.md")}`

Caveat:

- species support remains low because the held-out query species are absent
  from the current same-marker 12S reference by split design.

### Results Subsection 2: Order/No-Call Decision

Claim:

> Order-level output can be precise when the pipeline is allowed to abstain.

Numbers:

- stable default:
  {int(ctx['stable'].n_assigned):,}/{int(ctx['stable'].n_queries):,} order
  calls, {pct(ctx['stable'].coverage_pct)} coverage,
  {pct(ctx['stable'].diagnostic_precision_pct)} diagnostic precision.
- high-coverage diagnostic:
  {int(ctx['high'].n_assigned):,}/{int(ctx['high'].n_queries):,} order calls,
  {pct(ctx['high'].coverage_pct)} full-table coverage,
  {pct(ctx['high'].nested_mean_heldout_coverage_pct)} mean held-out coverage,
  {pct(ctx['high'].nested_mean_heldout_precision_pct)} mean held-out precision.

Figure/table:

- `{rel(figure_dir / "marker_mirror_order_policy_tradeoff.png")}`
- `{rel(slide_dir / "marker_mirror_order_policy_slide_table.md")}`

Caveat:

- high-coverage mode remains explicit order/no-call research mode, not the
  conservative default and not species identification.

### Results Subsection 3: Why Family/Genus Stay Disabled

Claim:

> The family/genus limitation is evidence-level, not just a thresholding
> failure.

Numbers:

- family best single-label repair:
  {pct(ctx['family'].single_label_best_coverage_pct)} coverage at
  {pct(ctx['family'].single_label_best_precision_pct)} precision, target met in
  {pct(ctx['family'].single_label_target_met_rate_pct)} of repeats.
- genus best single-label repair:
  {pct(ctx['genus'].single_label_best_coverage_pct)} coverage at
  {pct(ctx['genus'].single_label_best_precision_pct)} precision, target met in
  {pct(ctx['genus'].single_label_target_met_rate_pct)} of repeats.
- family set-valued fallback:
  {pct(ctx['family'].best_set_full_query_truth_coverage_pct)} covered with mean
  set size {float(ctx['family'].best_set_mean_size):.1f}.
- genus set-valued fallback:
  {pct(ctx['genus'].best_set_full_query_truth_coverage_pct)} covered with mean
  set size {float(ctx['genus'].best_set_mean_size):.1f}.

Figure/table:

- `{rel(figure_dir / "marker_mirror_rank_boundary.png")}`
- `{rel(slide_dir / "marker_mirror_rank_boundary_slide_table.md")}`

Conclusion:

- keep species, genus, and family disabled for the current 12S wrapper.

### Results Subsection 4: Runtime And Executability

Claim:

> The full wrapper is executable and the decision policy is not the bottleneck.

Numbers:

- full wrapper: {ctx['total_seconds']:.1f} seconds for
  {int(ctx['stable'].n_queries):,} queries on the Vast RTX host;
- MarkerMirror candidate generation: 15.0 seconds;
- BLASTN: 254.6 seconds;
- VSEARCH: 48.8 seconds;
- stable policy: 1.6 seconds.

Figure/table:

- `{rel(figure_dir / "marker_mirror_runtime_breakdown.png")}`
- `{rel(slide_dir / "marker_mirror_runtime_slide_table.md")}`

### Discussion Paragraph

Points to make:

- The strongest result is not that deep learning beats BLAST.
- MarkerMirror supplies cross-marker candidate evidence, while BLASTN/VSEARCH
  provide strong same-marker checks.
- The pipeline is valuable because it refuses unsupported species/family/genus
  calls and can still make precise order-level assignments.
- Family/genus improvement requires new evidence, not another threshold-only
  repair.

Use:

- `{rel(text_dir / "marker_mirror_claim_boundary_box.md")}`
- `{rel(text_dir / "marker_mirror_results_paragraph.md")}`

## Figure Placement

1. Main figure panel or supplemental figure:
   candidate support by evidence source.
2. Main figure panel:
   stable versus high-coverage order/no-call tradeoff.
3. Supplemental or claim-boundary figure:
   family/genus rank boundary and set-size failure.
4. Supplemental methods/implementation figure:
   runtime breakdown.

## Reviewer-Risk Notes

- Do not state that species identification is solved.
- Do not state that family/genus are production-ready.
- Do not frame MarkerMirror as replacing BLASTN/VSEARCH.
- State clearly that this is controlled 12S marker-reference validation and
  executable wrapper testing, not field-eDNA validation.
- State that high-coverage order mode is explicit research/diagnostic mode,
  while `stable_order` is the conservative default.

## Next Evidence Needed For Family/Genus

Future family/genus attempts should add at least one genuinely new evidence
stream:

- lineage-specific reference coverage;
- alignment-backed marker-resolvability;
- geography/range priors;
- co-occurrence priors;
- external curation/reference completeness signals.

Do not rerun threshold-only or set-only repairs unless one of those evidence
streams changes.
"""


def checklist(ctx: dict[str, Any], text_dir: Path, figure_dir: Path, slide_dir: Path) -> pd.DataFrame:
    rows = [
        {
            "manuscript_section": "Methods",
            "asset": "methods paragraph",
            "source": rel(text_dir / "marker_mirror_methods_paragraph.md"),
            "status": "ready for draft",
            "caveat": "order/no-call research wrapper, not deployed API",
        },
        {
            "manuscript_section": "Results: candidate support",
            "asset": "candidate support figure",
            "source": rel(figure_dir / "marker_mirror_candidate_support_bars.png"),
            "status": "draft figure ready",
            "caveat": "species low by split design",
        },
        {
            "manuscript_section": "Results: order/no-call",
            "asset": "order policy figure",
            "source": rel(figure_dir / "marker_mirror_order_policy_tradeoff.png"),
            "status": "draft figure ready",
            "caveat": "high-coverage mode is diagnostic/research mode",
        },
        {
            "manuscript_section": "Results: rank boundary",
            "asset": "rank boundary figure",
            "source": rel(figure_dir / "marker_mirror_rank_boundary.png"),
            "status": "draft figure ready",
            "caveat": "family/genus/species disabled",
        },
        {
            "manuscript_section": "Results: runtime",
            "asset": "runtime figure",
            "source": rel(figure_dir / "marker_mirror_runtime_breakdown.png"),
            "status": "draft figure ready",
            "caveat": f"runtime measured on Vast for {int(ctx['stable'].n_queries):,} queries",
        },
        {
            "manuscript_section": "Discussion",
            "asset": "claim-boundary box",
            "source": rel(text_dir / "marker_mirror_claim_boundary_box.md"),
            "status": "ready for draft",
            "caveat": "not field-eDNA validation",
        },
        {
            "manuscript_section": "Coauthor review",
            "asset": "slide package outline",
            "source": rel(slide_dir / "marker_mirror_slide_package_outline.md"),
            "status": "ready for review",
            "caveat": "not a polished deck",
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    logger = ProgressLogger(args.log_file or default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).stem)
    logger.log(f"asset_dir={rel(args.asset_dir)}")
    logger.log(f"text_dir={rel(args.text_dir)}")
    logger.log(f"figure_dir={rel(args.figure_dir)}")
    logger.log(f"slide_dir={rel(args.slide_dir)}")
    logger.log(f"output_dir={rel(args.output_dir)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ctx = context(args.asset_dir)
    outline_path = args.output_dir / "marker_mirror_manuscript_section_outline.md"
    checklist_path = args.output_dir / "marker_mirror_manuscript_section_checklist.csv"
    manifest_path = args.output_dir / "marker_mirror_manuscript_section_manifest.json"
    outline_path.write_text(
        build_outline(ctx, args.text_dir, args.figure_dir, args.slide_dir),
        encoding="utf-8",
    )
    logger.log(f"wrote {rel(outline_path)}")
    check = checklist(ctx, args.text_dir, args.figure_dir, args.slide_dir)
    check.to_csv(checklist_path, index=False)
    logger.log(f"wrote {rel(checklist_path)} rows={len(check)}")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": rel(Path(__file__)),
        "inputs": {
            "asset_dir": rel(args.asset_dir),
            "text_dir": rel(args.text_dir),
            "figure_dir": rel(args.figure_dir),
            "slide_dir": rel(args.slide_dir),
        },
        "outputs": {
            "outline": rel(outline_path),
            "checklist": rel(checklist_path),
        },
        "claim_boundary": "Manuscript section outline only. It reorganizes existing MarkerMirror metrics and text assets.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.log(f"wrote {rel(manifest_path)}")
    logger.done(Path(__file__).stem)


if __name__ == "__main__":
    main()
