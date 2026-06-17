#!/usr/bin/env python3
"""Summarize current Global_eDNA method benchmarks, including reference-status strata."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


RANKS = ["species", "genus", "family", "order"]
TOP_K = [1, 5, 10]


DEFAULT_METHODS = [
    {
        "method": "mamba_sequence_only",
        "predictions": "results/remote_runs/2026-05-26/rtx_pro/global_edna_multisource_teleo_hier_strong_seed1207_predictions/zero_shot_candidate_predictions.csv",
        "metrics": "results/edna/global_tropical_validation/multisource_teleo_hier_strong_seed1207/global_edna_validation_metrics.json",
    },
    {
        "method": "mamba_same_sample_cooccurrence_w005",
        "predictions": "results/edna/global_tropical_validation/multisource_teleo_hier_strong_seed1207_cooccurrence_w005/sample_cooccurrence_reranked_predictions.csv",
        "metrics": "results/edna/global_tropical_validation/multisource_teleo_hier_strong_seed1207_cooccurrence_w005/global_edna_validation/global_edna_validation_metrics.json",
    },
    {
        "method": "mamba_rls_geographic_site20_w005",
        "predictions": "results/edna/global_tropical_validation/multisource_teleo_hier_strong_seed1207_rls_geo_site20_w005/geographic_prior_reranked_predictions.csv",
        "metrics": "results/edna/global_tropical_validation/multisource_teleo_hier_strong_seed1207_rls_geo_site20_w005/global_edna_validation/global_edna_validation_metrics.json",
    },
    {
        "method": "mamba_obis_occurrence_site20_w005",
        "predictions": "results/edna/global_tropical_validation/multisource_teleo_hier_strong_seed1207_obis_site20_w005/occurrence_prior_reranked_predictions.csv",
        "metrics": "results/edna/global_tropical_validation/multisource_teleo_hier_strong_seed1207_obis_site20_w005/global_edna_validation/global_edna_validation_metrics.json",
    },
    {
        "method": "mamba_rls_obis_site20_w005",
        "predictions": "results/edna/global_tropical_validation/multisource_teleo_hier_strong_seed1207_rls_obis_site20_w005/combined_prior_reranked_predictions.csv",
        "metrics": "results/edna/global_tropical_validation/multisource_teleo_hier_strong_seed1207_rls_obis_site20_w005/global_edna_validation/global_edna_validation_metrics.json",
    },
    {
        "method": "ssm_learned_cooccurrence_w025",
        "predictions": "results/edna/taxdna_ssm/global_edna_multisource_teleo_ssm_learned_cooccurrence_w025/learned_cooccurrence_predictions.csv",
        "metrics": "results/edna/taxdna_ssm/global_edna_multisource_teleo_ssm_learned_cooccurrence_w025/global_edna_validation/global_edna_validation_metrics.json",
    },
    {
        "method": "ssm_learned_cooccurrence_w050",
        "predictions": "results/edna/taxdna_ssm/global_edna_multisource_teleo_ssm_learned_cooccurrence_w050/learned_cooccurrence_predictions.csv",
        "metrics": "results/edna/taxdna_ssm/global_edna_multisource_teleo_ssm_learned_cooccurrence_w050/global_edna_validation/global_edna_validation_metrics.json",
    },
    {
        "method": "ssm_learned_cooccurrence_w100",
        "predictions": "results/edna/taxdna_ssm/global_edna_multisource_teleo_ssm_learned_cooccurrence_w100/learned_cooccurrence_predictions.csv",
        "metrics": "results/edna/taxdna_ssm/global_edna_multisource_teleo_ssm_learned_cooccurrence_w100/global_edna_validation/global_edna_validation_metrics.json",
    },
    {
        "method": "ssm_learned_cooccurrence_w200",
        "predictions": "results/edna/taxdna_ssm/global_edna_multisource_teleo_ssm_learned_cooccurrence_w200/learned_cooccurrence_predictions.csv",
        "metrics": "results/edna/taxdna_ssm/global_edna_multisource_teleo_ssm_learned_cooccurrence_w200/global_edna_validation/global_edna_validation_metrics.json",
    },
    {
        "method": "cnn_learned_cooccurrence_w025",
        "predictions": "results/edna/taxdna_ssm/global_edna_multisource_teleo_cnn_learned_cooccurrence_w025/learned_cooccurrence_predictions.csv",
        "metrics": "results/edna/taxdna_ssm/global_edna_multisource_teleo_cnn_learned_cooccurrence_w025/global_edna_validation/global_edna_validation_metrics.json",
    },
    {
        "method": "cnn_learned_cooccurrence_w050",
        "predictions": "results/edna/taxdna_ssm/global_edna_multisource_teleo_cnn_learned_cooccurrence_w050/learned_cooccurrence_predictions.csv",
        "metrics": "results/edna/taxdna_ssm/global_edna_multisource_teleo_cnn_learned_cooccurrence_w050/global_edna_validation/global_edna_validation_metrics.json",
    },
    {
        "method": "cnn_learned_cooccurrence_w100",
        "predictions": "results/edna/taxdna_ssm/global_edna_multisource_teleo_cnn_learned_cooccurrence_w100/learned_cooccurrence_predictions.csv",
        "metrics": "results/edna/taxdna_ssm/global_edna_multisource_teleo_cnn_learned_cooccurrence_w100/global_edna_validation/global_edna_validation_metrics.json",
    },
    {
        "method": "cnn_learned_cooccurrence_w200",
        "predictions": "results/edna/taxdna_ssm/global_edna_multisource_teleo_cnn_learned_cooccurrence_w200/learned_cooccurrence_predictions.csv",
        "metrics": "results/edna/taxdna_ssm/global_edna_multisource_teleo_cnn_learned_cooccurrence_w200/global_edna_validation/global_edna_validation_metrics.json",
    },
    {
        "method": "rls_prior_only_site20",
        "predictions": "results/edna/global_tropical_validation/rls_prior_only_site20/rls_prior_only_predictions.csv",
        "metrics": "results/edna/global_tropical_validation/rls_prior_only_site20/global_edna_validation/global_edna_validation_metrics.json",
    },
    {
        "method": "obis_occurrence_prior_only_site20",
        "predictions": "results/edna/global_tropical_validation/obis_prior_only_site20/occurrence_prior_only_predictions.csv",
        "metrics": "results/edna/global_tropical_validation/obis_prior_only_site20/global_edna_validation/global_edna_validation_metrics.json",
    },
    {
        "method": "blast_train_reference",
        "predictions": "results/edna/global_tropical_validation/blast_train_reference/global_edna_blast_zero_shot_predictions.csv",
        "metrics": "results/edna/global_tropical_validation/blast_train_reference/global_edna_validation/global_edna_validation_metrics.json",
    },
]


def nonempty(value: object) -> bool:
    return not pd.isna(value) and str(value).strip() not in {"", "nan", "None"}


def parse_labels(value: object) -> list[str]:
    if not nonempty(value):
        return []
    text = str(value).strip()
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item).strip().replace(" ", "_") for item in parsed if nonempty(item)]
        except json.JSONDecodeError:
            pass
    for sep in ["|", ",", ";"]:
        if sep in text:
            return [part.strip().replace(" ", "_") for part in text.split(sep) if part.strip()]
    return [text.replace(" ", "_")]


def rank_value(label: str | None, rank: str, taxonomy: dict[str, dict[str, object]]) -> str:
    if not label:
        return ""
    if rank == "species":
        return label
    if rank == "genus":
        value = taxonomy.get(label, {}).get("genus_name")
        return str(value) if nonempty(value) else label.split("_", 1)[0]
    value = taxonomy.get(label, {}).get(f"{rank}_name")
    return str(value) if nonempty(value) else ""


def topk_correct(true_label: str, labels: list[str], rank: str, k: int, taxonomy: dict[str, dict[str, object]]) -> bool:
    true_value = rank_value(true_label, rank, taxonomy)
    if not true_value:
        return False
    pred_values = {rank_value(label, rank, taxonomy) for label in labels[:k]}
    return true_value in pred_values


def load_method_predictions(path: Path) -> pd.DataFrame:
    pred = pd.read_csv(path)
    if "query_processid" in pred.columns:
        pred["join_processid"] = pred["query_processid"].astype(str)
    else:
        pred["join_processid"] = pred["processid"].astype(str)
    return pred


def stratified_rows(
    method: str,
    predictions_path: Path,
    sample_map: pd.DataFrame,
    candidates: pd.DataFrame,
    taxonomy: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    pred = load_method_predictions(predictions_path)
    if "sample_id" in pred.columns:
        merged = pred.copy()
        if "true_tree_label" not in merged.columns:
            merged = merged.merge(
                sample_map[["sample_id", "query_processid", "true_tree_label"]],
                on=["sample_id", "query_processid"],
                how="left",
            )
    else:
        merged = sample_map.merge(
            pred,
            left_on="query_processid",
            right_on="join_processid",
            how="left",
            suffixes=("", "_pred"),
        )
    ref_flags = candidates.set_index("tree_label")["has_reference_sequence"].to_dict()
    merged["true_tree_label_clean"] = merged["true_tree_label"].astype(str).str.replace(" ", "_", regex=False)
    merged["reference_status"] = merged["true_tree_label_clean"].map(
        lambda label: "reference_species" if bool(ref_flags.get(label, False)) else "zero_shot_species"
    )

    rows = []
    for status, sub in [("all", merged)] + list(merged.groupby("reference_status")):
        out: dict[str, object] = {"method": method, "reference_status": status, "rows": int(len(sub))}
        for rank in RANKS:
            for k in TOP_K:
                valid = 0
                correct = 0
                for _, row in sub.iterrows():
                    labels = parse_labels(row.get("top_tree_labels"))
                    true_label = str(row.get("true_tree_label_clean", ""))
                    if not labels or not true_label:
                        continue
                    valid += 1
                    correct += int(topk_correct(true_label, labels, rank, k, taxonomy))
                out[f"{rank}_top{k}"] = correct / valid if valid else None
                out[f"{rank}_n_top{k}"] = valid
        rows.append(out)
    return rows


def overall_rows(methods: list[dict[str, str]]) -> list[dict[str, object]]:
    rows = []
    for method in methods:
        metrics_path = Path(method["metrics"])
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text())
        row = {"method": method["method"], "metrics_json": str(metrics_path)}
        for rank in RANKS:
            for k in TOP_K:
                row[f"asv_{rank}_top{k}"] = metrics["asv_metrics"][rank][f"top{k}"]
            row[f"sample_{rank}_top1_jaccard"] = metrics["sample_metric_summary"][f"{rank}_top1_jaccard"]["mean"]
            row[f"sample_{rank}_top1_recall"] = metrics["sample_metric_summary"][f"{rank}_top1_recall"]["mean"]
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/edna/real_edna_queries/global_tropical_multisource_teleo"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/edna/global_tropical_validation/summary"))
    parser.add_argument("--methods-json", type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    methods = DEFAULT_METHODS
    if args.methods_json:
        methods = json.loads(args.methods_json.read_text())

    sample_map = pd.read_csv(args.input_dir / "sample_query_map.csv")
    candidates = pd.read_csv(args.input_dir / "candidate_species.csv")
    taxonomy = candidates.set_index("tree_label").to_dict(orient="index")

    overall = pd.DataFrame(overall_rows(methods))
    overall_path = args.output_dir / "global_edna_method_overall_metrics.csv"
    overall.to_csv(overall_path, index=False)

    strat_rows = []
    for method in methods:
        pred_path = Path(method["predictions"])
        if pred_path.exists():
            strat_rows.extend(stratified_rows(method["method"], pred_path, sample_map, candidates, taxonomy))
    strat = pd.DataFrame(strat_rows)
    strat_path = args.output_dir / "global_edna_method_stratified_metrics.csv"
    strat.to_csv(strat_path, index=False)

    manifest = {
        "input_dir": str(args.input_dir),
        "methods": methods,
        "overall_metrics_csv": str(overall_path),
        "stratified_metrics_csv": str(strat_path),
    }
    (args.output_dir / "global_edna_method_summary_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"Wrote {overall_path} and {strat_path}")


if __name__ == "__main__":
    main()
