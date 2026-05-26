"""
Export per-query predictions from a trained 6-mer MarineMamba checkpoint.

This is designed for downstream confidence/no-call and hybrid BLAST/neural
experiments. It recreates the label maps from the split CSVs, loads the saved
6-mer checkpoint, then writes direct-head and embedding-kNN predictions for
supervised_test, eval_c_unseen_species, and unseen when present.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Dataset


RANKS = ["species", "genus", "family", "order"]
RANK_COLS = {
    "species": "species_name",
    "genus": "genus_name",
    "family": "family_name",
    "order": "order_name",
}


def load_curriculum_module():
    script_path = Path(__file__).resolve().parent / "11_curriculum_6mer.py"
    spec = importlib.util.spec_from_file_location("curriculum_6mer_module", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SeqDataset(Dataset):
    def __init__(self, seqs: list[str], tokenize_fn, max_tokens: int):
        self.seqs = seqs
        self.tokenize_fn = tokenize_fn
        self.max_tokens = max_tokens

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.tokenize_fn(self.seqs[idx], max_tokens=self.max_tokens), dtype=torch.long)


def clean_text(value: object) -> str:
    text = str(value)
    return "" if text == "nan" else text


def build_label_maps(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    labels = {}
    for rank, col in RANK_COLS.items():
        values = sorted(all_df[col].dropna().astype(str).unique())
        labels[rank] = {
            "values": values,
            "to_idx": {value: i for i, value in enumerate(values)},
            "from_idx": {i: value for i, value in enumerate(values)},
        }
    return labels


def extract_batches(model, seqs: list[str], module, max_tokens: int, batch_size: int, device: str):
    ds = SeqDataset(seqs, module.tokenize_6mer, max_tokens)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
    features = []
    direct = {rank: [] for rank in RANKS}
    confidences = {rank: [] for rank in RANKS}

    model.eval()
    with torch.no_grad():
        for x in dl:
            x = x.to(device)
            logits = model(x)
            features.append(model.get_features(x).cpu().numpy())
            for rank in RANKS:
                probs = torch.softmax(logits[rank], dim=1)
                conf, pred = probs.max(dim=1)
                direct[rank].extend(pred.cpu().numpy().tolist())
                confidences[rank].extend(conf.cpu().numpy().tolist())
    return np.vstack(features), direct, confidences


def majority_taxonomy(df: pd.DataFrame, child_col: str, parent_col: str) -> dict[str, str]:
    counts = df.groupby([child_col, parent_col], dropna=False).size().reset_index(name="n")
    counts = counts.sort_values([child_col, "n", parent_col], ascending=[True, False, True])
    best = counts.drop_duplicates(subset=[child_col], keep="first")
    return {clean_text(row[child_col]): clean_text(row[parent_col]) for _, row in best.iterrows()}


def split_predictions(
    split_name: str,
    query_df: pd.DataFrame,
    query_features: np.ndarray,
    direct_idx: dict[str, list[int]],
    direct_conf: dict[str, list[float]],
    train_df: pd.DataFrame,
    train_features: np.ndarray,
    labels: dict,
    output_path: Path,
) -> dict:
    nn = NearestNeighbors(n_neighbors=min(2, len(train_df)), metric="cosine", n_jobs=-1)
    nn.fit(train_features)
    distances, indices = nn.kneighbors(query_features)

    train_reset = train_df.reset_index(drop=True)
    genus_to_family = majority_taxonomy(train_reset, "genus_name", "family_name")
    genus_to_order = majority_taxonomy(train_reset, "genus_name", "order_name")

    rows = []
    for pos, row in query_df.reset_index(drop=True).iterrows():
        nearest = train_reset.iloc[int(indices[pos][0])]
        nearest_genus = clean_text(nearest["genus_name"])
        nearest_species = clean_text(nearest["species_name"])
        knn_distance = float(distances[pos][0])
        knn_margin = None
        if distances.shape[1] > 1:
            knn_margin = float(distances[pos][1] - distances[pos][0])

        out = {
            "split": split_name,
            "query_index": int(pos),
            "processid": clean_text(row.get("processid", "")),
            "sequence_length": int(len(str(row.get("nucleotides", "")))),
            "knn_distance": knn_distance,
            "knn_margin": knn_margin,
            "knn_ref_index": int(indices[pos][0]),
            "knn_pred_species": nearest_species,
            "knn_pred_genus": nearest_genus,
            "knn_pred_family": genus_to_family.get(nearest_genus, ""),
            "knn_pred_order": genus_to_order.get(nearest_genus, ""),
        }
        for rank, col in RANK_COLS.items():
            out[f"true_{rank}"] = clean_text(row.get(col, ""))
            out[f"direct_pred_{rank}"] = labels[rank]["from_idx"].get(int(direct_idx[rank][pos]), "")
            out[f"direct_conf_{rank}"] = float(direct_conf[rank][pos])
        rows.append(out)

    pred_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output_path, index=False)

    metrics = {}
    for source in ["direct", "knn"]:
        source_metrics = {}
        for rank in RANKS:
            if split_name == "eval_c_unseen_species" and rank == "species":
                source_metrics[rank] = {
                    "accuracy": None,
                    "note": "Not applicable: Eval C query species are absent from training labels.",
                }
                continue
            pred_col = f"{source}_pred_{rank}"
            true_col = f"true_{rank}"
            valid = pred_df[
                pred_df[true_col].astype(bool) & pred_df[pred_col].astype(bool)
            ]
            if len(valid):
                source_metrics[rank] = {
                    "accuracy": float((valid[true_col] == valid[pred_col]).mean()),
                    "n_correct": int((valid[true_col] == valid[pred_col]).sum()),
                    "n_total": int(len(valid)),
                }
        metrics[source] = source_metrics
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-tokens", type=int, default=655)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    module = load_curriculum_module()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_df = pd.read_csv(data_dir / "supervised_train.csv")
    val_df = pd.read_csv(data_dir / "supervised_val.csv")
    test_df = pd.read_csv(data_dir / "supervised_test.csv")
    labels = build_label_maps(train_df, val_df, test_df)

    model = module.MultiHeadMamba6mer(
        d_model=384,
        n_orders=len(labels["order"]["values"]),
        n_families=len(labels["family"]["values"]),
        n_genera=len(labels["genus"]["values"]),
        n_species=len(labels["species"]["values"]),
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.to(device)

    train_features, _, _ = extract_batches(
        model, train_df["nucleotides"].astype(str).tolist(), module,
        args.max_tokens, args.batch_size, device,
    )

    split_files = [
        ("supervised_test", data_dir / "supervised_test.csv"),
        ("eval_c_unseen_species", data_dir / "eval_c_unseen_species.csv"),
        ("unseen", data_dir / "unseen.csv"),
    ]
    results = {
        "experiment": "6mer_checkpoint_predictions",
        "data_dir": str(data_dir),
        "checkpoint": str(args.checkpoint),
        "max_tokens": args.max_tokens,
        "splits": {},
    }
    for split_name, split_path in split_files:
        if not split_path.exists():
            continue
        query_df = pd.read_csv(split_path)
        features, direct_idx, direct_conf = extract_batches(
            model, query_df["nucleotides"].astype(str).tolist(), module,
            args.max_tokens, args.batch_size, device,
        )
        pred_path = output_dir / f"{split_name}_neural_predictions.csv"
        results["splits"][split_name] = split_predictions(
            split_name, query_df, features, direct_idx, direct_conf,
            train_df, train_features, labels, pred_path,
        )
        results["splits"][split_name]["prediction_csv"] = str(pred_path)

    result_path = output_dir / "neural_prediction_metrics.json"
    result_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"saved: {result_path}")


if __name__ == "__main__":
    main()
