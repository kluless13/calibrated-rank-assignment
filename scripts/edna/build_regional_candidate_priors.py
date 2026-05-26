#!/usr/bin/env python3
"""Build simple regional candidate priors from real eDNA/trawl occurrences."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lam = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def read_occurrences(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        if "tree_label" not in df.columns:
            continue
        frames.append(df)
    if not frames:
        raise RuntimeError("No occurrence CSVs with tree_label column found.")
    df = pd.concat(frames, ignore_index=True)
    df = df[df["tree_label"].notna()].copy()
    df = df[df["is_in_tree"].astype(str).str.lower().isin(["true", "1"])].copy()
    df["decimalLatitude"] = pd.to_numeric(df["decimalLatitude"], errors="coerce")
    df["decimalLongitude"] = pd.to_numeric(df["decimalLongitude"], errors="coerce")
    df = df.dropna(subset=["eventID", "tree_label", "decimalLatitude", "decimalLongitude"]).copy()
    return df


def event_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (source_id, event_id), sub in df.groupby(["source_id", "eventID"], dropna=False):
        labels = sorted(set(sub["tree_label"].astype(str)))
        abundance = Counter()
        for _, row in sub.iterrows():
            quantity = pd.to_numeric(row.get("organismQuantity"), errors="coerce")
            count = float(quantity) if not pd.isna(quantity) and quantity > 0 else 1.0
            abundance[str(row["tree_label"])] += count
        rows.append({
            "source_id": source_id,
            "eventID": event_id,
            "latitude": float(sub["decimalLatitude"].mean()),
            "longitude": float(sub["decimalLongitude"].mean()),
            "observed_tree_labels": "|".join(labels),
            "observed_species_count": len(labels),
            "observed_record_count": int(len(sub)),
            "observed_abundance_json": json.dumps(dict(sorted(abundance.items())), sort_keys=True),
        })
    return pd.DataFrame(rows)


def build_priors(events: pd.DataFrame, radius_km: float, same_source_only: bool) -> pd.DataFrame:
    rows = []
    event_records = events.to_dict(orient="records")
    for event in event_records:
        candidate_counts: Counter[str] = Counter()
        neighbor_events = 0
        nearest_distance = None
        for other in event_records:
            if other["eventID"] == event["eventID"] and other["source_id"] == event["source_id"]:
                continue
            if same_source_only and other["source_id"] != event["source_id"]:
                continue
            dist = haversine_km(
                event["latitude"],
                event["longitude"],
                other["latitude"],
                other["longitude"],
            )
            if nearest_distance is None or dist < nearest_distance:
                nearest_distance = dist
            if dist <= radius_km:
                neighbor_events += 1
                for label in str(other["observed_tree_labels"]).split("|"):
                    if label:
                        candidate_counts[label] += 1
        candidates = [label for label, _ in candidate_counts.most_common()]
        rows.append({
            "source_id": event["source_id"],
            "eventID": event["eventID"],
            "latitude": event["latitude"],
            "longitude": event["longitude"],
            "observed_tree_labels": event["observed_tree_labels"],
            "observed_species_count": event["observed_species_count"],
            "neighbor_events_within_radius": neighbor_events,
            "nearest_neighbor_distance_km": nearest_distance,
            "radius_km": radius_km,
            "same_source_only": same_source_only,
            "regional_candidate_count": len(candidates),
            "regional_candidate_tree_labels": "|".join(candidates),
            "regional_candidate_counts_json": json.dumps(dict(candidate_counts), sort_keys=True),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--occurrence-csvs",
        nargs="+",
        type=Path,
        default=sorted(Path("results/edna/real_edna_overlap").glob("*.records.augmented.csv")),
        help="Augmented real occurrence CSVs from eval_real_edna_occurrence_overlap.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/edna/real_edna_priors"),
    )
    parser.add_argument("--radius-km", type=float, default=250.0)
    parser.add_argument("--cross-source", action="store_true", help="Allow neighbor events from other source tables.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    occurrences = read_occurrences(args.occurrence_csvs)
    events = event_table(occurrences)
    priors = build_priors(events, radius_km=args.radius_km, same_source_only=not args.cross_source)

    events_path = args.output_dir / "real_edna_events.csv"
    priors_path = args.output_dir / "regional_candidate_priors.csv"
    events.to_csv(events_path, index=False)
    priors.to_csv(priors_path, index=False)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "occurrence_csvs": [str(path) for path in args.occurrence_csvs],
        "radius_km": args.radius_km,
        "same_source_only": not args.cross_source,
        "occurrence_records_used": int(len(occurrences)),
        "event_count": int(len(events)),
        "events_csv": str(events_path),
        "priors_csv": str(priors_path),
        "events_with_any_regional_candidate": int((priors["regional_candidate_count"] > 0).sum()),
        "median_regional_candidate_count": float(priors["regional_candidate_count"].median()) if len(priors) else 0.0,
    }
    manifest_path = args.output_dir / "regional_candidate_priors_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {priors_path} for {len(events)} events.")


if __name__ == "__main__":
    main()
