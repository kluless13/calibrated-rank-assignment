#!/usr/bin/env python3
"""Export Global_eDNA SWARM MOTU tables as TAXDNA-compatible representative FASTA."""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


DNA_RE = re.compile(r"^[ACGTURYKMSWBDHVNacgturykmswbdhvn]+$")


def fasta_wrap(seq: str, width: int = 80) -> str:
    return "\n".join(seq[i : i + width] for i in range(0, len(seq), width))


def clean_id(value: object, fallback: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text or text.lower() == "nan":
        text = fallback
    return re.sub(r"[^A-Za-z0-9_.:-]+", "_", text)


def parse_int(value: object, default: int = 1) -> int:
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        return default
    return max(1, parsed)


def export_table(path: Path, output_dir: Path, min_length: int) -> dict[str, object]:
    df = pd.read_csv(path, sep="\t")
    if "sequence" not in df.columns:
        raise SystemExit(f"{path} is missing a sequence column")
    abundance_column = "total" if "total" in df.columns else "count" if "count" in df.columns else None
    otu_column = "OTU" if "OTU" in df.columns else "id" if "id" in df.columns else None

    rows = []
    for idx, row in df.iterrows():
        seq = str(row.get("sequence", "")).strip().upper()
        if len(seq) < min_length or not DNA_RE.fullmatch(seq):
            continue
        abundance = parse_int(row.get(abundance_column), 1) if abundance_column else 1
        otu_id = clean_id(row.get(otu_column), f"row{idx + 1}") if otu_column else f"row{idx + 1}"
        rows.append((otu_id, abundance, seq))

    out_path = output_dir / f"{path.parent.name}_{path.stem}_repr.fasta"
    with out_path.open("w") as handle:
        for otu_id, abundance, seq in rows:
            # TAXDNA's predict.py reads abundance from the final underscore-delimited token.
            handle.write(f">{otu_id}_{abundance}\n{fasta_wrap(seq)}\n")

    return {
        "source_table": str(path),
        "representative_fasta": str(out_path),
        "rows_in_table": int(len(df)),
        "representatives_written": int(len(rows)),
        "min_length": min_length,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-edna-dir", type=Path, default=Path("/Users/kluless/Downloads/Global_eDNA"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/edna/raw/real_edna/global_tropical_swarm_representatives"),
    )
    parser.add_argument("--min-length", type=int, default=20)
    args = parser.parse_args()

    swarm_dir = args.global_edna_dir / "data" / "swarm"
    if not swarm_dir.exists():
        raise SystemExit(f"Missing Global_eDNA swarm directory: {swarm_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    table_paths = sorted(swarm_dir.glob("*/*_teleo_table_motu.csv"))
    outputs = [export_table(path, args.output_dir, args.min_length) for path in table_paths]
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "global_edna_dir": str(args.global_edna_dir),
        "swarm_dir": str(swarm_dir),
        "output_dir": str(args.output_dir),
        "source_table_count": len(table_paths),
        "total_representatives_written": int(sum(row["representatives_written"] for row in outputs)),
        "outputs": outputs,
        "notes": [
            "Global_eDNA local files contain published SWARM MOTU tables, not raw FASTQ/FASTA reads.",
            "These FASTA files are representative sequences exported from the SWARM outputs for TAXDNA-style downstream prediction.",
            "This is not a de novo rerun of SWARM from raw reads.",
        ],
    }
    manifest_path = args.output_dir / "global_edna_swarm_representatives_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: manifest[k] for k in ["source_table_count", "total_representatives_written"]}, indent=2))
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
