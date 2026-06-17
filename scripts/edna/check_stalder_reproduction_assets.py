#!/usr/bin/env python3
"""Check whether official TAXDNA assets are ready for exact reproduction."""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_FILES = [
    {
        "name": "official_tree",
        "path": "data/fish_tree_of_life/actinopt_single.tree",
        "min_bytes": 100_000,
    },
    {
        "name": "species_info",
        "path": "data/species_info/species_info.json",
        "min_bytes": 100_000,
    },
    {
        "name": "validation_species",
        "path": "data/val_species/val_species.json",
        "min_bytes": 1_000,
    },
    {
        "name": "reference_sequences",
        "path": "data/sequence_data/species_sequences.json",
        "min_bytes": 100_000,
    },
]

REQUIRED_GLOBS = [
    {
        "name": "tree_embedder_checkpoints",
        "glob": "trained_models/tree_embedder/**/checkpoints/*.ckpt",
        "min_matches": 1,
        "min_bytes": 1_000_000,
    },
    {
        "name": "dna_model_checkpoints",
        "glob": "trained_models/dna_models/**/checkpoints/*.ckpt",
        "min_matches": 1,
        "min_bytes": 1_000_000,
    },
    {
        "name": "cooccurrence_model_checkpoints",
        "glob": "trained_models/co_occ_models/**/checkpoints/*.ckpt",
        "min_matches": 1,
        "min_bytes": 1_000_000,
    },
]


def is_lfs_pointer(path: Path) -> bool:
    if not path.exists() or path.stat().st_size > 1024:
        return False
    try:
        first = path.read_text(errors="replace").splitlines()[0].strip()
    except IndexError:
        return False
    return first == "version https://git-lfs.github.com/spec/v1"


def file_record(repo: Path, rel: str, min_bytes: int = 1) -> dict[str, object]:
    path = repo / rel
    exists = path.exists()
    bytes_on_disk = path.stat().st_size if exists else 0
    pointer = is_lfs_pointer(path) if exists else False
    ok = exists and not pointer and bytes_on_disk >= min_bytes
    return {
        "path": rel,
        "exists": exists,
        "bytes_on_disk": bytes_on_disk,
        "is_lfs_pointer": pointer,
        "min_bytes": min_bytes,
        "ok": ok,
    }


def glob_record(repo: Path, pattern: str, min_matches: int, min_bytes: int) -> dict[str, object]:
    matches = sorted(repo.glob(pattern))
    records = [
        file_record(repo, path.relative_to(repo).as_posix(), min_bytes=min_bytes)
        for path in matches
    ]
    usable = [record for record in records if record["ok"]]
    return {
        "glob": pattern,
        "matches": len(matches),
        "usable_matches": len(usable),
        "min_matches": min_matches,
        "min_bytes": min_bytes,
        "ok": len(usable) >= min_matches,
        "files": records,
    }


def parse_config_paths(repo: Path) -> list[dict[str, object]]:
    """Find TAXDNA config references to data/trained assets and check them.

    Official config paths are written relative to src/taxDNA, not relative to
    the config file location.
    """
    base = repo / "src" / "taxDNA"
    rows: list[dict[str, object]] = []
    token_re = re.compile(r"(\.\./\.\./[A-Za-z0-9_./=-]+)")
    for cfg in sorted(repo.glob("trained_models/**/*.cfg")):
        text = cfg.read_text(errors="replace")
        for token in token_re.findall(text):
            cleaned = token.rstrip(",]")
            resolved = (base / cleaned).resolve()
            try:
                rel = resolved.relative_to(repo.resolve()).as_posix()
            except ValueError:
                rel = str(resolved)
            if not (cleaned.startswith("../../data/") or cleaned.startswith("../../trained_models/")):
                continue
            rows.append(
                {
                    "config": cfg.relative_to(repo).as_posix(),
                    "token": cleaned,
                    **file_record(repo, rel, min_bytes=1),
                }
            )
    unique: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        unique[(str(row["config"]), str(row["path"]))] = row
    return list(unique.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("data/edna/raw/stalder_taxdna"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/edna/stalder_reproduction/asset_readiness.json"),
    )
    args = parser.parse_args()

    if not args.repo.exists():
        raise SystemExit(f"TAXDNA repo does not exist: {args.repo}")

    required_files = {
        item["name"]: file_record(args.repo, str(item["path"]), int(item["min_bytes"]))
        for item in REQUIRED_FILES
    }
    required_globs = {
        item["name"]: glob_record(
            args.repo,
            str(item["glob"]),
            int(item["min_matches"]),
            int(item["min_bytes"]),
        )
        for item in REQUIRED_GLOBS
    }
    config_paths = parse_config_paths(args.repo)
    missing_config_paths = [
        row for row in config_paths if not row["exists"] or row["is_lfs_pointer"]
    ]

    ready = (
        all(record["ok"] for record in required_files.values())
        and all(record["ok"] for record in required_globs.values())
        and not missing_config_paths
    )
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "repo": str(args.repo),
        "ready": ready,
        "required_files": required_files,
        "required_globs": required_globs,
        "config_referenced_paths": config_paths,
        "missing_or_pointer_config_paths": missing_config_paths,
        "next_action": (
            "Official TAXDNA assets look ready for baseline reproduction."
            if ready
            else "Fetch official Git LFS/Renku assets before claiming exact reproduction."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"Wrote {args.output}")
    print(f"ready={ready}")
    if not ready:
        print("Missing/pointer essentials:")
        for name, record in required_files.items():
            if not record["ok"]:
                print(f"  {name}: {record}")
        for name, record in required_globs.items():
            if not record["ok"]:
                print(
                    f"  {name}: usable={record['usable_matches']} "
                    f"matches={record['matches']} glob={record['glob']}"
                )
        if missing_config_paths:
            print("Config-referenced paths missing or unresolved:")
            for row in missing_config_paths[:25]:
                print(f"  {row['config']} -> {row['path']}")
            if len(missing_config_paths) > 25:
                print(f"  ... {len(missing_config_paths) - 25} more")


if __name__ == "__main__":
    main()
