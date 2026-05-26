#!/usr/bin/env python3
"""Inventory the local TAXDNA/Stalder clone and its Git LFS placeholders."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_value(repo: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    value = result.stdout.strip()
    return value or None


def lfs_pointer(path: Path) -> dict[str, object] | None:
    try:
        text = path.read_text(errors="replace")
    except UnicodeDecodeError:
        return None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines or lines[0] != "version https://git-lfs.github.com/spec/v1":
        return None
    pointer: dict[str, object] = {"version": lines[0]}
    for line in lines[1:]:
        if line.startswith("oid sha256:"):
            pointer["oid_sha256"] = line.removeprefix("oid sha256:")
        elif line.startswith("size "):
            try:
                pointer["lfs_size_bytes"] = int(line.removeprefix("size "))
            except ValueError:
                pointer["lfs_size_bytes"] = line.removeprefix("size ")
    return pointer


def classify_path(rel: str) -> str:
    if rel.startswith("data/fish_tree_of_life/"):
        return "fish_tree"
    if rel.startswith("data/species_info/"):
        return "species_info"
    if rel.startswith("data/val_species/"):
        return "validation_species"
    if rel.startswith("trained_models/tree_embedder/"):
        return "tree_embedder_checkpoint"
    if rel.startswith("trained_models/dna_models/"):
        return "dna_model"
    if rel.startswith("trained_models/co_occ_models/"):
        return "co_occurrence_model"
    if rel.startswith("src/taxDNA/"):
        return "source_code"
    if rel.endswith(".cfg"):
        return "config"
    return "other"


def build_inventory(repo: Path) -> dict[str, object]:
    files: list[dict[str, object]] = []
    skipped_dirs = {".git", "__pycache__"}
    for path in sorted(p for p in repo.rglob("*") if p.is_file()):
        if any(part in skipped_dirs for part in path.relative_to(repo).parts):
            continue
        rel = path.relative_to(repo).as_posix()
        pointer = lfs_pointer(path)
        record: dict[str, object] = {
            "path": rel,
            "category": classify_path(rel),
            "bytes_on_disk": path.stat().st_size,
            "sha256_on_disk": sha256_file(path),
            "is_lfs_pointer": pointer is not None,
        }
        if pointer:
            record.update(pointer)
        files.append(record)

    by_category: dict[str, dict[str, int]] = {}
    for record in files:
        category = str(record["category"])
        summary = by_category.setdefault(
            category,
            {"files": 0, "lfs_pointers": 0, "bytes_on_disk": 0, "lfs_target_bytes": 0},
        )
        summary["files"] += 1
        summary["bytes_on_disk"] += int(record["bytes_on_disk"])
        if record["is_lfs_pointer"]:
            summary["lfs_pointers"] += 1
            target_bytes = record.get("lfs_size_bytes")
            if isinstance(target_bytes, int):
                summary["lfs_target_bytes"] += target_bytes

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "repo_path": str(repo),
        "git": {
            "head": git_value(repo, "rev-parse", "HEAD"),
            "branch": git_value(repo, "rev-parse", "--abbrev-ref", "HEAD"),
            "remote_origin_url": git_value(repo, "remote", "get-url", "origin"),
        },
        "summary": {
            "total_files": len(files),
            "lfs_pointer_files": sum(1 for record in files if record["is_lfs_pointer"]),
            "bytes_on_disk": sum(int(record["bytes_on_disk"]) for record in files),
            "lfs_target_bytes": sum(
                int(record.get("lfs_size_bytes", 0))
                for record in files
                if isinstance(record.get("lfs_size_bytes"), int)
            ),
            "by_category": by_category,
        },
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path("data/edna/raw/stalder_taxdna"),
        help="Local TAXDNA/Stalder repository clone.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/edna/raw/stalder_taxdna_manifest.json"),
        help="Output JSON manifest.",
    )
    args = parser.parse_args()

    if not args.repo.exists():
        raise SystemExit(f"Repo does not exist: {args.repo}")

    inventory = build_inventory(args.repo)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")

    summary = inventory["summary"]
    print(
        "Wrote {output} with {files} files, {pointers} LFS pointers, "
        "{target_mb:.1f} MB target payload.".format(
            output=args.output,
            files=summary["total_files"],
            pointers=summary["lfs_pointer_files"],
            target_mb=summary["lfs_target_bytes"] / 1_000_000,
        )
    )


if __name__ == "__main__":
    main()
