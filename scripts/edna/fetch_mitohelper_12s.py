"""
Download versioned Mitohelper 12S fish references for the eDNA track.

Primary source:
  Nov2025 reference datasets for Mitohelper
  https://zenodo.org/records/17602902

The TSV is the primary training source because it contains accession,
taxonomy, and sequence in one table. The non-redundant FASTA is useful for
VSEARCH/BLAST-style baselines.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


FILES = {
    "mitofish.12S.Nov2025.tsv": {
        "url": "https://zenodo.org/records/17602902/files/mitofish.12S.Nov2025.tsv?download=1",
        "md5": "585b5a4082f1c3fdf9d742f2d640d0b2",
        "role": "primary taxonomy+sequence table",
    },
    "mitofish.12S.Nov2025_NR.fasta": {
        "url": "https://zenodo.org/records/17602902/files/mitofish.12S.Nov2025_NR.fasta?download=1",
        "md5": "0c4c779ee2b957d21cb51e3dfccd98c6",
        "role": "non-redundant FASTA for alignment baselines",
    },
}


def hash_file(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, force: bool) -> None:
    if dest.exists() and not force:
        print(f"exists: {dest}")
        return

    tmp = dest.with_suffix(dest.suffix + ".tmp")
    print(f"download: {url}")
    with urlopen(url) as response, open(tmp, "wb") as out:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    tmp.replace(dest)
    print(f"wrote: {dest}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/edna/raw")
    parser.add_argument("--skip-fasta", action="store_true",
                        help="Download only the TSV needed for split building")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "source": "Mitohelper Nov2025 12S reference datasets",
        "zenodo_record": "https://zenodo.org/records/17602902",
        "doi": "10.5281/zenodo.17602902",
        "files": {},
    }

    for filename, meta in FILES.items():
        if args.skip_fasta and filename.endswith(".fasta"):
            continue
        path = out_dir / filename
        download(meta["url"], path, force=args.force)
        md5 = hash_file(path, "md5")
        sha256 = hash_file(path, "sha256")
        if md5 != meta["md5"]:
            raise RuntimeError(f"MD5 mismatch for {filename}: expected {meta['md5']}, got {md5}")
        manifest["files"][filename] = {
            **meta,
            "path": str(path),
            "bytes": path.stat().st_size,
            "md5": md5,
            "sha256": sha256,
        }

    manifest_path = out_dir / "reference_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
