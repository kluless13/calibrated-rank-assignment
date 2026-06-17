from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ZeroShotCandidateEvaluationTests(unittest.TestCase):
    def test_hidden_species_still_scores_query_rank_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            input_dir = work / "input"
            output_dir = work / "out"
            input_dir.mkdir()

            with (input_dir / "zero_shot_queries.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "processid",
                        "tree_label",
                        "species_name",
                        "genus_name",
                        "family_name",
                        "order_name",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "processid": "q1",
                        "tree_label": "Hidden_alpha",
                        "species_name": "Hidden alpha",
                        "genus_name": "Hidden",
                        "family_name": "Hiddenidae",
                        "order_name": "Hiddeniformes",
                    }
                )

            with (input_dir / "candidate_species.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["tree_label", "species_name", "genus_name", "family_name", "order_name"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "tree_label": "Hidden_beta",
                        "species_name": "Hidden beta",
                        "genus_name": "Hidden",
                        "family_name": "Hiddenidae",
                        "order_name": "Hiddeniformes",
                    }
                )

            predictions = work / "predictions.csv"
            with predictions.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["processid", "top_tree_labels"])
                writer.writeheader()
                writer.writerow({"processid": "q1", "top_tree_labels": json.dumps(["Hidden_beta"])})

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "edna" / "eval_zero_shot_candidate_predictions.py"),
                    "--input-dir",
                    str(input_dir),
                    "--predictions",
                    str(predictions),
                    "--output-dir",
                    str(output_dir),
                ],
                check=True,
                cwd=ROOT,
            )

            metrics = json.loads((output_dir / "zero_shot_candidate_metrics.json").read_text())["metrics"]
            self.assertEqual(metrics["species"]["top1"], 0.0)
            self.assertEqual(metrics["genus"]["top1"], 1.0)
            self.assertEqual(metrics["family"]["top1"], 1.0)
            self.assertEqual(metrics["order"]["top1"], 1.0)


if __name__ == "__main__":
    unittest.main()
