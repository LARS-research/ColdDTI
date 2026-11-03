"""Collect completed runs and report sample mean/std across training seeds."""
import argparse
import json
from pathlib import Path

import pandas as pd


METRICS = ("auc", "aupr", "f1", "precision", "recall", "accuracy")


def summarize(directory):
    directory = Path(directory)
    records = [json.loads(file.read_text()) for file in sorted(directory.glob("*/*/*/seed_*/metrics.json"))]
    if not records:
        return None
    frame = pd.DataFrame(records)
    frame.to_csv(directory / "results.csv", index=False)
    groups = ["dataset", "split", "variant", "boundary_seed"]
    statistics = frame.groupby(groups)[list(METRICS)].agg(["mean", "std"])
    statistics.columns = [f"{metric}_{statistic}" for metric, statistic in statistics.columns]
    # A single run has no estimable standard deviation; leave it empty.
    statistics["runs"] = frame.groupby(groups).size()
    statistics.to_csv(directory / "summary.csv")
    return statistics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "output" / "ablations")
    args = parser.parse_args()
    result = summarize(args.output_dir)
    print(result.to_string() if result is not None else "No completed runs found")
