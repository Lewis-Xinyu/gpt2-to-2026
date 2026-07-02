"""
Plot train/eval loss from a training log.csv.

Run:
  python scripts/plot_log.py --log out/baseline_5070/log.csv --out out/baseline_5070/loss_curve.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_log(path: Path) -> dict[str, tuple[list[int], list[float]]]:
    series = {
        "train": ([], []),
        "eval_train": ([], []),
        "eval_val": ([], []),
    }
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            split = row["split"]
            if split not in series or not row["loss"]:
                continue
            xs, ys = series[split]
            xs.append(int(row["iter"]))
            ys.append(float(row["loss"]))
    return series


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    log_path = Path(args.log)
    out_path = Path(args.out)
    series = read_log(log_path)

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit("matplotlib is required for plotting; run `pip install -r requirements.txt`.") from exc

    plt.figure(figsize=(8, 5), dpi=160)
    for label, (xs, ys) in series.items():
        if xs:
            plt.plot(xs, ys, label=label)
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.title(log_path.parent.name)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
