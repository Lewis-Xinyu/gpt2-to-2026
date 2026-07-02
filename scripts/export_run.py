"""
Copy lightweight training artifacts from an ignored out/ run into reports/runs/.

Checkpoints stay in out/ and should not be committed. Logs, resolved configs,
JSON summaries, and generated samples are small enough to version and review.

Run:
  python scripts/export_run.py --out_dir out/baseline_5070
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path


def read_best_val(log_path: Path) -> tuple[int | None, float | None]:
    if not log_path.exists():
        return None, None
    best_iter, best_loss = None, None
    with log_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("split") != "eval_val" or not row.get("loss"):
                continue
            loss = float(row["loss"])
            if best_loss is None or loss < best_loss:
                best_iter = int(row["iter"])
                best_loss = loss
    return best_iter, best_loss


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--report_dir", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    run_name = out_dir.name
    report_dir = Path(args.report_dir) if args.report_dir else Path("reports") / "runs" / run_name
    report_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    for name in ("config.yaml", "log.csv", "summary.json", "samples.txt", "samples.md", "loss_curve.png"):
        if copy_if_exists(out_dir / name, report_dir / name):
            copied.append(name)

    best_iter, best_loss = read_best_val(out_dir / "log.csv")
    summary = {}
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        with summary_path.open() as f:
            summary = json.load(f)

    lines = [
        f"# {run_name}",
        "",
        "## Result",
        "",
        f"- best val loss: {best_loss:.4f} at iter {best_iter}" if best_loss is not None else "- best val loss: not available",
        f"- device: {summary.get('device', 'unknown')}",
        f"- dtype: {summary.get('dtype', 'unknown')}",
        "",
        "## Artifacts",
        "",
    ]
    lines.extend(f"- `{name}`" for name in copied)
    lines.extend(
        [
            "",
            "Checkpoint weights are intentionally kept under `out/` and ignored by Git.",
            "",
        ]
    )
    (report_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"exported {run_name} -> {report_dir}")


if __name__ == "__main__":
    main()
