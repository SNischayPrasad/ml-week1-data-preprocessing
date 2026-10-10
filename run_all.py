"""Reproduce every number and figure in reports/REPORT.md from scratch.

    python run_all.py

Runs the seven steps in order. Each step writes a JSON of its own results to
outputs/, so any figure or claim in the report can be traced to the step that
produced it. Total runtime on a laptop CPU: roughly 4-6 minutes, dominated by
the cross-validated ablations in step 6.
"""
import runpy
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
RAW = ROOT / "data" / "raw"

UCI = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
FILES = ["adult.data", "adult.test", "adult.names"]

STEPS = [
    ("step00_assemble", "Assemble the two UCI files into one typed CSV"),
    ("step01_audit", "Data quality audit"),
    ("step02_clean_split", "Cleaning decisions, then the stratified split"),
    ("step02b_unseen_category_risk", "How often a random split yields an unseen category"),
    ("step03_eda", "Exploratory figures"),
    ("step04_pipeline", "Fit the preprocessing pipeline (train only)"),
    ("step05_feature_selection", "Feature selection, three views"),
    ("step06_experiments", "Ablations"),
    ("step07_summary_figures", "Summary figures"),
    ("verify_report", "Cross-check every number quoted in reports/REPORT.md"),
]


def fetch_raw() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dest = RAW / name
        if dest.exists():
            print(f"  {name:14s} already present ({dest.stat().st_size:,} bytes)")
            continue
        print(f"  {name:14s} downloading...", end=" ", flush=True)
        req = urllib.request.Request(UCI + name, headers={"User-Agent": "Mozilla/5.0"})
        dest.write_bytes(urllib.request.urlopen(req, timeout=60).read())
        print(f"{dest.stat().st_size:,} bytes")


def main() -> int:
    t0 = time.time()
    print("=" * 72)
    print("Week 1 - Python for Machine Learning & Data Preprocessing")
    print("=" * 72)

    print("\n[raw data]")
    fetch_raw()

    sys.path.insert(0, str(SRC))
    for i, (mod, desc) in enumerate(STEPS):
        print(f"\n{'=' * 72}\n[{i}/{len(STEPS) - 1}] {mod} - {desc}\n{'=' * 72}")
        t = time.time()
        result = subprocess.run([sys.executable, str(SRC / f"{mod}.py")], cwd=SRC)
        if result.returncode != 0:
            print(f"\nFAILED at {mod} (exit {result.returncode})")
            return result.returncode
        print(f"\n-- {mod} finished in {time.time() - t:.1f}s")

    print(f"\n{'=' * 72}")
    print(f"All steps completed in {time.time() - t0:.1f}s")
    print("  processed data -> data/processed/")
    print("  metrics        -> outputs/*.json")
    print("  figures        -> reports/figures/")
    print("  write-up       -> reports/REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
