# run_pipeline.py
"""
End-to-end thesis experimental pipeline.

Runs the full workflow in-process so the trained-model prediction cache flows
straight into the figure generator:

  1. (optional) fetch data        --fetch
  2. train + evaluate TCN & LSTM, rule-based benchmark, buy-and-hold
                                   -> results/model_comparison_metrics.csv
  3. sensitivity analysis          -> results/sensitivity_thresholds.csv
                                      results/sensitivity_lookback.csv
  4. publication figures (300 DPI) -> results/figures/*.png
  5. thesis summary markdown       -> results/thesis_numbers_summary.md

Usage:
  python run_pipeline.py                      # full run, all 10 stocks
  python run_pipeline.py --quick              # 2 stocks, few epochs (smoke test)
  python run_pipeline.py --epochs 40          # cap epochs
  python run_pipeline.py --no-sensitivity     # skip the expensive sweeps
  python run_pipeline.py --fetch              # refresh data first
"""
from __future__ import annotations

import argparse
import subprocess
import sys

from experiment.core import ALL_STOCKS, ensure_results_dirs
from experiment.evaluate_all import evaluate_universe
from experiment.summary import build_summary
from models.sensitivity_analysis import lookback_sensitivity, threshold_sensitivity
import generate_thesis_plots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="2 stocks, 5 epochs")
    ap.add_argument("--epochs", type=int, default=None, help="cap training epochs")
    ap.add_argument("--no-sensitivity", action="store_true")
    ap.add_argument("--fetch", action="store_true", help="refresh raw data first")
    args = ap.parse_args()

    symbols = ALL_STOCKS
    epochs = args.epochs
    if args.quick:
        symbols = ["AAPL", "CAT"]
        epochs = epochs or 5

    ensure_results_dirs()

    if args.fetch:
        print("\n[1/5] Fetching data ...")
        subprocess.run([sys.executable, "-m", "data.fetchdata"], check=True)
    else:
        print("\n[1/5] Using existing data/raw/combined_data.csv")

    print("\n[2/5] Training & evaluating models + benchmarks ...")
    res = evaluate_universe(symbols=symbols, epochs=epochs, verbose=True)
    cache = res["cache"]

    if not args.no_sensitivity:
        print("\n[3/5] Sensitivity analysis ...")
        threshold_sensitivity(symbols, epochs=epochs)
        lookback_sensitivity(symbols, epochs=epochs)
    else:
        print("\n[3/5] Sensitivity analysis skipped (--no-sensitivity)")

    print("\n[4/5] Generating figures ...")
    generate_thesis_plots.generate_all(cache=cache)

    print("\n[5/5] Building thesis summary markdown ...")
    out = build_summary()
    print(f"\nPipeline finished. Thesis summary: {out}")
    print("   - Metrics:     results/model_comparison_metrics.csv")
    print("   - Sensitivity: results/sensitivity_thresholds.csv, results/sensitivity_lookback.csv")
    print("   - Figures:     results/figures/*.png")
    print("   - Summary:     results/thesis_numbers_summary.md")


if __name__ == "__main__":
    main()
