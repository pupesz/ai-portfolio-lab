# run_pipeline.py
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

def run_module(mod_name: str):
    """Run `python -m <module>` with error checking."""
    print(f"\n==============================")
    print(f"▶ Running module: {mod_name}")
    print("==============================")
    subprocess.run([sys.executable, "-m", mod_name], check=True)

def main():
    # 1) Fetch & prepare data
    run_module("data.fetchdata")

    # 2) Train TCN model
    run_module("models.train_tcn")

    # 3) Evaluate TCN on all symbols
    run_module("models.evaluate_tcn")

    print("\n🎉 Pipeline finished!")
    print("   - Data: data/raw/*.csv")
    print("   - Model: model.pt")
    print("   - Evaluation summary: evaluation_results.csv")

if __name__ == "__main__":
    main()
