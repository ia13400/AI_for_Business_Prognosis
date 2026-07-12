"""Run the June-2025 frozen-origin experiment."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from gold_forecasting.config import load_yaml
from gold_forecasting.experiments import run_holdout
from gold_forecasting.paths import PROCESSED

def main():
    config=load_yaml("experiments.yaml"); parser=argparse.ArgumentParser(); parser.add_argument("--cutoff",default=config["cutoff_date"]); parser.add_argument("--mode",choices=["quick_mode","full_mode"],default=config["mode"]); parser.add_argument("--force-retrain",action="store_true"); parser.add_argument("--dataset",type=Path); args=parser.parse_args()
    dataset=args.dataset or max(PROCESSED.glob("gold_dataset_*.parquet"),key=lambda p:p.stat().st_mtime); run_holdout(pd.read_parquet(dataset),args.cutoff,args.mode,args.force_retrain)
if __name__=="__main__": main()
